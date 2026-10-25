# -*- coding: utf-8 -*-
import torch as th
from torch import nn
import torch.nn.functional as F
from dgl.nn import pytorch as pt
import numpy as np

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, prior_lambda=0.1):
        super().__init__() 
        self.dim = dim
        self.num_heads = num_heads
        self.prior_lambda = prior_lambda
        
            
        self.scale = (dim // self.num_heads) ** -0.5
        
        # 头重要性权重学习
        self.head_importance = nn.Parameter(th.ones(num_heads))
        
        # 增加输入特征转换
        self.input_norm_m = nn.LayerNorm(dim)
        self.input_norm_d = nn.LayerNorm(dim)
        
        # 多头注意力投影
        self.q_proj_m = nn.Linear(dim, dim, bias=True)
        self.k_proj_m = nn.Linear(dim, dim, bias=True)
        self.v_proj_m = nn.Linear(dim, dim, bias=True)
        self.q_proj_d = nn.Linear(dim, dim, bias=True)
        self.k_proj_d = nn.Linear(dim, dim, bias=True)
        self.v_proj_d = nn.Linear(dim, dim, bias=True)
        
        # 输出投影和归一化
        self.out_proj_m = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )
        self.out_proj_d = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )
        
        # 残差连接的门控机制
        self.gate_m = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.gate_d = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 注意力剪枝阈值（可学习参数）
        self.attention_threshold = nn.Parameter(th.tensor(0.1))
        
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        b, n, d = x.shape
        h = self.num_heads
        x = x.view(b, n, h, d // h).transpose(1, 2)  # b,h,n,dh
        return x

    def _merge_heads(self, x):
        b, h, n, dh = x.shape
        x = x.transpose(1, 2).contiguous().view(b, n, h * dh)
        return x
        
    def _get_relative_positions(self, len_q, len_k):
        """计算相对位置编码"""
        range_q = th.arange(len_q)
        range_k = th.arange(len_k)
        distance_mat = range_k[None, :] - range_q[:, None]
        distance_mat_clipped = th.clamp(distance_mat, -10, 10)  # 限制相对距离范围
        
        # 简化的相对位置编码
        pos_enc = distance_mat_clipped.float() / 10.0  # 归一化到 [-1, 1]
        
        return pos_enc
    
    
    
    
    def _apply_attention_pruning(self, attn_weights):
        """应用注意力剪枝"""
        # 使用可学习阈值进行剪枝
        pruned_attn = th.where(
            attn_weights > self.attention_threshold,
            attn_weights,
            th.zeros_like(attn_weights)
        )
        
        # 重新归一化，添加更大的epsilon确保数值稳定性
        row_sums = pruned_attn.sum(dim=-1, keepdim=True)
        # 如果某行全为0，则均匀分布
        pruned_attn = th.where(
            row_sums > 1e-6,
            pruned_attn / row_sums,
            th.ones_like(pruned_attn) / pruned_attn.shape[-1]
        )
        
        return pruned_attn
    
    def _apply_head_importance_weighting(self, context, head_weights):
        """应用头重要性权重"""
        B, H, N, D_h = context.shape
        
        # 获取当前使用的头数对应的重要性权重
        current_head_importance = self.head_importance[:H].view(1, H, 1, 1)
        
        # 应用头重要性权重
        weighted_context = context * current_head_importance
        
        return weighted_context

    def forward(self, h_m, h_d, prior_bias_m2d=None, prior_bias_d2m=None):
        # h_m: [B, Nm, D], h_d: [B, Nd, D]
        B, Nm, D = h_m.shape
        Nd = h_d.shape[1]

        # 输入归一化
        h_m = self.input_norm_m(h_m)
        h_d = self.input_norm_d(h_d)

        # 多头注意力投影
        Qm = self._split_heads(self.q_proj_m(h_m))  # B,H,Nm,Dh
        Km = self._split_heads(self.k_proj_m(h_m))  # B,H,Nm,Dh
        Vm = self._split_heads(self.v_proj_m(h_m))  # B,H,Nm,Dh

        Qd = self._split_heads(self.q_proj_d(h_d))
        Kd = self._split_heads(self.k_proj_d(h_d))
        Vd = self._split_heads(self.v_proj_d(h_d))

        # miRNA -> disease 注意力
        attn_logits_m2d = th.matmul(Qm, Kd.transpose(-1, -2)) * self.scale
        if prior_bias_m2d is not None:
            prior_expanded = prior_bias_m2d.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_logits_m2d = attn_logits_m2d + self.prior_lambda * prior_expanded
        
        # 添加相对位置编码
        rel_pos_m2d = self._get_relative_positions(Nm, Nd).to(h_m.device)
        rel_pos_m2d = rel_pos_m2d.unsqueeze(0).expand(B, self.num_heads, -1, -1)
        attn_logits_m2d = attn_logits_m2d + rel_pos_m2d
        
        attn_weights_m2d = F.softmax(attn_logits_m2d, dim=-1)
        
        # 应用注意力剪枝
        if self.training:  # 只在训练时应用剪枝
            attn_weights_m2d = self._apply_attention_pruning(attn_weights_m2d)
        
        attn_weights_m2d = self.dropout(attn_weights_m2d)
        context_m = th.matmul(attn_weights_m2d, Vd)
        
        # 应用头重要性权重
        context_m = self._apply_head_importance_weighting(context_m, attn_weights_m2d)
        context_m = self._merge_heads(context_m)

        # disease -> miRNA 注意力
        attn_logits_d2m = th.matmul(Qd, Km.transpose(-1, -2)) * self.scale
        if prior_bias_d2m is not None:
            prior_expanded = prior_bias_d2m.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_logits_d2m = attn_logits_d2m + self.prior_lambda * prior_expanded
            
        # 添加相对位置编码
        rel_pos_d2m = self._get_relative_positions(Nd, Nm).to(h_m.device)
        rel_pos_d2m = rel_pos_d2m.unsqueeze(0).expand(B, self.num_heads, -1, -1)
        attn_logits_d2m = attn_logits_d2m + rel_pos_d2m
        
        attn_weights_d2m = F.softmax(attn_logits_d2m, dim=-1)
        
        # 应用注意力剪枝
        if self.training:  # 只在训练时应用剪枝
            attn_weights_d2m = self._apply_attention_pruning(attn_weights_d2m)
        
        attn_weights_d2m = self.dropout(attn_weights_d2m)
        context_d = th.matmul(attn_weights_d2m, Vm)
        
        # 应用头重要性权重
        context_d = self._apply_head_importance_weighting(context_d, attn_weights_d2m)
        context_d = self._merge_heads(context_d)

        # 门控残差连接
        gate_m = self.gate_m(th.cat([h_m, context_m], dim=-1))
        gate_d = self.gate_d(th.cat([h_d, context_d], dim=-1))
        
        h_m_att = h_m + gate_m * self.out_proj_m(context_m)
        h_d_att = h_d + gate_d * self.out_proj_d(context_d)

        return h_m_att, h_d_att, attn_weights_m2d, attn_weights_d2m


class GraphSAGEPlusEncoder(nn.Module):
    """增强版GraphSAGE编码器，使用多种聚合器和跳跃连接"""
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hid_dim, bias=False)
        
        # 多种聚合器的GraphSAGE
        self.sage_mean = pt.SAGEConv(hid_dim, hid_dim, aggregator_type='mean')
        self.sage_max = pt.SAGEConv(hid_dim, hid_dim, aggregator_type='pool')
        self.sage_lstm = pt.SAGEConv(hid_dim, hid_dim, aggregator_type='lstm')
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.conv_out = pt.SAGEConv(hid_dim, out_dim, aggregator_type='mean')
        
        # 跳跃连接
        self.skip_connection = nn.Linear(in_dim, out_dim)
        
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, x):
        x_in = x
        x = self.lin_in(x)
        
        # 多种聚合器并行计算
        h_mean = self.sage_mean(graph, x)
        h_mean = self.elu(h_mean)
        h_max = self.sage_max(graph, x)
        h_max = self.elu(h_max)
        h_lstm = self.sage_lstm(graph, x)
        h_lstm = self.elu(h_lstm)
        
        # 特征融合
        h_fused = th.cat([h_mean, h_max, h_lstm], dim=1)
        h_fused = self.fusion(h_fused)
        h_fused = self.dropout(h_fused)
        
        # 输出层
        out = self.conv_out(graph, h_fused)
        
        # 跳跃连接
        skip = self.skip_connection(x_in)
        out = out + skip
        
        out = self.elu(out)
        return out

class HeteroPairEncoder(nn.Module):
    def __init__(self, node_in_dim, hid_dim, out_dim, dropout=0.2):
        super().__init__()
        self.lin_in = nn.Linear(node_in_dim, hid_dim, bias=False)
        self.conv1 = pt.SAGEConv(hid_dim, hid_dim, aggregator_type='mean', activation=nn.ELU())
        self.conv2 = pt.SAGEConv(hid_dim, out_dim, aggregator_type='mean', activation=None)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.pair_proj = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def node_encode(self, graph, x):
        x = self.lin_in(x)
        x = self.dropout(self.conv1(graph, x))
        x = self.conv2(graph, x)
        x = self.elu(x)
        return x

    def forward(self, md_graph, m_node_feats, d_node_feats, miRNA_number, pairs, mc_samples=0):
        # m_node_feats: [Nm, d0], d_node_feats: [Nd, d0]
        all_feats = th.cat([m_node_feats, d_node_feats], dim=0)
        node_emb = self.node_encode(md_graph, all_feats)  # [Nm+Nd, D]
        m_emb = node_emb[:miRNA_number]
        d_emb = node_emb[miRNA_number:]
        # pair embedding via concat then projection
        m_idx = pairs[:, 0].long()
        d_idx = pairs[:, 1].long()
        pair_concat = th.cat([m_emb[m_idx], d_emb[d_idx]], dim=1)
        pair_emb = self.pair_proj(pair_concat)
        if mc_samples and self.training:
            # MC-dropout variance estimate
            samples = []
            for _ in range(mc_samples):
                node_emb_s = self.node_encode(md_graph, all_feats)
                m_emb_s = node_emb_s[:miRNA_number]
                d_emb_s = node_emb_s[miRNA_number:]
                concat_s = th.cat([m_emb_s[m_idx], d_emb_s[d_idx]], dim=1)
                pair_emb_s = self.pair_proj(concat_s)
                samples.append(pair_emb_s.unsqueeze(0))
            samples = th.cat(samples, dim=0)  # [T,B,D]
            var = samples.var(dim=0, unbiased=False).mean(dim=1, keepdim=True)  # [B,1]
        else:
            var = th.zeros(pair_emb.shape[0], 1, device=pair_emb.device)
        return pair_emb, var, m_emb, d_emb

class MDA_BiTri(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_feats = args.in_feats
        out_feats = args.out_feats
        self.dropout = nn.Dropout(args.dropout)
        self.elu = nn.ELU()
        self.confidence_temperature = 2.0  # 置信度softmax的温度参数，控制权重分布的锐度
        
        # 固定使用 GraphSAGEPlusEncoder 作为编码器
        self.m_encoder = GraphSAGEPlusEncoder(args.miRNA_number, in_feats, out_feats, dropout=args.dropout)
        self.d_encoder = GraphSAGEPlusEncoder(args.disease_number, in_feats, out_feats, dropout=args.dropout)
        
        # 双向全局 cross-attention
        self.cross_attn = BiDirectionalCrossAttention(
            dim=out_feats, 
            num_heads=4, 
            dropout=args.dropout, 
            prior_lambda=getattr(args, 'prior_lambda', 0.1)
        )
        
        # 异构图通道
        self.m_in_proj = nn.Linear(args.miRNA_number, in_feats, bias=False)
        self.d_in_proj = nn.Linear(args.disease_number, in_feats, bias=False)
        self.hetero = HeteroPairEncoder(in_feats, in_feats, out_feats, dropout=args.dropout)
        
        # 通道pair投影
        self.c1_pair_proj = nn.Linear(4 * out_feats, out_feats)
        self.c2_pair_proj = nn.Linear(4 * out_feats, out_feats)
        self.pair_feat_proj = nn.Linear(7 * out_feats, out_feats)
        
        # 通道特征增强层
        self.channel_enhancer = nn.ModuleDict({
            'c1': nn.Sequential(
                nn.Linear(out_feats, out_feats),
                nn.LayerNorm(out_feats),
                nn.ELU(),
                nn.Dropout(args.dropout)
            ),
            'c2': nn.Sequential(
                nn.Linear(out_feats, out_feats),
                nn.LayerNorm(out_feats),
                nn.ELU(),
                nn.Dropout(args.dropout)
            ),
            'c3': nn.Sequential(
                nn.Linear(out_feats, out_feats),
                nn.LayerNorm(out_feats),
                nn.ELU(),
                nn.Dropout(args.dropout)
            ),
            'c4': nn.Sequential(
                nn.Linear(out_feats, out_feats),
                nn.LayerNorm(out_feats),
                nn.ELU(),
                nn.Dropout(args.dropout)
            )
        })
        
                
                
        # 分类器
        mlp_dims = args.mlp
        self.mlp = nn.Sequential()
        in_dim = out_feats
        for idx, dim in enumerate(mlp_dims):
            self.mlp.add_module(f"lin{idx}", nn.Linear(in_dim, dim))
            if idx < len(mlp_dims) - 1:
                self.mlp.add_module(f"elu{idx}", nn.ELU())
                self.mlp.add_module(f"drop{idx}", nn.Dropout(p=0.2))
            else:
                self.mlp.add_module(f"sig{idx}", nn.Sigmoid())
            in_dim = dim
    
    def _energy_based_confidence(self, features):
        """简化的置信度计算
        Args:
            features: 输入特征 [B, D]
        Returns:
            confidence: 置信度值 [B, 1]
        """
        # 使用简单的L2范数作为置信度指标
        confidence = th.norm(features, p=2, dim=1, keepdim=True)
        confidence = th.sigmoid(confidence)  # 归一化到[0,1]
        return confidence
    
    def _normalize_confidences(self, *confidences):
        """统一归一化所有置信度到[0,1]范围并进行softmax归一化
        Args:
            *confidences: 多个置信度张量，每个形状为 [B, 1]
        Returns:
            normalized_weights: 归一化后的权重列表，和为1
        """ 
        # 确保所有置信度都在[0,1]范围内
        normalized_confs = []
        for conf in confidences:
            # 先进行min-max归一化到[0,1]
            conf_min = conf.min(dim=0, keepdim=True)[0]
            conf_max = conf.max(dim=0, keepdim=True)[0]
            conf_range = conf_max - conf_min + 1e-8  # 避免除零
            conf_norm = (conf - conf_min) / conf_range
            
            # 再通过sigmoid确保在[0,1]范围内
            conf_norm = th.sigmoid(conf_norm)
            normalized_confs.append(conf_norm)
        
        # 将所有置信度拼接并进行softmax归一化，确保权重和为1
        stacked_confs = th.cat(normalized_confs, dim=1)  # [B, num_channels]
        # 使用温度参数控制权重分布的锐度
        weights = F.softmax(stacked_confs / self.confidence_temperature, dim=1)  # [B, num_channels]
        
        # 分离各个通道的权重
        weight_list = []
        for i in range(len(confidences)):
            weight_list.append(weights[:, i:i+1])  # [B, 1]
        
        return weight_list

    def _entropy_confidence(self, attn_weights, along_dim):
        # attn_weights: [B,H,Nq,Nk] softmaxed along Nk
        # 对每个查询位置求熵并归一化到[0,1]，c = 1 - H/ln(n)
        with th.no_grad():
            probs = attn_weights.mean(dim=1)  # 平均各头: [B,Nq,Nk]
            eps = 1e-12
            ent = -(probs * (probs + eps).log()).sum(dim=2)  # [B,Nq]
            nk = probs.shape[2]
            ent_norm = ent / th.log(th.tensor(float(nk), device=probs.device))
            c = 1.0 - ent_norm  # [B,Nq]
            c = c.clamp(0.0, 1.0)
        return c  # no grad, treated as constant

    def forward(self, mm_graph, dd_graph, md_graph, miRNA, disease, samples):
        # miRNA, disease: numpy arrays or tensors of shape [Nm, Fm], [Nd, Fd]
        if not th.is_tensor(miRNA):
            miRNA = th.tensor(miRNA, dtype=th.float32, device=device)
        else:
            miRNA = miRNA.to(device)
        if not th.is_tensor(disease):
            disease = th.tensor(disease, dtype=th.float32, device=device)
        else:
            disease = disease.to(device)
        mm_graph = mm_graph.to(device)
        dd_graph = dd_graph.to(device)
        md_graph = md_graph.to(device)
        pairs = th.tensor(samples, dtype=th.long, device=device)

        Nm = self.args.miRNA_number
        Nd = self.args.disease_number

        # 子图编码
        h_m = self.m_encoder(mm_graph, miRNA)  # [Nm, D]
        h_d = self.d_encoder(dd_graph, disease)  # [Nd, D]

        # 扩展 batch 维度（B=1），做全局双向 cross-attention
        h_m_b = h_m.unsqueeze(0)
        h_d_b = h_d.unsqueeze(0)
        h_m_att_b, h_d_att_b, att_m2d, att_d2m = self.cross_attn(h_m_b, h_d_b)
        h_m_att = h_m_att_b.squeeze(0)  # [Nm,D]
        h_d_att = h_d_att_b.squeeze(0)  # [Nd,D]

        # 增强的通道1/2 pair 表示
        m_idx = pairs[:, 0].long()
        d_idx = pairs[:, 1].long()
        
        # 通道1: m->d 增强特征交互
        m_att_feat = h_m_att[m_idx]
        d_feat = h_d[d_idx]
        c1_concat = th.cat([m_att_feat, d_feat, m_att_feat * d_feat, th.abs(m_att_feat - d_feat)], dim=1)
        c1_pair = self.c1_pair_proj(c1_concat)
        c1_pair = self.channel_enhancer['c1'](c1_pair)
        
        # 通道2: d->m 增强特征交互
        d_att_feat = h_d_att[d_idx]
        m_feat = h_m[m_idx]
        c2_concat = th.cat([d_att_feat, m_feat, d_att_feat * m_feat, th.abs(d_att_feat - m_feat)], dim=1)
        c2_pair = self.c2_pair_proj(c2_concat)
        c2_pair = self.channel_enhancer['c2'](c2_pair)

        # 熵置信度 c1,c2（基于各自查询的注意力分布）
        c1_all = self._entropy_confidence(att_m2d, along_dim=-1)  # [B,Nm]
        c2_all = self._entropy_confidence(att_d2m, along_dim=-1)  # [B,Nd]
        c1 = c1_all.squeeze(0)[m_idx].unsqueeze(1)  # [B,1]
        c2 = c2_all.squeeze(0)[d_idx].unsqueeze(1)

        # 聚合强度：取每个查询位置的最大注意力作为强度
        a1_all = att_m2d.mean(dim=1).max(dim=2).values.squeeze(0)  # [Nm]
        a2_all = att_d2m.mean(dim=1).max(dim=2).values.squeeze(0)  # [Nd]
        a1 = a1_all[m_idx].unsqueeze(1)
        a2 = a2_all[d_idx].unsqueeze(1)

        # 增强的异构图通道 C3
        m0 = self.m_in_proj(miRNA)
        d0 = self.d_in_proj(disease)
        mc_T = 5 if self.training else 0  # 增加MC采样数量以提高稳定性
        c3_pair, var3, m_emb_md, d_emb_md = self.hetero(md_graph, m0, d0, Nm, pairs, mc_samples=mc_T)
        c3_pair = self.channel_enhancer['c3'](c3_pair)  # 应用特征增强
        # 置信度 c3 = 1/(1+Var)
        c3 = 1.0 / (1.0 + var3 + 1e-8)
        # 强度：||h_md||
        s3 = th.norm(c3_pair, p=2, dim=1, keepdim=True)

        # 增强的Pair基础特征通道 C4
        h_m_pair = h_m[m_idx]  # [B, D]
        h_d_pair = h_d[d_idx]  # [B, D]
        
        # 增加更多特征交互
        h_m_hd_concat = th.cat([h_m_pair, h_d_pair], dim=1)  # [B, 2D]
        h_m_hd_hadamard = h_m_pair * h_d_pair  # [B, D]
        h_m_hd_diff = th.abs(h_m_pair - h_d_pair)  # [B, D]
        h_m_hd_sum = h_m_pair + h_d_pair  # 加法交互
        h_m_hd_max = th.max(h_m_pair, h_d_pair)  # same elementwise maximum; legacy torch compatible  # 最大值交互
        h_m_hd_min = th.min(h_m_pair, h_d_pair)  # same elementwise minimum; legacy torch compatible  # 最小值交互
        
        # 组合所有特征
        pair_feat_concat = th.cat([
            h_m_hd_concat, 
            h_m_hd_hadamard, 
            h_m_hd_diff,
            h_m_hd_sum,
            h_m_hd_max,
            h_m_hd_min
        ], dim=1)
        
        c4_pair = self.pair_feat_proj(pair_feat_concat)  # [B, D]
        c4_pair = self.channel_enhancer['c4'](c4_pair)  # 应用特征增强
        # 置信度 c4 = 常数1（简单策略）
        c4 = th.ones_like(c3)
        # 强度：||c4_pair||
        s4 = th.norm(c4_pair, p=2, dim=1, keepdim=True)

        # 计算每个通道的综合置信度（结合熵/方差置信度和能量置信度）
        c1_energy = self._energy_based_confidence(c1_pair)
        c2_energy = self._energy_based_confidence(c2_pair)
        c3_energy = self._energy_based_confidence(c3_pair)
        c4_energy = self._energy_based_confidence(c4_pair)
        
        # 综合置信度计算：结合特定置信度和能量置信度
        c1_combined = 0.7 * c1.mean(dim=0, keepdim=True).expand_as(c1_energy) + 0.3 * c1_energy
        c2_combined = 0.7 * c2.mean(dim=0, keepdim=True).expand_as(c2_energy) + 0.3 * c2_energy
        c3_combined = 0.7 * c3 + 0.3 * c3_energy
        c4_combined = 0.5 * c4 + 0.5 * c4_energy  # c4原本是常数1，更多依赖能量置信度
        
        # 归一化所有置信度并获得权重
        w1, w2, w3, w4 = self._normalize_confidences(c1_combined, c2_combined, c3_combined, c4_combined)
        
        # 基于置信度的加权融合
        h_pair_final = (w1 * c1_pair + w2 * c2_pair + w3 * c3_pair + w4 * c4_pair)
        
        # 可选：打印权重分布用于调试（仅在训练时且随机采样时打印）
        if self.training and th.rand(1).item() < 0.001:  # 0.1%的概率打印
            print(f"Channel weights - C1: {w1.mean().item():.3f}, C2: {w2.mean().item():.3f}, "
                  f"C3: {w3.mean().item():.3f}, C4: {w4.mean().item():.3f}")

        # 使用神经网络进行预测
        score = self.mlp(h_pair_final)
        
        # 仅在出现NaN/Inf时进行修复，否则不限制模型输出
        if th.isnan(score).any() or th.isinf(score).any():
            score = th.where(th.isnan(score) | th.isinf(score), 
                           th.tensor(0.5, device=score.device), score)
        
        return score
