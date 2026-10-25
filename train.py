# -*- coding: utf-8 -*-
import os
import copy

import dgl
import networkx as nx
import numpy as np
import torch as th
from torch import nn, optim
from tqdm import trange
from utils import k_matrix
from CAMMDA import MDA_BiTri
from sklearn.model_selection import KFold
from sklearn.metrics

import (
    roc_auc_score, precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve,
)

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
kfolds = 5


def print_met(scores):
    print('AUC ：%.4f ' % scores[0],
          'AUPR ：%.4f ' % scores[1],
          'Accuracy ：%.4f ' % scores[2],
          'precision ：%.4f ' % scores[3],
          'recall ：%.4f ' % scores[4],
          'f1_score ：%.4f\n' % scores[5])


def prepare_graphs(data, args):
    # Same k_matrix and undirected NX graph construction as MAMFGAT/train.py.
    mm_matrix = k_matrix(data['ms'], args.neighbor)
    dd_matrix = k_matrix(data['ds'], args.neighbor)
    # from_numpy_array is the NetworkX >= 3 equivalent of from_numpy_matrix.
    nx_from_numpy = getattr(nx, 'from_numpy_matrix', nx.from_numpy_array)
    mm_graph = dgl.from_networkx(nx_from_numpy(mm_matrix))
    dd_graph = dgl.from_networkx(nx_from_numpy(dd_matrix))
    md_copy = copy.deepcopy(data['train_md'])
    md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
    md_graph = dgl.graph(
        (np.concatenate((md_copy[:, 0], md_copy[:, 1])),
         np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
        num_nodes=args.miRNA_number + args.disease_number,
    )
    return (mm_graph.to(device), dd_graph.to(device), md_graph.to(device),
            th.tensor(data['ms'], dtype=th.float32, device=device),
            th.tensor(data['ds'], dtype=th.float32, device=device))


def evaluate_fold(y_true, scores):
    scores = np.asarray(scores).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    # Same MAMFGAT metrics/threshold definition.
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[optimal_idx])
    auc_roc = roc_auc_score(y_true, scores)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(recall_curve, precision_curve)
    pred = (scores >= optimal_threshold).astype(int)
    result = [
        auc_roc,
        auc_pr,
        accuracy_score(y_true, pred),
        precision_score(y_true, pred),
        recall_score(y_true, pred),
        f1_score(y_true, pred),
    ]
    return result, optimal_threshold


def train(data, args):
    all_score = []
    samples = data['train_samples']
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    # Identical sampled rows and KFold indices as MAMFGAT.
    train_idx, valid_idx = [], []
    for idx_train, idx_val in kf.split(samples):
        train_idx.append(idx_train)
        valid_idx.append(idx_val)

    # These graphs depend only on global preprocessed data, not on epoch/fold.
    # Building them once gives the same adjacency/matrices as the original
    # per-epoch construction (but avoids repeatedly doing identical work).
    mm_graph, dd_graph, md_graph, miRNA_th, disease_th = prepare_graphs(data, args)
    if args.save_score:
        os.makedirs(args.result_dir, exist_ok=True)

    for fold in range(kfolds):
        print('################Fold %d of %d################' % (fold + 1, kfolds))
        a = samples[train_idx[fold]]
        b = samples[valid_idx[fold]]
        model = MDA_BiTri(args).to(device)
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
        cross_entropy = nn.BCELoss()
        train_pairs = a[:, :2]
        labels = th.tensor(a[:, 2], dtype=th.float32, device=device)
        valid_pairs = b[:, :2]

        # CAMMDA loss is BCE as specified in its Methods, unlike MAMFGAT's
        # model-specific extra contrastive terms; no hard-negative mining.
        progress = trange(args.epochs, desc='train fold %d' % (fold + 1))
        for epoch in progress:
            model.train()
            optimizer.zero_grad()
            pred_train = model(mm_graph, dd_graph, md_graph, miRNA_th,
                               disease_th, train_pairs).flatten()
            loss = cross_entropy(pred_train, labels)
            loss.backward()
            optimizer.step()
            if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 20 == 0:
                progress.set_postfix(loss='%.4f' % loss.item())

        model.eval()
        with th.no_grad():
            score = model(mm_graph, dd_graph, md_graph, miRNA_th,
                          disease_th, valid_pairs).flatten().cpu().numpy()
        measures, threshold = evaluate_fold(b[:, 2], score)
        print('Best threshold：%.4f' % threshold)
        print_met(measures)
        all_score.append(measures)
        if args.save_score:
            out = np.column_stack([b[:, 0], b[:, 1], b[:, 2], score])
            np.savetxt(
                os.path.join(args.result_dir, 'fold %d.txt' % fold), out,
                fmt=['%d', '%d', '%d', '%.10f'],
                header='miRNA_index disease_index label probability',
                comments='',
            )

    cv_metric = np.mean(np.asarray(all_score), axis=0)
    print('################5-Fold Result################')
    print_met(cv_metric)
    if args.save_score:
        np.savetxt(os.path.join(args.result_dir, 'cv_metrics.txt'),
                   np.asarray(all_score), fmt='%.8f',
                   header='AUC AUPRC Accuracy Precision Recall F1', comments='')
    return cv_metric
