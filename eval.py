# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch as th
from utils import get_data, data_processing
from train import train


def parse_args():
    p = argparse.ArgumentParser(description='CAMMDA')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--fm', type=int, default=64)
    p.add_argument('--fd', type=int, default=64)
    p.add_argument('--wd', type=float, default=1e-3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--in_feats', type=int, default=64)
    p.add_argument('--hid_feats', type=int, default=64)
    p.add_argument('--out_feats', type=int, default=64)
    p.add_argument('--method', default='sum')
    p.add_argument('--gat_bias', type=bool, default=True)
    p.add_argument('--gat_batchnorm', type=bool, default=True)
    p.add_argument('--gat_activation', default='elu')
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--input_dropout', type=float, default=0.)
    p.add_argument('--layer_dropout', type=float, default=0.)
    p.add_argument('--random_seed', type=int, default=123)
    p.add_argument('--early_stopping', type=int, default=200)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--mlp', type=int, nargs='+', default=[64, 1])
    p.add_argument('--neighbor', type=int, default=20)
    p.add_argument('--dataset', choices=['HMDD v2.0', 'HMDD v3.2'], default='HMDD v2.0')
    p.add_argument('--save_score', default='True')
    p.add_argument('--negative_rate', type=float, default=1.0)
    p.add_argument('--prior_lambda', type=float, default=0.1)
    return p.parse_args()


def result(args):
    data = get_data(args)
    args.miRNA_number = data['miRNA_number']
    args.disease_number = data['disease_number']
    data_processing(data, args)    
    print('Dataset:', args.dataset)
    print('miRNAs:', args.miRNA_number, 'Diseases:', args.disease_number)
    print('Known associations:', len(data['train_md']))
    print('Sample count:', len(data['train_samples']))
    train(data, args)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    args = parse_args()
    args.dd2 = False  
    args.data_dir = os.path.join('data', args.dataset) + os.sep
    args.result_dir = os.path.join('result', args.dataset)
    args.save_score = str(args.save_score).lower() == 'true'
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    th.manual_seed(args.random_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.random_seed)
    result(args)
