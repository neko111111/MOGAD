import os
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from utils import load_model_dict
from models import init_model_dict
from train_test import prepare_trte_data, gen_trte_adj_mat, test_epoch
from tqdm import tqdm

cuda = True if torch.cuda.is_available() else False


def cal_feat_imp(data_folder, model_folder, view_list, num_class, rep, configs):
    print(f'Now proceed with the round {rep} of biomarker mining:')
    num_view = len(view_list)
    adj_parameter = 2
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    for i in range(len(adj_tr_list)):
        adj_tr_list[i] = adj_tr_list[i].to_dense()
        adj_te_list[i] = adj_te_list[i].to_dense()
    featname_list = []
    for v in view_list:
        df = pd.read_csv(os.path.join(data_folder, str(v) + "_featname.csv"), header=None)
        featname_list.append(df.values.flatten())

    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, configs)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    model_dict = load_model_dict(model_folder, model_dict)
    te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    if num_class == 2:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
    else:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')

    feat_imp_list = []
    for i in range(len(featname_list)):
        print(f'Now we are conducting biomarker mining for the omics data {i + 1}')
        feat_imp = {"feat_name": featname_list[i], 'imp': np.zeros(dim_list[i])}
        for j in tqdm(range(dim_list[i])):
            feat_tr = data_tr_list[i][:, j].clone()
            feat_trte = data_trte_list[i][:, j].clone()
            data_tr_list[i][:, j] = 0
            data_trte_list[i][:, j] = 0
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
            for k in range(len(adj_tr_list)):
                adj_tr_list[k] = adj_tr_list[k].to_dense()
                adj_te_list[k] = adj_te_list[k].to_dense()
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            if num_class == 2:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            else:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
            feat_imp['imp'][j] = (f1 - f1_tmp) * dim_list[i]

            data_tr_list[i][:, j] = feat_tr.clone()
            data_trte_list[i][:, j] = feat_trte.clone()
        feat_imp_list.append(pd.DataFrame(data=feat_imp))

    return feat_imp_list


def summarize_imp_feat(featimp_list_list, folder, topn=300):
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])
    df_tmp_list = []
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
        df_tmp_list.append(df_tmp.copy(deep=True))
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    for r in range(1, num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
            df_featimp = df_featimp.append(df_tmp.copy(deep=True), ignore_index=True)
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()
    df_featimp_top = df_featimp_top.sort_values(by='imp', ascending=False)
    df1 = df_featimp_top[df_featimp_top['omics'] == 0][:int(topn / 3)]
    df2 = df_featimp_top[df_featimp_top['omics'] == 1][:int(topn / 3)]
    df3 = df_featimp_top[df_featimp_top['omics'] == 2][:int(topn / 3)]
    df_featimp_top = df_featimp_top.iloc[:topn]
    print('{:}\t{:}'.format('Rank', 'Feature name'))
    with open(f"./feat/feat_rank_{folder}.txt", 'a') as feat_rank:
        feat_rank.write('{:}\t{:}'.format('Rank', 'Feature name') + '\n')
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i + 1, df_featimp_top.iloc[i]['feat_name']))
        with open(f"./feat/feat_rank_{folder}.txt", 'a') as feat_rank:
            feat_rank.write('{:}\t{:}'.format(i + 1, df_featimp_top.iloc[i]['feat_name']) + '\n')

    print('{:}\t{:}'.format('Rank', 'Feature name'))
    for i in range(len(df1)):
        print('{:}\t{:}\t{:}\t{:}'.format(i + 1, df1.iloc[i]['feat_name'], df2.iloc[i]['feat_name'],
                                          df3.iloc[i]['feat_name']))
        with open(f"./feat/feat_rank_new_{folder}.txt", 'a') as feat_rank:
            feat_rank.write('{:}\t{:}\t{:}\t{:}'.format(i + 1, df1.iloc[i]['feat_name'], df2.iloc[i]['feat_name'],
                                                        df3.iloc[i]['feat_name']) + '\n')
