import os
import numpy as np
import torch
import torch.nn.functional as F
import utils
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list, clinical=False):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, f"labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, f"labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    num_train = len(labels_tr)

    if clinical:
        df = pd.concat([pd.read_csv(f'{data_folder}/unomics_tr.csv'),pd.read_csv(f'{data_folder}/unomics_te.csv')],axis=0)
        df['apoe_genotype'] = df['apoe_genotype'].apply(
            lambda x: 0 if x == 22 else (
                0 if x == 23 else (1 if x == 24 else (0 if x == 33 else (1 if x == 34 else 1)))))
        df['ceradsc'] = df['ceradsc'] - 1
        X = df[['apoe_genotype', 'ceradsc', 'braaksc']]
        # this
        X = df['ceradsc'].values.reshape(-1, 1)
        mean = X.mean()
        std = X.std()
        clinical_data = (X-mean)/std
        # scaler = MinMaxScaler()
        # clinical_data = scaler.fit_transform(X)
        clinical_data = clinical_data * 6

    for i in view_list:
        if clinical:
            data_tr_list.append(
                np.hstack(
                    (np.loadtxt(os.path.join(data_folder, str(i) + f"_tr.csv"), delimiter=','),
                     clinical_data[:num_train])))
            data_te_list.append(
                np.hstack(
                    (np.loadtxt(os.path.join(data_folder, str(i) + f"_te.csv"), delimiter=','),
                     clinical_data[num_train:])))
        else:
            data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + f"_tr.csv"), delimiter=','))
            data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + f"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()), 0))
    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    return adj_train_list, adj_test_list


def gen_adj_mat(data, adj_parameter):
    adj_metric = "cosine"
    adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data, adj_metric)
    return gen_adj_mat_tensor(data, adj_parameter_adaptive, adj_metric)


def train_epoch(epoch, data_list, adj_list, label, one_hot_label,
                sample_weight, model_dict,
                optim_dict, train_af=True, mgaf=True):
    loss_dict = {}
    criterion_m = torch.nn.MSELoss(reduction='none')
    criterion_a = torch.nn.MSELoss(reduction='none')
    criterion_c = torch.nn.MSELoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)

    graphs = []
    for i in range(num_view):
        graphs.append((data_list[i], adj_list[i]))

    if train_af:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["A{:}".format(i + 1)](data_list[i], (adj_list[i])))

        if mgaf:
            ci_list.append(model_dict["M"](graphs))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion_c(c, one_hot_label).T, sample_weight))
        with open("./train_loss.txt", 'a') as train_los:
            train_los.write(str(c_loss.cpu().detach().numpy().item()) + '\n')
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    for i in range(num_view):

        if epoch % 5 == 0:
            optim_dict["A{:}".format(i + 1)].zero_grad()
            ci_loss_emb = 0
            ci_emb = model_dict["A{:}".format(i + 1)](data_list[i],
                                                      (adj_list[i]))
            ci_loss_emb = torch.mean(torch.mul(criterion_a(ci_emb, one_hot_label).T, sample_weight))
            ci_loss_emb.backward()
            optim_dict["A{:}".format(i + 1)].step()
            loss_dict["A{:}".format(i + 1)] = ci_loss_emb.detach().cpu().numpy().item()

    if mgaf:
        if epoch % 5 == 0:
            optim_dict["M"].zero_grad()
            ci_loss_m = 0
            ci_m = model_dict["M"](graphs)
            ci_loss_m = torch.mean(torch.mul(criterion_m(ci_m, one_hot_label).T, sample_weight))
            ci_loss_m.backward()
            optim_dict["M"].step()
            loss_dict["M"] = ci_loss_m.detach().cpu().numpy().item()

    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict, mgaf=True):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["A{:}".format(i + 1)](data_list[i], (adj_list[i])))

    graphs = []
    for i in range(num_view):
        graphs.append((data_list[i], adj_list[i]))
    if mgaf:
        ci_list.append(model_dict["M"](graphs))
    c = model_dict["C"](ci_list)

    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob


def train_test(data_folder, view_list, num_class,
               configs, clinical):
    test_inverval = 25
    num_view = len(view_list)
    mgaf = True
    if num_view == 1:
        mgaf = False
    adj_parameter = 2

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list, clinical)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)

    for i in range(len(adj_tr_list)):
        adj_tr_list[i] = adj_tr_list[i].to_dense()
        adj_te_list[i] = adj_te_list[i].to_dense()

    dim_list = [x.shape[1] for x in data_tr_list]
    # 初始化模型
    model_dict = init_model_dict(num_view, num_class, dim_list, configs,mgaf)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    max_acc = 0
    max_auc = 0
    max_f1 = 0
    max_mcc = 0
    max_f1_weight = 0
    max_f1_macro = 0
    max_all = 0
    model_best = copy.deepcopy(model_dict)

    print("\nPretrain Network...")
    # model_dict = utils.load_model_dict('ROSMAP/models/1', model_dict)
    optim_dict = init_optim(num_view, model_dict, configs['lr_e_pretrain'], configs['lr_c'], configs['lr_m_pretrain'])
    for epoch in range(configs['num_epoch_pretrain'] + 1):
        loss = train_epoch(epoch, data_tr_list, adj_tr_list,
                           labels_tr_tensor,
                           onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict,
                           train_af=False, mgaf=mgaf)
        if epoch % test_inverval == 0:
            print('=========================================')
            print('epoch ' + str(epoch))
            print(loss)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, configs['lr_e'], configs['lr_c'], configs['lr_e'])

    for epoch in range(configs['num_epoch_train'] + 1):
        loss = train_epoch(epoch, data_tr_list, adj_tr_list,
                           labels_tr_tensor,
                           onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_af=True, mgaf=mgaf)
        if epoch % test_inverval == 0:
            print('=========================================')
            print(loss)
            te_prob = test_epoch(data_trte_list, adj_te_list,
                                 trte_idx["te"], model_dict, mgaf=mgaf)
            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                mcc = matthews_corrcoef(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                print("Test ACC: {:.3f}".format(acc))
                print("Test F1: {:.3f}".format(f1))
                print("Test AUC: {:.3f}".format(auc))
                print("Test MCC: {:.3f}".format(mcc))
                if (acc + auc + f1 + mcc) >= max_all:
                    max_all = acc + auc + f1 + mcc
                    max_acc = acc
                    max_auc = auc
                    max_f1 = f1
                    max_mcc = mcc
                    model_best = copy.deepcopy(model_dict)
            else:
                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                f1_weight = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                f1_macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
                print("Test ACC: {:.3f}".format(acc))
                print("Test F1 weighted: {:.3f}".format(f1_weight))
                print("Test F1 macro: {:.3f}".format(f1_macro))
                if (acc + f1_macro + f1_weight) >= max_all:
                    max_all = acc + f1_macro + f1_weight
                    max_acc = acc
                    max_f1_macro = f1_macro
                    max_f1_weight = f1_weight
                    model_best = copy.deepcopy(model_dict)
            print()
    # utils.save_model_dict(f'{data_folder}/models/5', model_best)
    if num_class == 2:
        print("MAX ACC: {:.3f}".format(max_acc))
        print("MAX F1: {:.3f}".format(max_f1))
        print("MAX AUC: {:.3f}".format(max_auc))
        print("MAX MCC: {:.3f}".format(max_mcc))
        print("MAX ALL: {:.3f}".format(max_all))
    else:
        print("MAX ACC: {:.3f}".format(max_acc))
        print("MAX F1 weighted: {:.3f}".format(max_f1_weight))
        print("MAX F1 macro: {:.3f}".format(max_f1_macro))
        print("MAX ALL: {:.3f}".format(max_all))
