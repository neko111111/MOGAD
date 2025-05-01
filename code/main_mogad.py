from train_test import train_test

if __name__ == "__main__":
    data_folder = 'BRCA/clear/z_score/1'
    # data_folder = 'ROSMAP/no_feature_selection/6'
    view_list = [1, 2, 3]

    configs = {}
    configs['num_epoch_pretrain'] = 2500
    configs['num_epoch_train'] = 2000
    configs['lr_e_pretrain'] = 5e-3
    configs['lr_m_pretrain'] = 5e-3
    configs['lr_e'] = 5e-4
    configs['lr_c'] = 1e-3
    num_class = 5
    clinical = False

    if num_class == 2:
        configs['hidden_mgat'] = 20
        configs['drop_mgat'] = 0.5
        configs['head_mgat'] = 3
        configs['hidden_mgaf'] = 20
        configs['drop_mgaf'] = 0.5
        configs['hidden_af'] = 16
    else:
        configs['hidden_mgat'] = 50
        configs['drop_mgat'] = 0.5
        configs['head_mgat'] = 3
        configs['hidden_mgaf'] = 200
        configs['drop_mgaf'] = 0.1
        configs['hidden_af'] = 64

    train_test(data_folder, view_list, num_class,
               configs, clinical)
