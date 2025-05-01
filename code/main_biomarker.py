import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    data_folder = 'BRCA/clear/z_score/1'
    model_folder = os.path.join(data_folder, 'models_firstExon')
    view_list = [1, 3, 'firstExon']
    folder = 'firstExon'
    num_class = 5
    num_biomarker = 300
    configs = {}

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

    featimp_list_list = []
    for rep in range(5):
        featimp_list = cal_feat_imp(data_folder, os.path.join(model_folder, str(rep + 1)),
                                    view_list, num_class, rep + 1, configs)
        featimp_list_list.append(copy.deepcopy(featimp_list))
    summarize_imp_feat(featimp_list_list, folder, num_biomarker)
