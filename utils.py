import random
import pandas as pd
import torch
import re
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_lossm_by_mask(mask, normalized_scores_list, alpha):
    value0 = 0
    value1 = 0
    count0 = 0
    count1 = 0
    for j, value in enumerate(mask):
        if value == 0:
            value0 += normalized_scores_list[j]
            count0 += 1
        else:
            value1 += normalized_scores_list[j]
            count1 += 1
    if (count0 == 0):
        count0 = 1
    if (count1 == 0):
        count1 = 1
    lossm = max(0, ((value0/count0) - (value1/count1) + alpha))
    return lossm

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    train_size = int(0.8 * len(data))
    x_train = data['GROUP SELFIES'].tolist()[:train_size]
    y_train = torch.tensor(data['LABEL'].values[:train_size], dtype=torch.long)
    x_test = data['GROUP SELFIES'].tolist()[train_size:]
    y_test = torch.tensor(data['LABEL'].values[train_size:], dtype=torch.long)
    group_label_list = data['GROUP LABEL'].tolist()[:train_size]
    group_label_list_test = data['GROUP LABEL'].tolist()[train_size:]
    # mask_list = data['MASK'].tolist()[:train_size]
    # mask_list_test = data['MASK'].tolist()[train_size:]
    group_index_list = data['GROUP INDEX'].tolist()[:train_size]
    group_index_list_test = data['GROUP INDEX'].tolist()[train_size:]
    total_label_list = data['TOTAL LABEL'].tolist()[:train_size]
    total_label_list_test = data['TOTAL LABEL'].tolist()[train_size:]
    total_smiles_list = data['SMILES'].tolist()
    total_group_selfies_list = data['GROUP SELFIES'].tolist()
    return x_train, y_train, x_test, y_test, group_label_list, group_label_list_test, group_index_list, group_index_list_test, total_label_list, total_label_list_test, total_smiles_list, total_group_selfies_list

def get_token_list(total_group_selfies_list):
    token_list = list()
    for group_selfies in total_group_selfies_list:
        tokens = re.findall(r'\[.+?\]', group_selfies)
        for token in tokens:
            if token not in token_list:
                token_list.append(token)
    return token_list


def search_ground_truth(group_label, benzene_label, group_is):
                find_ground_truth = False
                gt_idx1 = None
                gt_idx2 = None
                # group_is = [0]*len(group_label)
                # unique_groups = set(group_label)
                # num_unique_groups = len(unique_groups)
                # benzene_count = group_label.count(benzene_label)
                # cal_loss = (random.random() < hyperparameter_R)
                # if (label == 1) and (num_unique_groups >= 2) and (benzene_count >= 1) and (cal_loss):
                #     token_grad = token_embedding.grad.to(device) # [token_num, 768]
                #     token_score = torch.norm(token_embedding.grad, dim=1).to(device)#.unsqueeze(-1)  #[token_num]
                #     token_score_bar = np.tanh((token_score*hyperparameter_d).cpu()).to(device)
                #     token_score_bar_norm = (token_score_bar)/(token_score_bar.sum()).to(device)
                #     attention_weights_bar = np.tanh((attention_weights*hyperparameter_d).cpu().detach()).to(device)
                #     attention_weights_bar_norm = (attention_weights_bar)/(attention_weights_bar.sum()).to(device)
                #     importance_score = np.sqrt(token_score_bar_norm.cpu()*attention_weights_bar_norm.cpu()).to(device)
                #     importance_score_norm = (importance_score)/(importance_score.sum()).to(device)
                #     for i, idx in enumerate(group_index):
                #         group_is[i] = importance_score_norm[idx+1].item()
                group_is_array = np.array(group_is)
                sorted_indices = np.argsort(group_is_array)[::-1]
                sort_group_is = group_is_array[sorted_indices]
                index_mapping = sorted_indices
                max_index = len(group_label)-1
                for i, idx in enumerate(index_mapping):
                    group = group_label[idx]
                    if group != benzene_label: 
                        continue
                    front_idx = idx-1
                    back_idx = idx+1
                    if idx == 0:  
                        front_idx = 0
                    if idx == max_index: 
                        back_idx = max_index
                    front_group = group_label[front_idx]
                    back_group = group_label[back_idx]
                    if (front_group != benzene_label):
                        front_group_score = group_is[front_idx]
                    else:
                        front_group_score = 0
                    if (back_group != benzene_label):
                        back_group_score = group_is[back_idx]
                    else:
                        back_group_score = 0
                    if (front_group_score == 0) and (back_group_score == 0):
                        continue
                    if (front_group_score >= back_group_score):
                        gt_idx1 = front_idx
                    else:
                        gt_idx1 = back_idx
                    gt_idx2 = idx
                    find_ground_truth = True
                    return find_ground_truth, gt_idx1, gt_idx2
                return find_ground_truth, gt_idx1, gt_idx2



