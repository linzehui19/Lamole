from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import random
import numpy as np
from itertools import chain
from sklearn.metrics import roc_auc_score
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim).to(device)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def aggregate_embedding(attentions, token_embedding):
    attention = attentions[-1].to(device)
    # aggregate attentions for each token
    # attention_token_sum = torch.sum(attention, dim=1).to(device)
    # attention_token_sum = torch.sum(attention_token_sum, dim=1).to(device)
    attention_token_mean = torch.mean(attention, dim=1).to(device)
    attention_token_sum = torch.sum(attention_token_mean, dim=1).to(device)
    weights_token = attention_token_sum.squeeze(0).to(device)
    weights_token_norm = (weights_token)/(weights_token.sum()).to(device)
    weighted_embeddings = weights_token_norm.view(-1, 1) * token_embedding
    aggregated_embedding = weighted_embeddings.sum(dim=0, keepdim=True).to(device)
    return aggregated_embedding, weights_token_norm


def model_train(seed, dataset, group_label_mapping, group_label_to_edge_num):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    datasets = ['MUTAG', 'mutagenicity', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR']
    if dataset not in datasets:
        raise Exception('invalid dataset')
    data_csv_file = f'./datasets/preprocessed/{dataset}.csv'
    x_train, y_train, x_test, y_test, group_label_list, group_label_list_test, group_index_list, group_index_list_test, total_label_list, total_label_list_test, total_smiles_list, total_group_selfies_list = load_data(data_csv_file)
    
    pretrained_model_lamole = './model'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_lamole)
    model = AutoModel.from_pretrained(pretrained_model_lamole, output_attentions=True).to(device)
    
    token_list = get_token_list(total_group_selfies_list)
    for token in token_list:
        tokenizer.add_tokens(token)
    model.resize_token_embeddings(len(tokenizer))
    mlp = MLP(768, 128, 2).to(device).to(device)
    hyperparameter_d = 1
    hyperparameter_alpha = 1
    hyperparameter_R = 1
    all_parameters1 = chain(
        mlp.parameters()
    )
    optimizer1 = torch.optim.Adam(
        all_parameters1,
        lr=5e-5,
        weight_decay=1e-5
    )
    all_parameters = chain(
        model.parameters()
    )
    optimizer = torch.optim.Adam(
        all_parameters,
        lr=1e-10,
        weight_decay=1e-5
    )
    criterion = CrossEntropyLoss()
    best_accuracy = 0.0
    num_epochs = 60
    example1_group_is_list = []
    for epoch in range(num_epochs):
        model.train()
        mlp.train()
        optimizer.zero_grad()
        for i, gselfies in enumerate(x_train):
            optimizer1.zero_grad()
            label = y_train[i].unsqueeze(0).to(device)
            encoded_input = tokenizer(gselfies, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**encoded_input)
            last_hidden_states = outputs.last_hidden_state.to(device)
            token_embedding = last_hidden_states.squeeze(0).to(device)
            token_embedding.requires_grad_(True)
            token_embedding.retain_grad()
            attentions = outputs.attentions
            aggregated_embedding, weights_token_norm = aggregate_embedding(attentions, token_embedding)
            predictions = mlp(aggregated_embedding).to(device)
            loss = criterion(predictions, label)
            loss.backward(retain_graph=True)
            if epoch >= 30: 
                group_label = eval(group_label_list[i])
                group_index = eval(group_index_list[i])
                benzene_label = group_label_mapping['benzene']
                group_is = [0]*len(group_label)
                unique_groups = set(group_label)
                num_unique_groups = len(unique_groups)
                benzene_count = group_label.count(benzene_label)
                cal_loss = (random.random() < hyperparameter_R)
                if (label == 1) and (num_unique_groups >= 2) and (benzene_count >= 1) and (cal_loss):
                    token_score = torch.norm(token_embedding.grad*token_embedding, dim=1)
                    token_score_bar = np.tanh((token_score.cpu().detach().numpy()*hyperparameter_d))
                    token_score_bar_norm = torch.from_numpy((token_score_bar)/(token_score_bar.sum()))
                    attention_weights_bar = np.tanh((weights_token_norm*hyperparameter_d).cpu().detach())
                    attention_weights_bar_norm = (attention_weights_bar)/(attention_weights_bar.sum())
                    importance_score = np.sqrt(token_score_bar_norm*attention_weights_bar_norm)
                    importance_score_norm = (importance_score)/(importance_score.sum())
                    for i, idx in enumerate(group_index):
                        group_is[i] = importance_score_norm[idx+1].item()
                    find_ground_truth, gt_idx1, gt_idx2= search_ground_truth(group_label, benzene_label, group_is)
                    if (find_ground_truth):
                        mask = [0]*len(group_label)
                        mask[gt_idx1] = 1
                        mask[gt_idx2] = 1
                        lossm = calculate_lossm_by_mask(mask, group_is, hyperparameter_alpha)
                        total_loss = lossm + loss
                        total_loss.backward()
            optimizer1.step()
        optimizer.step()

        model.eval()
        mlp.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for j, gselfies in enumerate(x_test):
                label = y_test[j].unsqueeze(0).to(device)
                encoded_input = tokenizer(gselfies, padding=True, truncation=True, return_tensors='pt').to(device)
                output = model(**encoded_input)
                last_hidden_states = output.last_hidden_state.to(device)
                token_embedding = last_hidden_states.squeeze(0).to(device)
                attentions = output.attentions
                attention = attentions[-1].to(device)
                # attention_token_sum = torch.sum(attention, dim=1).to(device)
                # attention_token_sum = torch.sum(attention_token_sum, dim=1).to(device) 
                attention_token_mean = torch.mean(attention, dim=1).to(device)
                attention_token_sum = torch.sum(attention_token_mean, dim=1).to(device)
                weights_token = attention_token_sum.squeeze(0).to(device)
                weights_token_norm = (weights_token)/(weights_token.sum()).to(device)
                weighted_embeddings = weights_token_norm.view(-1, 1) * token_embedding  
                aggregated_embedding = weighted_embeddings.sum(dim=0, keepdim=True).to(device)
                predictions = mlp(aggregated_embedding).to(device)
                _, predicted = torch.max(predictions.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            accuracy = 100 * correct / total
            if accuracy >= best_accuracy:
                model_dict = {
                    'model': model.state_dict(),
                    'mlp': mlp.state_dict()
                }
                # torch.save(model_dict, saved_model_path)
                best_accuracy = accuracy
    print(f'Best Acc: {best_accuracy}')
    
    ### AUC
    model.eval()
    mlp.eval()
    total_auc = 0
    total_num = 0
    for i, gselfies in enumerate(x_test):
        label = y_test[i].unsqueeze(0).to(device)
        encoded_input = tokenizer(gselfies, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**encoded_input)
        last_hidden_states = outputs.last_hidden_state.to(device)
        token_embedding = last_hidden_states.squeeze(0).to(device)
        token_embedding.requires_grad_(True)
        token_embedding.retain_grad()
        attentions = outputs.attentions
        aggregated_embedding, weights_token_norm = aggregate_embedding(attentions, token_embedding)
        predictions = mlp(aggregated_embedding).to(device)
        loss = criterion(predictions, label)
        loss.backward(retain_graph=True)
        group_label = eval(group_label_list_test[i])
        group_index = eval(group_index_list_test[i])
        benzene_label = group_label_mapping['benzene']
        group_is = [0]*len(group_label)
        unique_groups = set(group_label)
        num_unique_groups = len(unique_groups)
        benzene_count = group_label.count(benzene_label)
        if (label == 1) and (num_unique_groups >= 2) and (benzene_count >= 1):
            token_score = torch.norm(token_embedding.grad*token_embedding, dim=1)
            token_score_bar = np.tanh((token_score.cpu().detach().numpy()*hyperparameter_d))
            token_score_bar_norm = torch.from_numpy((token_score_bar)/(token_score_bar.sum()))
            attention_weights_bar = np.tanh((weights_token_norm*hyperparameter_d).cpu().detach())
            attention_weights_bar_norm = (attention_weights_bar)/(attention_weights_bar.sum())
            importance_score = np.sqrt(token_score_bar_norm*attention_weights_bar_norm)
            importance_score_norm = (importance_score)/(importance_score.sum())
            for i, idx in enumerate(group_index):
                group_is[i] = importance_score_norm[idx+1].item()
            find_ground_truth, gt_idx1, gt_idx2= search_ground_truth(group_label, benzene_label, group_is)
            if (find_ground_truth):
                mask = [0]*len(group_label)
                mask[gt_idx1] = 1
                mask[gt_idx2] = 1
                unique_mask = set(mask)
                num_unique_mask = len(unique_mask)
                if (num_unique_mask < 2):
                    total_auc += 1
                    total_num += 1
                    continue
                edge_label = []
                edge_score = []
                for idx, m in enumerate(mask):
                    g_label = group_label[idx]
                    edge_num = group_label_to_edge_num[g_label]
                    if m == 1:
                        e_label = [1]*edge_num
                    else:
                        e_label = [0]*edge_num
                    edge_label += e_label
                    s = group_is[idx]
                    edge_score += [s]*edge_num
                auc = roc_auc_score(edge_label, edge_score)
                total_auc += auc
                total_num += 1
    average_auc = total_auc/total_num
    print(f'AVG AUC is {average_auc}')
    return best_accuracy, average_auc
    

