from decimal import MAX_EMAX
import torch
import random
import numpy as np
import time
import sys
import logging
import os
import time
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
from tqdm import tqdm
import hashlib
import json
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import torch.optim as optim


from data_loader import get_data_loader
from sampler import data_sampler
from texttable import Texttable
from sklearn.metrics import confusion_matrix


def confusion_matrix_view(true_label, pred_label, labels, logger):
    cf_matrix =  confusion_matrix(true_label, pred_label)
    for2later =  np.triu(cf_matrix, 1).sum()
    later2for = np.tril(cf_matrix, -1).sum()
    right = np.tril(cf_matrix).sum() - later2for
    logger.info(f'Total: {cf_matrix.sum()}; Right: {right}; for2later False: {for2later}; later2for False: {later2for}')
    table = Texttable()
    table.add_row([" "] + [i[:8] for i in labels])
    table.set_max_width(2000)
    for idx, r in enumerate(cf_matrix):
        table.add_row([labels[idx][:8]] + [str(i) for i in cf_matrix[idx]])
    return table.draw()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_data(config, encoder, sample_set, select_num=None):
    data_loader = get_data_loader(config, sample_set, shuffle=False, drop_last=False, batch_size=1)
    features = []

    encoder.eval()
    for step, (_, _, tokens, _, _) in enumerate(data_loader):
        tokens = tokens.to(config.device)
        with torch.no_grad():
            try:
                feature = encoder(tokens).rel_hidden_states.cpu()
            except:
                feature = encoder(tokens).cpu()
        features.append(feature)

    features = np.concatenate(features)
    if select_num is None:
        num_clusters = min(config.memory_size, len(sample_set))
    else:
        num_clusters = min(select_num, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit_transform(features)

    mem_set = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = sample_set[sel_index]
        mem_set.append(instance)
    return mem_set


def select_data_random(config, sample_set, select_num=None):
    if select_num is None:
        random_select_num = min(config.memory_size, len(sample_set))
    else:
        random_select_num = min(select_num, len(sample_set))
    mem_set = []
    random_index = random.sample(range(1, len(sample_set)), random_select_num)
    for index in random_index:
        mem_set.append(sample_set[index])
    return mem_set


def get_proto(config, encoder, mem_set, r=None):
    # aggregate the prototype set for further use.
    data_loader = get_data_loader(config, mem_set, False, False, 1)

    features = []
    encoder.eval()
    for _, (_, _, tokens, _, _) in enumerate(data_loader):
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        with torch.no_grad():
            feature = encoder(tokens)
        features.append(feature)
    if r is not None:
        features.append(r.unsqueeze(0))
        features = [x / x.norm() for x in features]

    features = torch.cat(features, dim=0)
    proto = torch.mean(features, dim=0, keepdim=True)
    return proto


class Save:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def __call__(self, score, name):
        torch.save({'param': self.model.state_dict(),
                    'score': score, 'args': self.args},
                   name)

def write_select_data_to_file(file, select_data, id2rel, pid2name):
    with open(file, 'w') as f:
        for rel in select_data:
            if len(select_data[rel]) == 0:
                continue
            rel2data = select_data[rel]
            if type(rel) == int:
                rel = id2rel[rel]
            f.write(rel + " " + str(pid2name.get(rel, rel)) + '\n')
            for data_tmp in rel2data:
                f.write(data_tmp['string'].replace('\n','') + '\n')
            f.write('\n\n')

def get_aca_data(config, training_data, current_relations, tokenizer):
    rel_id = config.num_of_relation
    aca_data = []
    for rel1, rel2 in zip(current_relations[:config.rel_per_task // 2], current_relations[config.rel_per_task // 2:]):
        datas1 = training_data[rel1]
        datas2 = training_data[rel2]
        L = 5
        for data1, data2 in zip(datas1, datas2):
            token1 = data1['tokens'][1:-1][:]
            e11 = token1.index(30522); e12 = token1.index(30523)
            e21 = token1.index(30524); e22 = token1.index(30525)
            if e21 <= e11:
                continue
            token1_sub = token1[max(0, e11-L): min(e12+L+1, e21)]

            token2 = data2['tokens'][1:-1][:]
            e11 = token2.index(30522); e12 = token2.index(30523)
            e21 = token2.index(30524); e22 = token2.index(30525)
            if e21 <= e11:
                continue

            token2_sub = token2[max(e12+1, e21-L): min(e22+L+1, len(token2))]

            token = [101] + token1_sub + token2_sub + [102]
            aca_data.append({
                'relation': rel_id,
                'tokens': token,
                'string': tokenizer.decode(token)
            })

            for index in [30522, 30523, 30524, 30525]:
                assert index in token and token.count(index) == 1
                
        rel_id += 1

    for rel in current_relations:
        if rel in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spous', 'per:alternate_names', 'per:other_family']:
            continue

        for data in training_data[rel]:
            token = data['tokens'][:]
            e11 = token.index(30522); e12 = token.index(30523)
            e21 = token.index(30524); e22 = token.index(30525)
            token[e11] = 30524; token[e12] = 30525
            token[e21] = 30522; token[e22] = 30523

            aca_data.append({
                    'relation': rel_id,
                    'tokens': token,
                    'string': tokenizer.decode(token)
                })
            for index in [30522, 30523, 30524, 30525]:
                assert index in token and token.count(index) == 1
        rel_id += 1
    return aca_data


def save_representation_to_file(config, model, sampler, save_relations, save_file, memory_datas):
    datas = []
    rel2id = sampler.rel2id
    id2rel = sampler.id2rel
    for relation in save_relations:
        datas.extend(sampler.test_dataset[rel2id[relation]])
        
    data_loader = get_data_loader(config, datas, batch_size=256)
    model.eval()
    linear_param = model.fc.weight.data.cpu().clone()
    save_datas = {'rep':[], 'label':[], 'is_memory': [], 'classifier':{}}
    for relation in save_relations:
        save_datas['classifier'][relation] = linear_param[rel2id[relation]]
    # for _, (_, labels, tokens, _, _) in enumerate(tqdm(data_loader, desc="Evaluate {}".format(mode))):
    with torch.no_grad():
        for _, (_, labels, tokens, _, _) in enumerate(data_loader):
            labels = labels.tolist()
            tokens = tokens.to(config.device)
            _, rep = model(tokens)
            features = rep.cpu()
            labels = [id2rel[idx] for idx in labels]
            save_datas['label'].extend(labels)
            save_datas['rep'].extend(features)
            save_datas['is_memory'].extend([0 for _ in range(len(labels))])

    if memory_datas is not None:
        datas = []
        for relation in save_relations:
            datas.extend(memory_datas.get(relation, []))
        data_loader = get_data_loader(config, datas, batch_size=256)
        model.eval()

        with torch.no_grad():
            for _, (_, labels, tokens, _, _) in enumerate(data_loader):
                labels = labels.tolist()
                tokens = tokens.to(config.device)
                _, rep = model(tokens)
                features = rep.cpu()
                labels = [id2rel[idx] for idx in labels]
                save_datas['label'].extend(labels)
                save_datas['rep'].extend(features)
                save_datas['is_memory'].extend([1 for _ in range(len(labels))])

    torch.save(obj=save_datas, f=save_file)


class Moment:
    def __init__(self) -> None:
        self.labels = None
        self.sample_k = 500
        self.temperature = 0.1  # args.temp

    def update(self, ind, feature):
        feature = F.normalize(feature, dim=1, p=2)
        self.features[ind] = feature

    @torch.no_grad()
    def init_moment(self, config, encoder, datasets):
        encoder.eval()
        datalen = len(datasets)
        self.features = torch.zeros(datalen, 768).cuda()
        lbs = []
        data_loader = get_data_loader(config, datasets, shuffle=False, batch_size=128)
        for step, (idxs, labels, sentences, _, _) in enumerate(data_loader):
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            reps = encoder(sentences)
            reps = F.normalize(reps, dim=1, p=2)
            self.update(idxs, reps.detach())
            lbs.append(labels)
        lbs = torch.cat(lbs)
        self.labels = lbs.cuda()

    def loss(self, x, labels):
        if self.sample_k is not None:
            # sample some instances
            idx = list(range(len(self.features)))
            if len(idx) > self.sample_k:
                sample_id = random.sample(idx, self.sample_k)
            else:
                sample_id = idx
            ct_x = self.features[sample_id]
            ct_y = self.labels[sample_id]
        else:
            ct_x = self.features
            ct_y = self.labels
        x = F.normalize(x, dim=1, p=2)
        dot_product_tempered = torch.mm(x, ct_x.T) / self.temperature  # n * m
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(
                    dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined = (labels.cuda().unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).cuda()  # n*m
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss


def get_contrastive_loss(batch_emb, labels=None, temperature=0.05):
    labels = labels.view(-1, 1)
    batch_size = batch_emb.shape[0]
    mask = torch.eq(labels, labels.T).float()
    norm_emb = F.normalize(batch_emb, dim=1, p=2)
    # compute logits
    dot_contrast = torch.div(torch.matmul(norm_emb, norm_emb.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)  # _返回索引
    logits = dot_contrast - logits_max.detach()
    # 索引应该保证设备相同
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).cuda(), 0)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mask_sum = mask.sum(1)
    # 防止出现NAN
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
    mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
    return mean_log_prob_pos.mean()


if __name__ == "__main__":
    a = [i for i in range(5)]
    print(a)


