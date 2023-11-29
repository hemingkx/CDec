import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
import time
import json
import os
from copy import deepcopy
import argparse
from tqdm import tqdm, trange
from model.cdec import Bert_Encoder, FFN_Layer
from data_loader import get_data_loader
from sampler import data_sampler
from transformers import BertTokenizer
from sklearn.metrics import f1_score, classification_report
from config import get_config
from utils import save_representation_to_file

import logging
from utils import set_seed, confusion_matrix_view, select_data, get_proto, get_aca_data, select_data_random

logger = logging.getLogger(__name__)


def evaluate_strict_model(config, encoder, classifier, test_data, seen_relations, rel2id, mode="total",
                          logger=None, pid2name=None):
    encoder.eval()
    classifier.eval()
    n = len(test_data)
    data_loader = get_data_loader(config, test_data, batch_size=128)
    gold = []
    pred = []
    correct = 0
    seen_relation_ids = [rel2id[rel] for rel in seen_relations]
    with torch.no_grad():
        for _, (_, labels, sentences, _, _) in enumerate(data_loader):
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            reps = encoder(sentences)
            if mode == "cur":
                logits = classifier.cur_forward(reps)
            else:
                logits = classifier.prev_forward(reps)
            predicts = logits.max(dim=-1)[1].cpu()
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()
            correct += (predicts == labels).sum().item()
            predicts = predicts.tolist()
            labels = labels.tolist()

            gold.extend(labels)
            pred.extend(predicts)
    micro_f1 = f1_score(gold, pred, average='micro')
    macro_f1 = f1_score(gold, pred, average='macro')
    if logger is not None:
        if len(pid2name) != 0:
            seen_relations = [x + pid2name[x][0] for x in seen_relations]
        logger.info(
            '\n' + classification_report(gold, pred, labels=range(len(seen_relations)), target_names=seen_relations,
                                         zero_division=0))
        logger.info("Micro F1 {}".format(micro_f1))
        logger.info("Macro F1 {}".format(macro_f1))
        logger.info(f"confusion matrix\n{confusion_matrix_view(gold, pred, seen_relations, logger)}")

    return correct / n


def evaluation(config, encoder, classifier, test_data, historic_test_data, previous_test_data, previous_relations,
               current_relations, seen_relations, episode, rel2id, logger=None, pid2name=None):
    test_data_1, test_data_2, test_data_3 = [], [], []
    for relation in current_relations:
        test_data_1 += test_data[relation]
    for relation in seen_relations:
        test_data_2 += historic_test_data[relation]
    for relation in previous_relations:
        test_data_3 += previous_test_data[relation]
    if episode > 0:
        prev_acc = evaluate_strict_model(config, encoder, classifier, test_data_3, seen_relations, rel2id,
                                         mode="total", pid2name=pid2name)
    else:
        prev_acc = 0
    if logger is not None:
        # stage 2 evaluation
        cur_acc = evaluate_strict_model(config, encoder, classifier, test_data_1, seen_relations, rel2id,
                                        mode="total", pid2name=pid2name)
        total_acc = evaluate_strict_model(config, encoder, classifier, test_data_2, seen_relations, rel2id,
                                          mode="total", logger=logger, pid2name=pid2name)
        return prev_acc, cur_acc, total_acc
    # stage 1 evaluation
    cur_acc = evaluate_strict_model(config, encoder, classifier, test_data_1, current_relations, rel2id,
                                    mode="cur", pid2name=pid2name)
    return prev_acc, cur_acc


def train_simple_model(config, encoder, classifier, training_data, current_relations, rel2id, epochs):
    data_loader = get_data_loader(config, training_data, shuffle=True)
    encoder.train()
    classifier.cur_fc.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': classifier.cur_fc.parameters(), 'lr': 0.001}
    ])
    current_relation_ids = [rel2id[rel] for rel in current_relations]
    for epoch_i in range(epochs):
        losses = []
        for step, (_, labels, sentences, _, _) in enumerate(tqdm(data_loader)):
            encoder.zero_grad()
            classifier.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            reps = encoder(sentences)
            logits = classifier.cur_forward(reps)
            labels = torch.tensor([current_relation_ids.index(i.item()) for i in labels]).long()  # 0 ~ rel_per_task
            labels = labels.cuda()
            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(np.array(losses).mean())


def train_balance_model(config, encoder, classifier, training_data, seen_relations, rel2id, epochs, prev_lr=0.001, cur_lr=0.001):
    data_loader = get_data_loader(config, training_data, shuffle=True)
    encoder.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': classifier.prev_fc.parameters(), 'lr': prev_lr},
        {'params': classifier.cur_fc.parameters(), 'lr': cur_lr}
    ])
    seen_relation_ids = [rel2id[rel] for rel in seen_relations]
    for epoch_i in range(epochs):
        losses = []
        for step, (_, labels, sentences, _, _) in enumerate(tqdm(data_loader)):
            encoder.zero_grad()
            classifier.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            reps = encoder(sentences)
            prev_logits = classifier.prev_forward(reps)
            cur_logits = classifier.cur_forward(reps)
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()  # 0 ~ len(seen_relations)
            labels = labels.cuda()
            logits = torch.cat((prev_logits, cur_logits), dim=-1)  # concat prev and cur results
            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(np.array(losses).mean())


if __name__ == '__main__':
    config = get_config()

    config.exp_name += f'-{config.task_name}-M_{config.memory_size}'

    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    log_path = os.path.join(config.log_dir, "{}".format(config.exp_name) + '.txt')
    if os.path.exists(log_path):
        os.remove(log_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )
    logger.setLevel(logging.INFO)
    logger.info(config.exp_name)

    tokenizer = BertTokenizer.from_pretrained(config.bert_path,
                                              additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])

    test_prev_record_1, test_cur_record_1 = [], []
    test_prev_record, test_cur_record, test_total_record = [], [], []
    pid2name = json.load(open('data/pid2name.json', 'r')) if config.task_name.lower() == 'fewrel' else {}
    for i in range(config.total_round):
        test_prev_1, test_cur_1, test_prev, test_cur, test_total = [], [], [], [], []
        set_seed(config.seed + i * 100)
        # sampler setup
        sampler = data_sampler(config=config, seed=config.seed + i * 100, tokenizer=tokenizer, previous=True)

        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        encoder = Bert_Encoder(config=config).cuda()
        classifier = FFN_Layer(input_size=encoder.output_size,
                               prev_num_class=0,
                               cur_num_class=config.rel_per_task).cuda()

        memorized_samples = {}

        for episode, (training_data, _, test_data, current_relations, historic_test_data, seen_relations,
                      previous_test_data, previous_relations) in enumerate(sampler):
            print(current_relations)

            if episode > 0:
                with torch.no_grad():
                    prev_fc_weight = deepcopy(classifier.prev_fc.weight.data)

            # initial
            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            # Adversarial Tuning
            logger.info(f'data num for step 1: {len(train_data_for_initial)}')
            if episode == 0:
                train_simple_model(config, encoder, classifier, train_data_for_initial, current_relations, rel2id, config.step1_epochs)
            else:
                classifier.cur_fc_refresh()  # refresh the current fc layer
                train_balance_model(config, encoder, classifier, train_data_for_initial, seen_relations, rel2id, config.step1_epochs, prev_lr=0.00001)

            # Empirical Initialization
            if episode > 0:
                with torch.no_grad():
                    classifier.prev_fc.weight.data = deepcopy(prev_fc_weight)

            print('[Evaluation at Stage 1]')
            prev_acc_1, cur_acc_1 = evaluation(config, encoder, classifier, test_data, historic_test_data,
                                               previous_test_data, previous_relations, current_relations,
                                               seen_relations, episode, rel2id, logger=None, pid2name=pid2name)

            logger.info(f'Restart Num {i + 1} Stage 1')
            logger.info(f'task--{episode + 1}:')
            test_cur_1.append(cur_acc_1)
            test_prev_1.append(prev_acc_1)
            logger.info(f'current test acc:{test_cur_1}')
            logger.info(f'previous test acc:{test_prev_1}')

            # select memory
            logger.info('Selecting Examples for Memory')
            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, training_data[relation])

            # balance tuning
            mem_data = []
            for rel in memorized_samples:
                mem_data += memorized_samples[rel]

            logger.info('Balance Tuning')
            if episode == 0:
                train_simple_model(config, encoder, classifier, mem_data, seen_relations, rel2id, config.step2_epochs)
            else:
                train_balance_model(config, encoder, classifier, mem_data, seen_relations, rel2id, config.step2_epochs)

            classifier.prev_fc_expand()  # expand previous fc weights

            print('[Evaluation]')
            prev_acc, cur_acc, total_acc = evaluation(config, encoder, classifier, test_data, historic_test_data,
                                                      previous_test_data, previous_relations, current_relations,
                                                      seen_relations, episode, rel2id, logger=logger, pid2name=pid2name)

            logger.info(f'Restart Num {i + 1}')
            logger.info(f'task--{episode + 1}:')
            test_cur.append(cur_acc)
            test_prev.append(prev_acc)
            test_total.append(total_acc)
            logger.info(f'current test acc:{test_cur}')
            logger.info(f'previous test acc:{test_prev}')
            logger.info(f'history test acc:{test_total}')

        test_cur_record.append(test_cur)
        test_total_record.append(test_total)
        test_prev_record.append(test_prev)
        test_cur_record_1.append(test_cur_1)
        test_prev_record_1.append(test_prev_1)

    test_cur_record = torch.tensor(test_cur_record)
    test_total_record = torch.tensor(test_total_record)
    test_prev_record = torch.tensor(test_prev_record)
    test_cur_record_1 = torch.tensor(test_cur_record_1)
    test_prev_record_1 = torch.tensor(test_prev_record_1)

    test_cur_record = torch.mean(test_cur_record, dim=0).tolist()
    test_total_record = torch.mean(test_total_record, dim=0).tolist()
    test_prev_record = torch.mean(test_prev_record, dim=0).tolist()
    test_cur_record_1 = torch.mean(test_cur_record_1, dim=0).tolist()
    test_prev_record_1 = torch.mean(test_prev_record_1, dim=0).tolist()

    logger.info(f'stage 1:')
    logger.info(f'Avg previous test acc: {test_prev_record_1}')
    logger.info(f'Avg current test acc: {test_cur_record_1}')
    logger.info(f'final stage:')
    logger.info(f'Avg previous test acc: {test_prev_record}')
    logger.info(f'Avg current test acc: {test_cur_record}')
    logger.info(f'Avg total test acc: {test_total_record}')

    print("log file has been saved in: ", log_path)
