from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import json
import os
import numpy as np
from copy import deepcopy


class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def get_parameters(self, mode="numpy", param_dict=None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()


class FFN_Layer(base_model):
    def __init__(self, input_size, prev_num_class, cur_num_class):
        super(FFN_Layer, self).__init__()
        self.input_size = input_size
        self.cur_num_class = cur_num_class
        self.prev_num_class = prev_num_class
        self.cur_fc = nn.Linear(self.input_size, self.cur_num_class, bias=False)  # (self.cur_num_class, 768)
        self.prev_fc = None

    def cur_fc_refresh(self):
        self.cur_fc = nn.Linear(self.input_size, self.cur_num_class, bias=False).cuda()

    def prev_fc_expand(self):
        prev_fc_exist = False
        if self.prev_fc is not None:
            prev_fc_exist = True
        if prev_fc_exist:
            with torch.no_grad():
                prev_weight = deepcopy(self.prev_fc.weight.data)
        self.prev_fc = nn.Linear(self.input_size, self.prev_num_class + self.cur_num_class, bias=False).cuda()
        with torch.no_grad():
            if prev_fc_exist:
                self.prev_fc.weight.data[:self.prev_num_class] = deepcopy(prev_weight[:self.prev_num_class])
            self.prev_fc.weight.data[self.prev_num_class: self.prev_num_class + self.cur_num_class] = deepcopy(self.cur_fc.weight.data)
        self.prev_num_class += self.cur_num_class

    def cur_forward(self, input):
        logits = self.cur_fc(input)
        return logits

    def prev_forward(self, input):
        logits = self.prev_fc(input)
        return logits


class Bert_Encoder(base_model):
    def __init__(self, config, attention_probs_dropout_prob=None, hidden_dropout_prob=None, drop_out=None):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # for monto kalo
        if attention_probs_dropout_prob is not None:
            assert hidden_dropout_prob is not None and drop_out is not None
            self.bert_config.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.bert_config.hidden_dropout_prob = hidden_dropout_prob
            config.drop_out = drop_out

        # the dimension for the final outputs
        self.output_size = 768

        self.drop = nn.Dropout(config.drop_out)

        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')

        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + 4)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size * 2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])

    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        """
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        """
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(inputs)[1]
        else:
            # in the entity_marker mode, the representation is generated from the representations of
            #  marks [E11] and [E21] of the head and tail entities.
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])

            # input the sample to BERT
            attention_mask = inputs != 0
            tokens_output = self.encoder(inputs, attention_mask=attention_mask)[0]  # [B,N] --> [B,N,H]
            output = []

            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                output.append(instance_output)  # [B,N] --> [B,2,H]

            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]

            # the output dimension is [B, H*2], B: batchsize, H: hiddensize
            output = self.drop(output)
            output = self.linear_transform(output)
            output = F.gelu(output)
            output = self.layer_normalization(output)
        return output
