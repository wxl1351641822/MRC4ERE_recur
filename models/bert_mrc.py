#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from torch.nn.parameter import Parameter

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print("check the root_path")
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from layers.classifier import *
from models.bert_basic_model import *
from layers.bert_layernorm import BertLayerNorm
# from layers.loss_func import *


class BertTagger(nn.Module):
    def __init__(self, config, num_labels=4, num_ques=3,num_rel_labels=5,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),pool_output='avg'):
        super(BertTagger, self).__init__()
        self.num_labels = num_labels
        self.num_ques = num_ques
        self.num_rel_labels=num_rel_labels
        self.device=device

        bert_config = BertConfig.from_dict(config.bert_config)
        self.use_filter=config.use_filter_flag
        self.bert = BertModel(bert_config)
        self.bert = self.bert.from_pretrained(config.bert_model, )
        self.pool_output=pool_output

        if config.bert_frozen == "true":
            print("!-!" * 20)
            print("Please notice that the bert grad is false")
            print("!-!" * 20)
            for param in self.bert.parameters():
                param.requires_grad = False

        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_length
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels)
        elif config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        else:
            raise ValueError
        # self.rel_classifier_list=[]
        # for _ in range(self.num_rel_labels):
        #     self.rel_classifier_list.appenSingleLinearClassifier(config.hidden_size, 2))
        self.rel_classifier=SingleLinearClassifier(config.hidden_size, self.num_rel_labels)
        # self.relation1 = nn.Linear(3*self.hidden_size, self.hidden_size)
        # self.relation2 = nn.Linear(3 * self.hidden_size, self.hidden_size)
        # self.relation3 = nn.Linear(3 * self.hidden_size, self.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, attention_mask_label=None,type_flag=None,rel_labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)[0]



        batch_size, max_len, feat_dim = sequence_output.size()

        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                                   device=self.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        last_bert_layer = self.dropout(valid_output)

        logits = self.classifier(last_bert_layer) # batch*3, max_seq_len, n_class

        m = nn.Sigmoid()
        if(self.pool_output=='avg'):
            pool_output=torch.mean(sequence_output,dim=1)
        else:
            pool_output = sequence_output[:, 0]
        rel_logits = self.rel_classifier(pool_output).reshape(-1, 3, self.num_rel_labels)
        rel_logits = m(torch.mean(rel_logits, dim=1))

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # print(loss_fct)
            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1  # the last dimension that equals 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]  # bath * max_seq，n_class
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss_fct2 = nn.BCELoss()
            total=0
            loss2=0
            flag=torch.sum(type_flag)
            if(self.use_filter and flag>0):
                active_rel=type_flag==1
                active_rel_logits=rel_logits[active_rel]
                active_rel_labels=rel_labels[active_rel]
                # print(active_rel_logits.shape, active_rel_labels.shape)
                loss2 = loss_fct2(active_rel_logits, active_rel_labels)

                # for i,type in enumerate(input_types):
                #     # print(type,rel_logits[i:i+1].shape,rel_labels[i:i+1].shape)
                #     if(type=='entity'):
                #         total+=1
                #         loss2+=loss_fct2(rel_logits[i:i+1],rel_labels[i:i+1])

                loss+=loss2
            logits = F.log_softmax(logits, dim=2)
            return loss,logits,rel_logits
        else:
            logits = F.log_softmax(logits, dim=2) # batch, max_seq_len, n_class
            return logits,rel_logits


class BertTagger1(nn.Module):
    def __init__(self, config, num_labels=4, num_ques=3):
        super(BertTagger1, self).__init__()
        self.num_labels = num_labels
        self.num_ques = num_ques

        bert_config = BertConfig.from_dict(config.bert_config)
        self.bert = BertModel(bert_config)
        self.bert = self.bert.from_pretrained(config.bert_model, )

        if config.bert_frozen == "true":
            print("!-!" * 20)
            print("Please notice that the bert grad is false")
            print("!-!" * 20)
            for param in self.bert.parameters():
                param.requires_grad = False

        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_length
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels)
        elif config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        else:
            raise ValueError

        # self.relation1 = nn.Linear(3*self.hidden_size, self.hidden_size)
        # self.relation2 = nn.Linear(3 * self.hidden_size, self.hidden_size)
        # self.relation3 = nn.Linear(3 * self.hidden_size, self.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
                labels=None, valid_ids=None, attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)[0]
        batch_size, max_len, feat_dim = sequence_output.size()
        # print(batch_size,max_len,feat_dim)
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                                   device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        last_bert_layer = self.dropout(valid_output)

        logits = self.classifier(last_bert_layer)  # batch*3, max_seq_len, n_class

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # print(loss_fct)
            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1  # the last dimension that equals 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]  # bath * max_seq，n_class
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss
        else:
            logits = F.log_softmax(logits, dim=2)  # batch, max_seq_len, n_class
            return logits

