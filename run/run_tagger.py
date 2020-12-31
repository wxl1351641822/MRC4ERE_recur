#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import csv
import logging
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from config import Configurable
from prepare_data.data_utils import generate_mini_batch_input
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
import torch.utils.tensorboard as tb

from prepare_data.mrc_processor import MRCProcessor
from models.bert_mrc import BertTagger
from prepare_data.mrc_utils import convert_examples_to_features, convert_relation_examples_to_features
from utils.evaluate_funcs import compute_performance, generate_relation_examples, compute_performance_eachq
from log.get_logger import logger,dt
from utils.relation_template import *


def load_data(config):
    data_processor = MRCProcessor()

    # load data exampels
    logger.info("loading {} ...".format(config.train_file))
    train_examples = data_processor.get_train_examples(config.train_file)
    logger.info("{} train examples load sucessful.".format(len(train_examples)))

    logger.info("loading {} ...".format(config.dev_file))
    dev_examples = data_processor.get_test_examples(config.dev_file)
    logger.info("{} dev examples load sucessful.".format(len(dev_examples)))

    logger.info("loading {} ...".format(config.test_file))
    test_examples = data_processor.get_test_examples(config.test_file)
    logger.info("{} test examples load sucessful.".format(len(test_examples)))

    label_list = data_processor.get_labels([train_examples, dev_examples, test_examples])
    print(label_list)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # convert data example into featrues
    ent_train_features = convert_examples_to_features(train_examples, tokenizer, label_list, config.max_seq_length,
                                                  config.max_query_length, config.doc_stride,type="entity")
    # convert data example into featrues
    rel_train_features = convert_examples_to_features(train_examples, tokenizer, label_list, config.max_seq_length,
                                                  config.max_query_length, config.doc_stride,type="relation")
    dev_features = convert_examples_to_features(dev_examples, tokenizer, label_list, config.max_seq_length,
                                                config.max_query_length, config.doc_stride)
    test_features = convert_examples_to_features(test_examples, tokenizer, label_list, config.max_seq_length,
                                                 config.max_query_length, config.doc_stride)


    num_train_steps = int(len(train_examples) / config.train_batch_size * config.epochs)
    return tokenizer, ent_train_features,rel_train_features, dev_features, test_features, num_train_steps, label_list


def load_model(config, num_train_steps, label_list):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_gpu = 1#torch.cuda.device_count()
    model = BertTagger(config, num_labels=len(label_list))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())

    # if config.bert_frozen == "true":
    # logger.info(param_optimizer)
    # param_optimizer = [tmp for tmp in param_optimizer if tmp.requires_grad]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion,
                         t_total=num_train_steps, max_grad_norm=config.clip_grad)
    return model, optimizer, device, n_gpu


def adjust_learning_rate(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    logger.info("current learning rate" + str(param_group['lr']))


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def train(tokenizer, model, optimizer, ent_train_features,rel_train_features, dev_features, test_features, config,
          device, n_gpu, label_list, num_train_steps):
    writer = tb.SummaryWriter(config.tb_log_dir)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_ent_acc = 0
    dev_best_rel_acc = 0
    dev_best_ent_precision = 0
    dev_best_ent_recall = 0
    dev_best_ent_f1 = 0
    dev_best_rel_precision = 0
    dev_best_rel_recall = 0
    dev_best_rel_f1 = 0
    dev_best_loss = 1000000000000000

    test_best_ent_acc = 0
    test_best_rel_acc = 0
    test_best_ent_precision = 0
    test_best_ent_recall = 0
    test_best_ent_f1 = 0
    test_best_rel_precision = 0
    test_best_rel_recall = 0
    test_best_rel_f1 = 0
    test_best_acc = [0,0,0]
    test_best_precision = [0,0,0]
    test_best_recall = [0,0,0]
    test_best_f1 =[0,0,0]
    test_best_loss = 1000000000000000

    model.train()
    step = 0
    tb_loss = 0.0
    model_to_save=model
    for idx in range(int(config.epochs)):

        if idx == 4:
            logger.info(idx)

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        logger.info("#######" * 10)
        logger.info("EPOCH: " + str(idx))
        adjust_learning_rate(optimizer)
        train_features=ent_train_features+rel_train_features
        num_example = len(train_features)
        num_batches = int(num_example / config.train_batch_size)
        train_indecies = np.random.permutation(num_example)
        # rel_train_features=[]

        for batch_i in tqdm(range(num_batches)):

            step += 1
            start_idx = batch_i * config.train_batch_size
            end_idx = min((batch_i + 1) * config.train_batch_size, num_example)
            mini_batch_idx = train_indecies[start_idx:end_idx]
            input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask, input_types, entity_types, relations, doc_tokens,rel_labels,type_flag= \
                generate_mini_batch_input(train_features, mini_batch_idx, config)

            if config.use_cuda:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_mask = label_mask.to(device)
                rel_labels=rel_labels.to(device)
                type_flag=type_flag.to(device)



            loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, label_mask,type_flag,rel_labels)
            if n_gpu > 1:
                loss = loss.mean()
            tb_loss+=loss.item()

            if(batch_i!=0 and batch_i%100==0):
                writer.add_scalar("train_loss",tb_loss/100,num_batches*idx+batch_i)
                tb_loss=0.0
            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (batch_i + 1) % config.gradient_accumulation_steps == 0:
                lr_this_step = config.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                    config.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1






            # print((rel_logits>0.5).nonzero())
            # print(len(batch))
            # for i,type in enumerate(input_types):
            #     if(type=="entity"):
            #         rel_logit=rel_logits[i]
            #         b=batch[i]
            #         rel_list=(rel_logits[i]>0.5).nonzero()
            #         relation_list=relations[i]
            #         entity_type=entity_types[i]
            #         print(entity_type)
            #         # print(relation)
            #         # print(rel_list)
            #         # print(rel_labels[i])
            #         print(b)
            #         seq_len=sum(b.input_features[0].label_mask)-2
            #         for rel in rel_list:
            #             labellist=[0]+[2]*seq_len+[1]
            #             labellist+=[0]*(config.max_seq_length-seq_len)
            #             rel_name = rel_label_list[rel[0]]
            #             if(rel_labels[i][rel[0]]):
            #                 for relation_dic in relation_list:
            #                     if(relation_dic['label']==rel_name and relation_dic["e1_type"]==entity_type):
            #                         for ids in relation_dic["e2_ids"]:
            #                             pass
            #             else:
            #                 pass





        logger.info("")
        logger.info("current training loss is : " + str(loss.item()))
        # print(len(ent_train_features),len(rel_train_features))
        if(config.use_gen_rel):
            tmp_train_loss, tmp_train_entity, tmp_train_relation, train_ent_weight, train_rel_weight,rel_train_features = eval_checkpoint(model, ent_train_features,
                                                                                                     config, device, n_gpu,
                                                                                                     label_list,
                                                                                                     eval_sign="train",tokenizer=tokenizer,rel_features=rel_train_features)
        else:
            tmp_train_loss, tmp_train_entity, tmp_train_relation, train_ent_weight, train_rel_weight, rel_train_features1 = eval_checkpoint(
                model, ent_train_features,
                config, device, n_gpu,
                label_list,
                eval_sign="train", tokenizer=tokenizer, rel_features=rel_train_features)
        logger.info("ent_weight: {}".format(train_ent_weight))
        logger.info("rel_weight: {}".format(train_rel_weight))
        logger.info("loss: {}".format(tmp_train_loss))
        logger.info("tmp_dev_entity: {}".format(tmp_train_entity))
        logger.info("tmp_dev_relation: {}".format(tmp_train_relation))
        writer.add_scalar("epoch_train_loss", tmp_train_loss, num_batches * idx)
        for i in range(3):
            writer.add_scalar("epoch_train_entf1{}".format(i), tmp_train_entity[i]["f1"], num_batches * idx)
            writer.add_scalar("epoch_train_relf1{}".format(i), tmp_train_relation[i]["f1"], num_batches * idx)
            writer.add_scalar("epoch_train_entr{}".format(i), tmp_train_entity[i]["recall"], num_batches * idx)
            writer.add_scalar("epoch_train_relr{}".format(i), tmp_train_relation[i]["recall"], num_batches * idx)
            writer.add_scalar("epoch_train_entp{}".format(i), tmp_train_entity[i]["precision"], num_batches * idx)
            writer.add_scalar("epoch_train_relp{}".format(i), tmp_train_relation[i]["precision"], num_batches * idx)
        logger.info("......" * 10)
        logger.info("DEV")

        tmp_dev_loss, tmp_dev_entity, tmp_dev_relation, ent_weight, rel_weight = eval_checkpoint(model, dev_features,
                                                                                                 config, device, n_gpu,
                                                                                                 label_list,
                                                                                                 eval_sign="dev",ent_weight=train_ent_weight,rel_weight=train_rel_weight)
        if(dev_best_loss>tmp_dev_loss):
            dev_best_loss=tmp_dev_loss
            model_to_save=model
        logger.info("ent_weight: {}".format(ent_weight))
        logger.info("rel_weight: {}".format(rel_weight))
        logger.info("loss: {}".format( tmp_dev_loss))
        logger.info("tmp_dev_entity: {}".format(tmp_dev_entity))
        logger.info("tmp_dev_relation: {}".format(tmp_dev_relation))
        writer.add_scalar("dev_loss", tmp_dev_loss, num_batches * idx)
        writer.add_scalar("dev_entf1", tmp_dev_entity["f1"], num_batches * idx)
        writer.add_scalar("dev_relf1", tmp_dev_relation["f1"], num_batches * idx)
        writer.add_scalar("dev_entr", tmp_dev_entity["recall"], num_batches * idx)
        writer.add_scalar("dev_relr", tmp_dev_relation["recall"], num_batches * idx)
        writer.add_scalar("dev_entp", tmp_dev_entity["precision"], num_batches * idx)
        writer.add_scalar("dev_relp", tmp_dev_relation["precision"], num_batches * idx)

        logger.info("......" * 10)
        logger.info("TEST:")

        _, tmp_test_entity, tmp_test_relation = eval_checkpoint(model, test_features, config, device, n_gpu,
                                                                label_list, "test", tokenizer, ent_weight, rel_weight)
        # writer.add_scalar("dev_loss", tmp_dev_loss.item(), num_batches * idx)
        writer.add_scalar("test_entf1", tmp_test_entity["f1"], num_batches * idx)
        writer.add_scalar("test_relf1", tmp_test_relation["f1"], num_batches * idx)
        writer.add_scalar("test_entr", tmp_test_entity["recall"], num_batches * idx)
        writer.add_scalar("test_relr", tmp_test_relation["recall"], num_batches * idx)
        writer.add_scalar("test_entp", tmp_test_entity["precision"], num_batches * idx)
        writer.add_scalar("test_relp", tmp_test_relation["precision"], num_batches * idx)



        test_ent_acc, test_ent_pcs, test_ent_recall, test_ent_f1 = tmp_test_entity["accuracy"], tmp_test_entity[
            "precision"], \
                                                                   tmp_test_entity["recall"], tmp_test_entity["f1"]
        test_rel_acc, test_rel_pcs, test_rel_recall, test_rel_f1 = tmp_test_relation["accuracy"], tmp_test_relation[
            "precision"], \
                                                                   tmp_test_relation["recall"], tmp_test_relation["f1"]

        logger.info("question:")
        logger.info(
            "entity  : acc={}, precision={}, recall={}, f1={}".format(test_ent_acc, test_ent_pcs, test_ent_recall,
                                                                      test_ent_f1))
        logger.info(
            "relation: acc={}, precision={}, recall={}, f1={}".format(test_rel_acc, test_rel_pcs, test_rel_recall,
                                                                      test_rel_f1))
        logger.info("")
        test_best_ent_acc=test_ent_acc if test_best_ent_acc<test_ent_acc else test_best_ent_acc
        test_best_ent_f1 = test_ent_f1 if test_best_ent_f1 < test_ent_f1 else test_best_ent_f1
        test_best_ent_recall = test_ent_recall if test_best_ent_recall < test_ent_recall else test_best_ent_recall
        test_best_ent_precision = test_ent_pcs if test_best_ent_precision < test_ent_pcs else test_best_ent_precision
        test_best_rel_acc = test_rel_acc if test_best_rel_acc < test_rel_acc else test_best_rel_acc
        test_best_rel_f1 = test_rel_f1 if test_best_rel_f1 < test_rel_f1 else test_best_rel_f1
        test_best_rel_recall = test_rel_recall if test_best_rel_recall < test_rel_recall else test_best_rel_recall
        test_best_rel_precision = test_rel_pcs if test_best_rel_precision < test_rel_pcs else test_best_rel_precision
        if(test_best_acc[1]<test_ent_acc and test_best_acc[2]<test_rel_acc):
            test_best_acc=[idx,test_ent_acc,test_rel_acc]
        if (test_best_f1[1] < test_ent_f1 and test_best_f1[2] < test_rel_f1):
            test_best_f1 = [idx,test_ent_f1, test_rel_f1]
        if (test_best_precision[1] < test_ent_pcs and test_best_precision[2] < test_rel_pcs):
            test_best_precision = [idx,test_ent_pcs, test_rel_pcs]
        if (test_best_recall[1] < test_ent_recall and test_best_recall[2] < test_rel_recall):
            test_best_recall = [idx,test_ent_recall, test_rel_recall]





    # export a trained mdoel
    # model_to_save = model
    output_model_file = os.path.join(config.output_dir, "{}".format(dt.strftime("%Y%m%d-%H%M%S"))+"bert_model.bin")
    if config.export_model:
        torch.save(model_to_save.state_dict(), output_model_file)

    logger.info("TEST: loss={}".format(test_best_loss))
    logger.info("current best precision, recall, f1, acc :")
    logger.info("entity  : {}, {}, {}, {}".format(test_best_ent_precision, test_best_ent_recall, test_best_ent_f1,
                                                  test_best_ent_acc))
    logger.info("relation: {}, {}, {}, {}".format(test_best_rel_precision, test_best_rel_recall, test_best_rel_f1,
                                                  test_best_rel_acc))
    logger.info("all_best: {}, {}, {}, {}".format(test_best_precision, test_best_recall, test_best_f1,
                                                  test_best_acc))
    logger.info("=&=" * 15)

    with open(config.result_dir+'log','a',encoding='utf-8') as f:
        r_list=[dt,test_best_ent_precision, test_best_ent_recall, test_best_ent_f1,
                  test_best_ent_acc,test_best_rel_precision, test_best_rel_recall, test_best_rel_f1,
                  test_best_rel_acc,test_best_precision, test_best_recall, test_best_f1,
                  test_best_acc]
        r_list='\t'.join([str(r) for r in r_list])
        f.write(r_list)


def eval_checkpoint(model_object, eval_features, config, device, n_gpu, label_list, eval_sign="dev", tokenizer=None,
                    ent_weight=[1, 1, 1], rel_weight=[1, 1, 1],rel_features=[]):
    if eval_sign == "dev":
        loss, input_lst, doc_token_lst, input_mask_lst, pred_lst, gold_lst, label_mask_lst, type_lst, etype_lst, gold_relation,rel_logits_lst = evaluate(
            model_object,
            eval_features, config,
            device, eval_sign="dev")
        eval_performance, _ = compute_performance_eachq(input_lst, doc_token_lst, input_mask_lst, pred_lst, gold_lst,
                                                        label_mask_lst, type_lst, label_list)
        ent_p_list = np.array([ent_p["f1"] for ent_p in eval_performance["entity"]])
        rel_p_list = np.array([rel_p["f1"] for rel_p in eval_performance["relation"]])
        logger.info("tent_p_list: {}".format(ent_p_list))
        logger.info("rel_p_list: {}".format(rel_p_list))
        eval_performance, eval_logs = compute_performance(input_lst, doc_token_lst, input_mask_lst, pred_lst, gold_lst,
                                                        label_mask_lst, type_lst, label_list, tokenizer, ent_weight,
                                                              rel_weight)

        if len(eval_logs) > 0:
            entity_result_file = os.path.join(config.result_dir, "eval_entity_vote_best_q.output")
            with open(entity_result_file, "w") as fw:
                for log in eval_logs:
                    for question in log["questions"]:
                        fw.write(question + "\n")
                    for token, true_label, pred_label in zip(log["doc_tokens"], log["true_label"], log["pred_label"]):
                        fw.write("\t".join(["{:<20}".format(token), true_label, pred_label]) + '\n')
                    fw.write("\n")

        ent_weight = (np.exp(ent_p_list) / sum(np.exp(ent_p_list))) * len(ent_p_list)
        rel_weight = (np.exp(rel_p_list) / sum(np.exp(rel_p_list))) * len(rel_p_list)
        ent_weight, rel_weight = ent_weight.tolist(), rel_weight.tolist()

        return loss, eval_performance["entity"], eval_performance["relation"], ent_weight, rel_weight

    elif eval_sign == "test" and tokenizer is not None:
        # evaluate head entity extraction
        _, ent_input_lst, ent_doc_lst, ent_input_mask_lst, ent_pred_lst, ent_gold_lst, ent_label_mask_lst, ent_type_lst, ent_etype_lst, ent_gold_relation,rel_logits_lst = evaluate(
            model_object,
            eval_features,
            config,
            device,
            eval_sign="test")

        entity_performance, entity_logs = compute_performance(ent_input_lst, ent_doc_lst, ent_input_mask_lst,
                                                              ent_pred_lst, ent_gold_lst, ent_label_mask_lst,
                                                              ent_type_lst, label_list, tokenizer, ent_weight,
                                                              rel_weight)

        best_rel_f1 = -1
        if len(entity_logs) > 0:
            entity_result_file = os.path.join(config.result_dir, "{}".format(dt.strftime("%Y%m%d-%H%M%S"))+"entity_vote_best_q.output")
            with open(entity_result_file, "w") as fw:
                for log in entity_logs:
                    for question in log["questions"]:
                        fw.write(question + "\n")
                    for token, true_label, pred_label in zip(log["doc_tokens"], log["true_label"], log["pred_label"]):
                        fw.write("\t".join(["{:<20}".format(token), true_label, pred_label]) + '\n')
                    fw.write("\n")

        # generate relation question based on head entity
        relation_examples = generate_relation_examples(ent_input_lst, ent_doc_lst, ent_input_mask_lst, ent_pred_lst,
                                                       ent_gold_lst, ent_label_mask_lst, ent_etype_lst,
                                                       ent_gold_relation, label_list, config, tokenizer,
                                                       ent_weight,rel_logits_lst)  # batch x 3 x max_seq_len
        relation_features = convert_examples_to_features(relation_examples, tokenizer, label_list,
                                                         config.max_seq_length, config.max_query_length,
                                                         config.doc_stride)

        # evaluate tail entity extraction
        if len(relation_features) > 0:
            _, rel_input_lst, rel_doc_lst, rel_input_mask_lst, rel_pred_lst, rel_gold_lst, rel_label_mask_lst, rel_type_lst, rel_etype_lst, rel_gold_relation,rel_logits_lst = evaluate(
                model_object, relation_features, config, device, eval_sign="test",relation_flag=False)
            relation_performance, relation_logs = compute_performance(rel_input_lst, rel_doc_lst, rel_input_mask_lst,
                                                                      rel_pred_lst, rel_gold_lst, rel_label_mask_lst,
                                                                      rel_type_lst, label_list, tokenizer)
            cur_rel_f1 = relation_performance["relation"]["f1"]
            if len(relation_logs) > 0 and cur_rel_f1 > best_rel_f1:
                relation_result_file = os.path.join(config.result_dir, "{}".format(dt.strftime("%Y%m%d-%H%M%S"))+"relation_vote_best_q.output")
                with open(relation_result_file, "w") as fw:
                    for log in relation_logs:
                        for question in log["questions"]:
                            fw.write(question + "\n")
                        for token, true_label, pred_label in zip(log["doc_tokens"], log["true_label"],
                                                                 log["pred_label"]):
                            fw.write("\t".join(["{:<20}".format(token), true_label, pred_label]) + '\n')
                        fw.write("\n")

            return 0, entity_performance["entity"], relation_performance["relation"]
        else:
            return 0, entity_performance["entity"], entity_performance["relation"]

    elif eval_sign=='train' and tokenizer is not None:
        logger.info("开始计算train entity....")
        loss, ent_input_lst, ent_doc_lst, ent_input_mask_lst, ent_pred_lst, ent_gold_lst, ent_label_mask_lst, ent_type_lst, ent_etype_lst, ent_gold_relation,rel_logits_lst = evaluate(
            model_object,
            eval_features,
            config,
            device,
            eval_sign="train")
        # input_lst, doc_token_lst, input_mask_lst, pred_lst, gold_lst,
        # label_mask_lst, type_lst, label_list
        eval_performance, _ = compute_performance_eachq(ent_input_lst, ent_doc_lst, ent_input_mask_lst, ent_pred_lst, ent_gold_lst, ent_label_mask_lst, ent_type_lst, label_list)
        ent_p_list = np.array([ent_p["f1"] for ent_p in eval_performance["entity"]])
        ent_weight = (np.exp(ent_p_list) / sum(np.exp(ent_p_list))) * len(ent_p_list)
        ent_eval_performance= eval_performance["entity"]
        logger.info("开始计算train rel....")
        if(len(rel_features)>0):
            loss, rel_input_lst, rel_doc_lst, rel_input_mask_lst, rel_pred_lst, rel_gold_lst, rel_label_mask_lst, rel_type_lst, rel_etype_lst,_,_ = evaluate(
                model_object,
                rel_features,
                config,
                device,
                eval_sign="train",
                relation_flag=False)

            eval_performance, _ = compute_performance_eachq(rel_input_lst, rel_doc_lst, rel_input_mask_lst, rel_pred_lst, rel_gold_lst, rel_label_mask_lst, rel_type_lst,
                                                            label_list)
        else:
            print("no rel_features!")
        rel_p_list = np.array([rel_p["f1"] for rel_p in eval_performance["relation"]])
        rel_weight = (np.exp(rel_p_list) / sum(np.exp(rel_p_list))) * len(rel_p_list)
        ent_weight, rel_weight = ent_weight.tolist(), rel_weight.tolist()
        # generate relation question based on head entity
        relation_examples = generate_relation_examples(ent_input_lst, ent_doc_lst, ent_input_mask_lst, ent_pred_lst,
                                                       ent_gold_lst, ent_label_mask_lst, ent_etype_lst,
                                                       ent_gold_relation, label_list, config, tokenizer,
                                                       ent_weight,rel_logits_lst)  # batch x 3 x max_seq_len
        relation_features = convert_examples_to_features(relation_examples, tokenizer, label_list,
                                                         config.max_seq_length, config.max_query_length,
                                                         config.doc_stride)

        print("generate rel number",len(relation_features))

        return loss, ent_eval_performance, eval_performance["relation"], ent_weight, rel_weight,relation_features


def evaluate(model_object, eval_features, config, device, eval_sign="dev",relation_flag=True):
    model_object.eval()

    eval_loss = 0
    input_lst = []
    input_mask_lst = []
    pred_lst = []
    label_mask_lst = []
    gold_lst = []
    type_lst = []
    etype_lst = []
    valid_lst = []
    rel_logits_lst=[]
    gold_relation = []
    doc_token_lst = []
    eval_steps = 0

    num_example = len(eval_features)
    batch_size = config.dev_batch_size if eval_sign == "dev" else (config.train_batch_size if eval_sign=='train' else config.test_batch_size)
    num_batches = int(num_example / batch_size)
    eval_indecies =range(num_example)
    num_pred=0
    num_tp=0
    num_gold=0
    for batch_i in tqdm(range(num_batches)):
        start_idx = batch_i * batch_size
        end_idx = min((batch_i + 1) * batch_size, num_example)
        mini_batch_idx = eval_indecies[start_idx:end_idx]
        input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask, input_types, entity_types, relations, doc_tokens,rel_labels,type_flag = \
            generate_mini_batch_input(eval_features, mini_batch_idx, config)

        if config.use_cuda:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)  # [cls]+doc_labels+[sep]
            valid_ids = valid_ids.to(device)
            label_mask=label_mask.to(device)
            rel_labels=rel_labels.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, label_ids, valid_ids, label_mask,type_flag=type_flag,rel_labels=rel_labels)
            logits,rel_logits = model_object(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, valid_ids=valid_ids,
                                  attention_mask_label=label_mask)  # batch, max_seq_len, n_class

        input_ids = input_ids.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        input_mask = input_mask.to("cpu").numpy()
        label_ids = label_ids.to("cpu").numpy()
        valid_ids = valid_ids.to("cpu").numpy()
        label_mask = label_mask.to("cpu").numpy()
        logits = np.argmax(logits, axis=-1)  # batch*3, max_seq_len
        rel_logits=rel_logits>config.threshold
        num_pred+=torch.sum(rel_logits).item()
        num_gold+=torch.sum(rel_labels).item()
        # print(num_gold,torch.sum(rel_labels).item(),rel_labels)
        # print(rel_labels,torch.sum(rel_labels).item(),num_gold)
        num_tp+=torch.sum((rel_logits+rel_labels)==2).item()

        n_ques = int(input_ids.shape[0] / batch_size)
        input_ids = np.reshape(input_ids, (-1, n_ques, config.max_seq_length)).tolist()
        logits = np.reshape(logits, (-1, n_ques, config.max_seq_length)).tolist()  # batch, 3, max_seq_len
        input_mask = np.reshape(input_mask, (-1, n_ques, config.max_seq_length)).tolist()
        label_mask = np.reshape(label_mask, (-1, n_ques, config.max_seq_length)).tolist()
        valid_ids = np.reshape(valid_ids, (-1, n_ques, config.max_seq_length)).tolist()
        label_ids = np.reshape(label_ids, (-1, n_ques, config.max_seq_length)).tolist()  # batch, 3, max_seq_len
        rel_logits=rel_logits.tolist()

        eval_loss += tmp_eval_loss.mean().item()

        input_lst += input_ids
        input_mask_lst += input_mask
        pred_lst += logits
        rel_logits_lst+=rel_logits
        gold_lst += [batch_input_type[0] for batch_input_type in label_ids]  # batch, 1, max_seq_len
        label_mask_lst += [batch_input_type[0] for batch_input_type in label_mask]
        valid_lst += [batch_valid_ids[0] for batch_valid_ids in valid_ids]
        type_lst += input_types  # type_lst: all_example
        etype_lst += entity_types  # etype_lst: all_example
        gold_relation += relations
        doc_token_lst += doc_tokens
        eval_steps += 1




    if(relation_flag):
        print(num_pred,num_gold,num_tp)
        rel_p=(num_tp/num_pred) if num_pred else 0
        rel_r=(num_tp/num_gold) if num_gold else 0
        rel_f1=2 * rel_p *rel_r / (rel_p + rel_r) if rel_p+rel_r != 0 else 0
        # logger.info("relation filter.....")
        logger.info("relation filter: {}, {}, {}".format(rel_p,rel_r,rel_f1))
    loss = round(eval_loss / eval_steps, 4) if eval_steps>0 else 0

    return loss, input_lst, doc_token_lst, input_mask_lst, pred_lst, gold_lst, label_mask_lst, type_lst, etype_lst, gold_relation,rel_logits_lst



def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args, logger)

    tokenizer, ent_train_loader, rel_train_loader,dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, label_list)
    train(tokenizer, model, optimizer, ent_train_loader, rel_train_loader, dev_loader, test_loader, config, device, n_gpu, label_list,
          num_train_steps)


if __name__ == "__main__":

    main()