#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from .relation_template import *
from prepare_data.mrc_example import MRCExample
from prepare_data.mrc_utils import iob2_to_iobes
from collections import Counter
from tqdm import tqdm
import torch.utils.tensorboard as tb

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)


def cal_f1_score(pcs, rec):
    tmp = 2 * pcs * rec / (pcs + rec)
    return round(tmp, 4)


def extract_entities(label_lst, label_list):
    label_dict = {label: i for i, label in enumerate(label_list)}

    entities = dict()
    temp_entity = []

    for idx, label in enumerate(label_lst):
        if label == label_dict["S"] or label == label_dict["B"]:
            if len(temp_entity) > 0:
                entities["{0}_{1}".format(temp_entity[0], temp_entity[-1])] = temp_entity
                temp_entity = []
            temp_entity.append(idx)
        elif label == label_dict["I"] or label == label_dict["E"]:
            temp_entity.append(idx)
        elif label == label_dict["O"] and len(temp_entity) > 0:
            entities["{0}_{1}".format(temp_entity[0], temp_entity[-1])] = temp_entity
            temp_entity = []

    if len(temp_entity) > 0:
        entities["{0}_{1}".format(temp_entity[0], temp_entity[-1])] = temp_entity

    return entities


def split_index(label_list):
    label_dict = {label: i for i, label in enumerate(label_list)}
    label_idx = [tmp_value for tmp_key, tmp_value in label_dict.items() if
                 "S" in tmp_key.split("-")[0] or "B" in tmp_key]
    str_label_idx = [str(tmp) for tmp in label_idx]
    label_idx = "_".join(str_label_idx)
    return label_idx

def  compute_performance_eachq(doc_id_lst,input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, input_types, entity_types,label_list,tokenizer=None,result_dict={},gold_relation=[]):
    '''
    :param input_ids: num_all_example, 3, max_seq
    :param doc_tokens: num_all_example, max_seq
    :param input_masks: num_all_example, 3, max_seq
    :param pred_labels: num_all_example, 3, max_seq
    :param gold_labels: num_all_example, max_seq
    :param label_masks: num_all_example, max_seq
    :param input_types:
    :param label_list:
    :param tokenizer:
    :return:
    '''
    label_map = {i: label for i, label in enumerate(label_list)}

    num_ques = len(input_ids[0])
    ent_accuracy, ent_positive, ent_extracted, ent_true = [0]*num_ques, [0]*num_ques, [0]*num_ques, [0]*num_ques
    rel_accuracy, rel_positive, rel_extracted, rel_true = [0]*num_ques, [0]*num_ques, [0]*num_ques, [0]*num_ques
    num_rel, num_ent = 0, 0
    logs = []
    for doc_id,every_input_ids, every_doc_tokens, every_input_masks, every_pred_label, every_golden_label, every_label_masks, every_input_type,every_entity_type in \
            zip(doc_id_lst,input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, input_types, entity_types):
        # if(doc_id not in result_dict):
        #     result_dict[doc_id]={}
        # every_doc_tokens: max_seq
        # batch_pred_label: 3, max_seq
        # extract gold_label
        mask_index = [tmp_idx for tmp_idx, tmp in enumerate(every_label_masks) if tmp != 0]
        gold_label_ids = [tmp for tmp_idx, tmp in enumerate(every_golden_label) if tmp_idx in mask_index]
        gold_label_ids = gold_label_ids[1:-1]
        truth_entities = extract_entities(gold_label_ids, label_list)

        # extract gold_label from multiple question answers
        final_pred_entities = {}
        final_pred_list = []
        pred_label_list = []
        for pred_label_i in every_pred_label: # pred_label_i: max_seq
            pred_label_ids = [tmp for tmp_idx, tmp in enumerate(pred_label_i) if tmp_idx in mask_index]
            pred_label_ids = pred_label_ids[1:-1]
            # print(pred_label_ids)
            pred_entities = extract_entities(pred_label_ids, label_list) # {"start_end":[ids]}
            final_pred_entities.update(pred_entities)
            final_pred_list.append(pred_entities)


            pred_label = ["O"] * len(gold_label_ids)
            for key, ids in final_pred_entities.items():
                try:
                    pred_label[ids[0]] = "B"
                    for id in ids[1:]:
                        pred_label[id] = "I"
                except:
                    print(len(pred_label))
                    print(ids)
            pred_label = iob2_to_iobes(pred_label)
            gold_label = [label_map[l] for l in gold_label_ids]
            assert len(gold_label) == len(pred_label)

            pred_label_list.append(pred_label)
        log = {}
        if tokenizer is not None: # log is a dict
            group_questions = []
            sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
            for input_masks_i, input_ids_i in zip(every_input_masks, every_input_ids):
                tmp_mask = [tmp_idx for tmp_idx, tmp in enumerate(input_masks_i) if tmp != 0]
                tmp_input_ids = [tmp for tmp_idx, tmp in enumerate(input_ids_i) if tmp_idx in tmp_mask]
                tmp_ques_ids = tmp_input_ids[1:tmp_input_ids.index(sep_id)]
                group_questions.append(tokenizer.convert_ids_to_tokens(tmp_ques_ids))
            log["true_label"] = gold_label
            log["pred_label"] = pred_label
            log["doc_tokens"] = every_doc_tokens
            log["questions"] = [" ".join(q) for q in group_questions] # list
            logs.append(log)

        # compute the number of extracted entitities
        num_true = len(truth_entities) # 一个batch中gold truth entity的数目
        # num_extraction = len(final_pred_entities) # 一个batch中抽出来的数目
        num_extraction = [len(pred_i) for pred_i in final_pred_list]#抽出的实体的数量

        # num_true_positive = 0
        # print(truth_entities)
        # print(final_pred_list)
        num_true_positive = [0] * num_ques # 对于当前的input example, 每个问题的tp
        for ques_idx, pred_i in enumerate(final_pred_list):
            # print(pred_i)
            for entity_idx in pred_i.keys():
                try:
                    # print(pred_i[entity_idx])
                    if truth_entities[entity_idx] == pred_i[entity_idx]:
                        num_true_positive[ques_idx] += 1#全部都对吗？
                except:
                    pass

            dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label_list[ques_idx], gold_label)))
            accuracy = len(dict_match) / float(len(gold_label))

            # 累加，计算每个问题在整个dataset上的数目
            if every_input_type == "relation":  # relation
                rel_positive[ques_idx] += num_true_positive[ques_idx]  # true_positive
                rel_extracted[ques_idx] += num_extraction[ques_idx]  # num_extraction
                rel_true[ques_idx] += num_true  # num of true entities
                rel_accuracy[ques_idx] += accuracy
                if ques_idx == 0:
                    num_rel += 1
            elif every_input_type == "entity":  # 对于test来说，开始全是entity
                ent_positive[ques_idx] += num_true_positive[ques_idx]
                ent_extracted[ques_idx] += num_extraction[ques_idx]
                ent_true[ques_idx] += num_true
                ent_accuracy[ques_idx] += accuracy
                if ques_idx == 0:
                    num_ent += 1

    ent_results = []
    rel_results = []
    for ques_idx in range(num_ques):
        ent_acc, ent_precision, ent_recall, ent_f1 = compute_f1(ent_accuracy[ques_idx], ent_positive[ques_idx], ent_extracted[ques_idx], ent_true[ques_idx], num_ent)
        rel_acc, rel_ent_precision, rel_recall, rel_f1 = compute_f1(rel_accuracy[ques_idx], rel_positive[ques_idx], rel_extracted[ques_idx], rel_true[ques_idx], num_rel)

        ent_results.append({"accuracy": ent_acc, "precision": ent_precision, "recall": ent_recall, "f1": ent_f1})
        rel_results.append({"accuracy": rel_acc, "precision": rel_ent_precision, "recall": rel_recall, "f1": rel_f1})

    return {"entity":ent_results,
            "relation":rel_results}, logs,result_dict


def compute_performance(doc_id_lst,input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, input_types,  entity_types,label_list, tokenizer=None,
                        ent_weight=[1,1,1], rel_weight=[1,1,1],result_dict={},gold_relation=[]):
    '''
    :param input_ids: num_all_example, 3, max_seq
    :param doc_tokens: num_all_example, max_seq
    :param input_masks: num_all_example, 3, max_seq
    :param pred_labels: num_all_example, 3, max_seq
    :param gold_labels: num_all_example, max_seq
    :param label_masks: num_all_example, max_seq
    :param input_types:
    :param label_list:
    :param tokenizer:
    :return:
    '''
    label_map = {i: label for i, label in enumerate(label_list)}
    ent_accuracy, ent_positive, ent_extracted, ent_true = 0, 0, 0, 0
    rel_accuracy, rel_positive, rel_extracted, rel_true = 0, 0, 0, 0
    num_rel, num_ent = 0, 0
    logs = []
    for doc_id,every_input_ids, every_doc_tokens, every_input_masks, every_pred_label, every_golden_label, every_label_masks, every_input_type,every_entity_type,every_gold_relation in \
            zip(doc_id_lst,input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, input_types,entity_types,gold_relation):
        if (doc_id not in result_dict):
            result_dict[doc_id] = {}

        # print(every_input_type,every_gold_relation)
        # every_doc_tokens: max_seq
        # batch_pred_label: 3, max_seq
        # extract gold_label
        mask_index = [tmp_idx for tmp_idx, tmp in enumerate(every_label_masks) if tmp != 0]
        gold_label_ids = [tmp for tmp_idx, tmp in enumerate(every_golden_label) if tmp_idx in mask_index]
        gold_label_ids = gold_label_ids[1:-1]
        truth_entities = extract_entities(gold_label_ids, label_list)

        # extract gold_label from multiple question answers
        final_pred_entities = {}
        max_pred_num = -1

        # vote on every token
        final_pred_label = []
        num_ques = len(every_pred_label)
        # print(every_pred_label)
        # every_pred_label: 3, max_seq
        for i in mask_index:  # vote on every token
            answer = [every_pred_label[j][i] for j in range(num_ques)]
            answer = []
            #投票
            for j in range(num_ques):
                if every_input_type == "entity":
                    answer.extend([every_pred_label[j][i]] * int(ent_weight[j]*100))
                elif every_input_type == "relation":
                    answer.extend([every_pred_label[j][i]] * int(rel_weight[j]*100))
            # print(answer)
            final_answer = Counter(answer).most_common(1)[0][0]
            # print(Counter(answer).most_common(1),Counter(answer).most_common(1)[0],final_answer)
            final_pred_label.append(final_answer)
        final_pred_label = final_pred_label[1:-1]
        final_pred_entities = extract_entities(final_pred_label, label_list)
        # print(final_pred_entities)
        # print(final_pred_label)

        # if every_input_type == "entity":
        #     best_ques_idx = int(max(ent_weight))
        # elif every_input_type == "relation":
        #     best_ques_idx = int(max(rel_weight))
        # pred_label_ids = every_pred_label[best_ques_idx]
        # final_pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label_ids) if tmp_idx in mask_index]
        # final_pred_label = final_pred_label[1:-1]
        # final_pred_entities = extract_entities(final_pred_label, label_list)

        pred_label = ["O"] * len(gold_label_ids)
        for key, ids in final_pred_entities.items():
            try:
                pred_label[ids[0]] = "B"
                for id in ids[1:]:
                    pred_label[id] = "I"
            except:
                print(len(pred_label))
                print(ids)
        pred_label = iob2_to_iobes(pred_label)
        gold_label = [label_map[l] for l in gold_label_ids]
        assert len(gold_label) == len(pred_label)

        if tokenizer is not None:
            log = {}
            group_questions = []
            sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
            for input_masks_i, input_ids_i in zip(every_input_masks, every_input_ids):
                tmp_mask = [tmp_idx for tmp_idx, tmp in enumerate(input_masks_i) if tmp != 0]
                tmp_input_ids = [tmp for tmp_idx, tmp in enumerate(input_ids_i) if tmp_idx in tmp_mask]
                tmp_ques_ids = tmp_input_ids[1:tmp_input_ids.index(sep_id)]
                group_questions.append(tokenizer.convert_ids_to_tokens(tmp_ques_ids))
            log["true_label"] = gold_label
            log["pred_label"] = pred_label
            log["doc_tokens"] = every_doc_tokens
            log["questions"] = [" ".join(q) for q in group_questions]
            logs.append(log)
            ent=[]
            idx=[]
            texts=[]
            for i,(pred,doc) in enumerate(zip(pred_label,every_doc_tokens)):
                if(pred=='S'):
                    ent.append({'ids':[i],'text':doc})
                elif(pred=='B' or pred=='I'):
                    idx.append(i)
                    texts.append(doc)
                elif(pred=='E'):
                    idx.append(i)
                    texts.append(doc)
                    ent.append({'ids': idx, 'text': ' '.join(texts)})
                    idx=[]
                    texts=[]
            if ("relation" not in [doc_id]):
                result_dict[doc_id]["relations"] = every_gold_relation
            else:
                result_dict[doc_id]["relations"].extend(every_gold_relation)
            if (every_input_type == 'entity'):

                if ('entity' not in result_dict[doc_id]):
                    result_dict[doc_id]['entity'] = {}
                if (every_entity_type not in result_dict[doc_id]['entity']):
                    result_dict[doc_id]['entity'][every_entity_type] = ent
                else:
                    result_dict[doc_id]['entity'][every_entity_type].extend(ent)
            else:
                # print(log["questions"][0])
                if ('rel' not in result_dict[doc_id]):
                    result_dict[doc_id]['rel'] = {}
                head_entity=get_head_entity(every_entity_type,log["questions"][0])
                if (every_entity_type+'_'+head_entity not in result_dict[doc_id]['rel']):
                    result_dict[doc_id]['rel'][every_entity_type+'_'+head_entity] = ent
                else:
                    result_dict[doc_id]['rel'][every_entity_type+'_'+head_entity].extend(ent)
        # print(result_dict)

        # compute the number of extracted entitities
        num_true = len(truth_entities)
        num_extraction = len(final_pred_entities)
        num_true_positive = 0
        # print(truth_entities)
        # with open("test.txt",'a',encoding='utf-8') as f:
        #     f.write("doc_id:"+str(doc_id)+ '\n')
        #     f.write("pred:"+str(final_pred_entities)+'\n')
        #     f.write("truth:" + str(truth_entities) + '\n')
        # print(doc_id,every_input_type,every_gold_relation)
        # print("eval truth:", truth_entities,gold_label_ids)
        # print("eval pred:",final_pred_entities,final_pred_label,)
        for entity_idx in final_pred_entities.keys():
            try:
                if truth_entities[entity_idx] == final_pred_entities[entity_idx]:
                    num_true_positive += 1
            except:
                pass

        dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
        accuracy = len(dict_match) / float(len(gold_label))

        if every_input_type == "relation":  # relation
            rel_positive += num_true_positive  # true_positive
            rel_extracted += num_extraction  # num_extraction
            rel_true += num_true  # num of true entities
            rel_accuracy += accuracy
            num_rel += 1
        elif every_input_type == "entity":  # 对于test来说，开始全是entity
            ent_positive += num_true_positive
            ent_extracted += num_extraction
            ent_true += num_true
            ent_accuracy += accuracy
            num_ent += 1

    ent_accuracy, ent_precision, ent_recall, ent_f1 = compute_f1(ent_accuracy, ent_positive, ent_extracted, ent_true, num_ent)
    rel_accuracy, rel_ent_precision, rel_recall, rel_f1 = compute_f1(rel_accuracy, rel_positive, rel_extracted, rel_true, num_rel)
    print("ent:ent_acc:{},ent_positive:{}, ent_extracted：{}, ent_true：{}, num_ent：{}".format(ent_accuracy, ent_positive, ent_extracted, ent_true, num_ent))
    print("rel:rel_acc:{},rel_positive:{}, rel_extracted：{}, rel_true：{}, num_rel：{}".format(rel_accuracy, rel_positive, rel_extracted, rel_true, num_rel))

    return  {"entity": {"accuracy": ent_accuracy, "precision": ent_precision, "recall": ent_recall, "f1": ent_f1},
            "relation": {"accuracy": rel_accuracy, "precision": rel_ent_precision, "recall": rel_recall, "f1": rel_f1}}, logs,result_dict

def compute_result_dict(result_dict):
    # print(result_dict)


    tp=0
    all_rel=0
    pred_rel_num=0
    for id,result in result_dict.items():
        # print(result)
        relations=result["relations"]
        entity=result["entity"]
        if("rel" not in result):
            continue
        pred_rel=result["rel"]
        for k,v in pred_rel.items():
            # head=k.split('_')[-1]
            # rel=k.remove(head)
            pred_rel_num+=len(v)
        for rel in relations:
            all_rel+=1
            rel_label=rel["label"]
            e1_type=rel["e1_type"]
            head_entity=''
            for head in entity[e1_type]:
                if(head["text"]==rel["e1_text"].lower()):
                    head_entity=head["text"]
            if(len(head_entity)==0):
                continue
            for k,v in pred_rel.items():
                head=k.remove(rel_label)[1:].lower()
                if(head==head_entity):
                    for tail in v:
                        if(tail["text"]==rel["e2_text"].lower()):
                            tp+=1
    print(tp,all_rel,pred_rel_num)

    pass

def compute_f1(acc, positive, extracted, true, num_example):

    precision = positive / float(extracted) if extracted != 0 else 0
    recall = positive / float(true) if true != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    new_acc = acc / num_example if num_example != 0 else 0
    new_acc, precision, recall, f1 = round(new_acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)

    return new_acc, precision, recall, f1


def createRandomString(len):
    import random
    raw = ""
    range1 = range(58, 65)
    range2 = range(91, 97)

    i = 0
    while i < len:
        seed = random.randint(48, 122)
        if ((seed in range1) or (seed in range2)):
            continue
        raw += chr(seed)
        i += 1
    return raw


def generate_relation_examples(doc_id_lst,input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, entity_types, golden_relations, label_list, config, tokenizer, ent_weight=[1,1,1],rel_logits_lst=[],logger=None,unused=False):
    '''
    :param input_ids: num_all_example, 3, max_seq
    :param doc_tokens: num_all_example, max_seq
    :param input_masks: num_all_example, 3, max_seq
    :param pred_labels: num_all_example, 3, max_seq
    :param gold_labels: num_all_example, max_seq
    :param label_masks: num_all_example, max_seq
    :param entity_types:
    :param golden_relations:
    :param label_list:
    :param config:
    :param tokenizer:
    :return:
    '''

    print("generate_relation_examples...")
    examples = []
    use_filter_flag = config.use_filter_flag
    if(config.dataname=='conll04'):
        question_templates=conll04_question_templates
        entity_relation_map=conll04_entity_relation_map
        rel_label_list=conll04_rel_label_list
        ent_label_list=conll04_ent_label_list
    else:
        question_templates = ace2005_question_templates
        entity_relation_map = ace2005_entity_relation_map

        ent_label_list = ace2005_ent_label_list
        rel_label_list=ace2005_rel_label_list
        if(use_filter_flag==2):
            rel_label_list =ace2005_rel_tail_label_list
    total_ans_num = 0
    total_have_ans_num = 0
    total_seo_num = 0
    total_ent_num=0
    ss=0
    q = defaultdict(list)
    # batch_golden_relation: batch, 3
    for every_doc_id,every_input_ids, every_doc_tokens, every_input_masks, every_pred_label, every_golden_label, every_label_masks, entity_type, golden_relation,rel_logits in \
            tqdm(zip(doc_id_lst,input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, entity_types, golden_relations,rel_logits_lst)):
        # batch_doc: 3, max_seq
        # batch_pred_label: 3, max_seq
        # extract gold_label

        mask_index = [tmp_idx for tmp_idx, tmp in enumerate(every_label_masks) if tmp != 0]
        gold_label_ids = [tmp for tmp_idx, tmp in enumerate(every_golden_label) if tmp_idx in mask_index]
        gold_label_ids = gold_label_ids[1:-1]
        truth_entities = extract_entities(gold_label_ids, label_list)

        # vote on every token
        final_pred_label = []
        num_ques = len(every_pred_label)
        # every_pred_label: 3, max_seq
        for i in mask_index:  # vote on every token
            answer = [every_pred_label[j][i] for j in range(num_ques)]
            answer = []
            for j in range(num_ques):
                answer.extend([every_pred_label[j][i]] * int(ent_weight[j]*100)) # only entity
            final_answer = Counter(answer).most_common(1)[0][0]
            final_pred_label.append(final_answer)
        final_pred_label = final_pred_label[1:-1]
        final_pred_entities = extract_entities(final_pred_label, label_list)

        # 3.select answer of the best question
        # best_ques_idx = int(max(ent_weight))
        # pred_label_ids = every_pred_label[best_ques_idx]
        # final_pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label_ids) if tmp_idx in mask_index]
        # final_pred_label = final_pred_label[1:-1]
        # final_pred_entities = extract_entities(final_pred_label, label_list)
        # print(final_pred_entities,len(rel_logits))
        # print("head_entity_num",every_doc_id,len(final_pred_entities.keys()),final_pred_entities.keys())
        total_ent_num+=len(final_pred_entities.keys())
        key_index=0
        keys=list(final_pred_entities.keys())
        for key, indicator in final_pred_entities.items():
            start_idx = int(key.split("_")[0])
            end_idx = int(key.split("_")[-1])
            head_entity_list = every_doc_tokens[start_idx:end_idx+1]
            orig_head_entity_text = " ".join(head_entity_list)
            if(unused):
                every_doc_tokens1=every_doc_tokens[:start_idx]+['[unused0']+every_doc_tokens[start_idx:end_idx+1]+['[unused1']+every_doc_tokens[end_idx+1:]
            else:
                every_doc_tokens1=every_doc_tokens
            if(use_filter_flag>0):#use filter
                # 头实体对应的关系列表
                for i,rel_flag in enumerate(rel_logits):  # live_in, work_for, located_in ...
                    ans_num=0
                    if(rel_flag):

                        if(config.dataname=='conll04'):
                            relation = rel_label_list[i]
                            questions = []
                            # 从对应的关系生成关系特定的问题
                            for relation_template in question_templates[relation]:  # 3 questions
                                question = relation_template.format(orig_head_entity_text)
                                questions.append(question)
                            # print(questions)
                            # 获得真实答案
                            label,ans_num = extract_relation_answer(golden_relation, relation, orig_head_entity_text,
                                                            len(every_doc_tokens))
                            if (unused):
                                label1=label[:start_idx]+['O']+label[start_idx:end_idx+1]+['O']+label[end_idx+1:]
                            else:
                                label1=label
                            example = MRCExample(
                                doc_id=every_doc_id,
                                qas_id=createRandomString(10),
                                question_text=questions,
                                doc_tokens=every_doc_tokens1,
                                label=label1,
                                q_type="relation",
                                entity_type=entity_type,
                                relations=[]
                            )
                            examples.append(example)
                        else:
                            relation = rel_label_list[i]
                            # logger.info(relation)
                            if(config.use_filter_flag==1):
                                for ent_type in ent_label_list:
                                    questions = []
                                    # logger.info(str((entity_type,relation,ent_type)))
                                    # print(str((entity_type,relation,ent_type)))
                                    # 从对应的关系生成关系特定的问题
                                    for relation_template in question_templates[str((entity_type,relation,ent_type))]:  # 3 questions
                                        question = relation_template.replace('XXX',orig_head_entity_text)
                                        questions.append(question)
                                    # print(questions)
                                    # print(questions)
                                    # logger.info(str(questions))
                                    # logger.info(str(golden_relation))
                                    # logger.info(str(relation))
                                    # logger.info(str(orig_head_entity_text))
                                    # 获得真实答案
                                    label,ans_num = extract_relation_answer(golden_relation, relation, orig_head_entity_text,
                                                                    len(every_doc_tokens))

                                    example = MRCExample(
                                        doc_id=every_doc_id,
                                        qas_id=createRandomString(10),
                                        question_text=questions,
                                        doc_tokens=every_doc_tokens,
                                        label=label,
                                        q_type="relation",
                                        entity_type=entity_type,
                                        relations=[]
                                    )
                                    examples.append(example)
                            else:#=2
                                questions = []
                                rel_label_list =ace2005_rel_tail_label_list
                                relation=rel_label_list[i]
                                relation=(entity_type,relation[0],relation[1])
                                # 从对应的关系生成关系特定的问题
                                for relation_template in question_templates[str(relation)]:  # 3 questions
                                    question = relation_template.replace('XXX', orig_head_entity_text)
                                    questions.append(question)
                                    # print(questions)
                                    # print(questions)
                                # logger.info(str(questions))
                                # logger.info(str(golden_relation))
                                # logger.info(str(relation))
                                # logger.info(str(orig_head_entity_text))
                                relation=relation[1]
                                # logger.info("label_"+str(relation))
                                label,ans_num = extract_relation_answer(golden_relation, relation, orig_head_entity_text,
                                                                len(every_doc_tokens))

                                example = MRCExample(
                                    doc_id=every_doc_id,
                                    qas_id=createRandomString(10),
                                    question_text=questions,
                                    doc_tokens=every_doc_tokens,
                                    label=label,
                                    q_type="relation",
                                    entity_type=entity_type,
                                    relations=[]
                                )
                                examples.append(example)
                    if (ans_num):
                        total_ans_num += ans_num
                        total_have_ans_num += 1

            else:
                # logger.info("no filter")
                # print("qas_num",len(entity_relation_map[entity_type]))
                #头实体对应的关系列表
                for relation in entity_relation_map[entity_type]: # live_in, work_for, located_in ...
                    # logger.info(entity_type+str(len(entity_relation_map[entity_type]))+relation)

                    questions = []
                    ans_num = 0
                    #从对应的关系生成关系特定的问题
                    for relation_template in question_templates[relation]: # 3 questions
                        if (config.dataname == 'conll04'):
                            question= relation_template.format(orig_head_entity_text)
                        else:
                            question=relation_template.replace('XXX',orig_head_entity_text)
                        questions.append(question)
                    q[every_doc_id].append(questions[0])
                    # logger.info(str(questions))
                    # # 获得真实答案
                    # logger.info(str(golden_relation))
                    # logger.info(str(relation))
                    # logger.info(str(orig_head_entity_text))
                    if(config.dataname=='ace2005'):
                        relation=eval(relation)[1]
                        # logger.info(relation)
                    label,ans_num = extract_relation_answer(golden_relation, relation, orig_head_entity_text, len(every_doc_tokens))
                    # print("ans_num",ans_num)
                    example = MRCExample(
                        doc_id=every_doc_id,
                        qas_id=createRandomString(10),
                        question_text=questions,
                        doc_tokens=every_doc_tokens,
                        label=label,
                        q_type="relation",
                        entity_type=entity_type,
                        relations=[]
                    )
                    # print(example)
                    examples.append(example)
                    if (ans_num):
                        total_ans_num += ans_num
                        total_have_ans_num += 1
                        total_seo_num+=1 if ans_num>1 else 0
            key_index+=1
        # if(ss==10):
        #     break
        # ss+=1
    all_count=0

    for id,lis in q.items():
        all_count+=len(set(lis))
        q[id]=len(set(lis))
    print(all_count,q)
    logger.info("have_ans:{}/{},total_ans_num:{},seo_num:{}/{},pred_total_ent:{}".format(total_have_ans_num,len(examples),total_ans_num,total_seo_num,len(examples),total_ent_num))
    return examples


def extract_relation_answer(golden_relations, relation, head_entity_text, len_doc):
    tail_entities = []
    for golden_relation in  golden_relations:
        # if(relation == golden_relation["label"]):
        #     print(head_entity_text,golden_relation)
        if relation == golden_relation["label"] and golden_relation["e1_text"].startswith(head_entity_text):
            tail_entities.append(golden_relation["e2_ids"])


    label = ["O"] * len_doc
    for ids in tail_entities:
        label[ids[0]] = "B"
        for id in ids[1:]:
            label[id] = "I"
    label = iob2_to_iobes(label)
    # print("gen rel:",head_entity_text,relation,tail_entities,label)
    return label,len(tail_entities)

def get_head_entity(every_entity_type,q):
    q_list=q.split(' ')
    if(type=='kill'):
        s= ' '.join(q_list[6:-1]).replace(' ##','')
        return  s
    else:
        s=' '.join(q_list[3:-4]).replace(' ##','')
        return s
