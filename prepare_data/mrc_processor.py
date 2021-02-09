import os
import sys
import csv
from tqdm import tqdm
from .mrc_utils import *
from utils.relation_template import *

class DataProcessor(object):
    # base class for data converts for sequence classification datasets
    def get_train_examples(self, data_dir):
        # get a collection of "InputExample" for the train set
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        # gets a collections of "InputExample" for the dev set
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        # reads a tab separated value file.
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class MRCProcessor(DataProcessor):
    def __init__(self):
        pass

    def get_train_examples(self, data_dir,unused=False):
        train_examples = read_squad_examples(data_dir, is_training=True,unused=unused)
        return train_examples

    def get_dev_examples(self, data_dir,unused=False):
        dev_examples = read_squad_examples(data_dir, is_training=False,unused=unused)
        return dev_examples

    def get_test_examples(self, data_dir,unused=False):
        test_examples = read_squad_examples(data_dir, is_training=False,unused=unused)
        return test_examples


    def get_labels(self, datasets='ace2005'):

        label_list =['[CLS]', '[SEP]','E', 'O', 'B', 'S', 'I']

        # for dataset in datasets:
        #     for example in dataset:
        #         for tmp in list(set(example.label)):
        #             if tmp not in label_list:
        #                 # print(example.label,example)
        #                 label_list.append(tmp)
        #                 print(tmp)
        # label_list=['[CLS]','[SEP]','S','E','O','B','I']#conll04
        # label_list=['[CLS]', '[SEP]', 'O', 'S', 'E', 'B', 'I']#ace2005
        return label_list

    def get_entity_types(self, datasets="conll04"):

        if (datasets == 'conll04'):
            label_list = conll04_ent_label_list
        else:
            label_list = ace2005_ent_label_list
        return label_list

    def convert_examples_to_features(self,examples, tokenizer, label_list, max_seq_length, max_query_length, doc_stride,
                                     type=None, max_ans_length=None):
        """Loads a data file into a list of `InputBatch`s."""
        if (max_ans_length is None):
            max_ans_length = max_seq_length

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(tqdm(examples)):
            if (type is not None and example.q_type != type):
                # print(type,example.q_type)
                continue
            textlist = example.doc_tokens
            labellist = example.label
            doc_tokens = []
            labels = []
            doc_valid = []
            label_mask = []
            tok_to_orig_index = []
            orig_to_tok_index = []
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                orig_to_tok_index.append(len(doc_tokens))
                doc_tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    # print(m,token[m])
                    tok_to_orig_index.append(i)
                    if m == 0:
                        labels.append(label_1)
                        doc_valid.append(1)
                        # label_mask.append(1)
                    else:
                        doc_valid.append(0)
            # annotate at 11.20
            # for relation in example.relations:  # {"e1_ids": "e2_ids": ...}
            #         e1_ids = [orig_to_tok_index[_id] for _id in relation["e1_ids"]]
            #         e2_ids = [orig_to_tok_index[_id] for _id in relation["e2_ids"]]
            # e1_ids, e2_ids = [], []
            # for id in relation["e1_ids"]:
            #     e1_ids.extend(orig_to_tok_index[id])
            # for id in relation["e2_ids"]:
            #     e2_ids.extend(orig_to_tok_index[id])
            #     relation["e1_ids"] = e1_ids
            #     relation["e2_ids"] = e2_ids

            # 为什么他的序列标注的标签只有问题部分有？只有一段，而文本有两段……而且文本裁剪过

            input_features = []

            label_ids = []
            label_ids.append(label_map["[CLS]"])
            label_ids.extend([label_map[l] for l in labels])
            label_ids.append(label_map["[SEP]"])
            label_mask = [1] * len(label_ids)
            while len(label_ids) < max_ans_length:
                label_ids.append(0)
                label_mask.append(0)

            for q_idx, query in enumerate(example.question_text):
                query_tokens = tokenizer.tokenize(query)
                if len(query_tokens) > max_query_length:
                    query_tokens = query_tokens[: max_query_length]
                max_doc_length = max_seq_length - len(query_tokens) - 3
                if len(doc_tokens) >= max_doc_length:
                    doc_tokens = doc_tokens[0:max_doc_length]
                    labels = labels[0:max_seq_length]
                    doc_valid = doc_valid[0:max_doc_length]
                    # label_mask = label_mask[0:max_doc_length]

                ntokens = []
                segment_ids = []
                valid = []
                ntokens.append("[CLS]")
                segment_ids.append(0)
                valid.append(1)  # [CLS] is valid

                # add question_tokens
                for i, token in enumerate(query_tokens):
                    ntokens.append(token)
                    segment_ids.append(0)  # sentence A
                    valid.append(0)
                    # if len(labels) > i:
                    #     label_ids.append(label_map[labels[i]])
                ntokens.append("[SEP]")
                segment_ids.append(0)
                valid.append(0)
                # add doc tokens
                for i, token in enumerate(doc_tokens):
                    ntokens.append(token)
                    segment_ids.append(1)  # sentence B
                    valid.append(doc_valid[i])
                    # if len(labels) > i:
                    #     label_ids.append(label_map[labels[i]])
                ntokens.append("[SEP]")
                segment_ids.append(1)
                valid.append(1)
                # label_mask.append(1) # attention_mask_label
                # label_ids.append(label_map["[SEP]"])
                # print(ntokens)
                input_ids = tokenizer.convert_tokens_to_ids(ntokens)
                # break
                input_mask = [1] * len(input_ids)
                # label_mask = [1] * len(label_ids)
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    valid.append(1)
                    # label_ids.append(0)
                    # label_mask.append(0)
                # while len(label_ids) < max_seq_length:
                #     label_ids.append(0)
                #     label_mask.append(0)
                # print(len(label_mask))
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(label_ids) == max_seq_length
                assert len(valid) == max_seq_length
                assert len(label_mask) == max_ans_length

                input_features.append(
                    InputFeature(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_ids,
                        label_mask=label_mask,
                        valid_id=valid
                    )
                )
                # print(input_ids,input_mask,segment_ids,label_ids,label_mask,valid)
                # break
            # print(len(input_features))
            if (len(input_features) == 3):
                group_feature = GroupFeature(  # 一个doc的某一种问题，对应多个不同表达方式但意思相同的问题
                    doc_id=example.doc_id,
                    doc_tokens=example.doc_tokens,
                    q_type=example.q_type,
                    entity_type=example.entity_type,
                    relations=example.relations,
                    input_features=input_features
                )
                features.append(group_feature)
            else:
                print(example.doc_id, example)

        return features








