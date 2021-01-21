# import sys
# sys,
import json
from tqdm import tqdm
from .mrc_example import *
from .mrc_processor import MRCProcessor
from utils.relation_template import *
class MRC4TPLinkerDataset(MRCProcessor):
    def __init__(self):
        MRCProcessor.__init__()
        pass

    def get_clip_labels(self,labels,orig_seq_len,seq_len):
        results=[]
        i=0
        print(seq_len,orig_seq_len)
        for row in range(orig_seq_len):
            for col in range(row,orig_seq_len):
                if(col<seq_len and row<seq_len):
                    results.append(labels[i])
                i+=1
        return labels

    def convert_examples_to_features(self,examples, tokenizer, label_list, max_seq_length, max_query_length, doc_stride,
                                     type=None, max_ans_length=None):
        """Loads a data file into a list of `InputBatch`s."""
        matrix_flag=True
        if (max_ans_length is None):
            matrix_flag=True
            max_ans_length = max_seq_length

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            if (type is not None and example.q_type != type):
                continue
            textlist = example.doc_tokens
            labellist = example.label
            doc_tokens = []
            labels =[]#example.label
            doc_valid = []
            label_mask = []
            tok_to_orig_index = []
            orig_to_tok_index = []
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                orig_to_tok_index.append(len(doc_tokens))
                doc_tokens.extend(token)
                # label_1 = labellist[i]
                for m in range(len(token)):
                    # print(m,token[m])
                    tok_to_orig_index.append(i)
                    if m == 0:
                        labels.append(labellist[i])
                        doc_valid.append(1)
                        # label_mask.append(1)
                    else:
                        doc_valid.append(0)
            # if(len(labels)!=len(labellist)):
            #     print('==')
            #     # print(labels)
            #     # print(labellist)
            # # else:
            #     print('!=',len(labels),len(labellist))
            #     print(labels)
            #     print(labellist)
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
                    # labels = labels[0:max_doc_length]
                    doc_valid = doc_valid[0:max_doc_length]
                    # label_mask = label_mask[0:max_doc_length]
                    orig_seq_len=len(textlist)
                    seq_len=sum(doc_valid)
                    if(matrix_flag):
                        labels=self.get_clip_labels(labels,orig_seq_len,seq_len)
                    else:
                        labels=labels[:max_doc_length]
                if (matrix_flag):
                    label_ids = []
                    label_ids.append(label_map["[CLS]"])
                    label_ids.extend([label_map[l] for l in labels])
                    label_ids.append(label_map["[SEP]"])
                    label_mask = [1] * len(label_ids)
                    while len(label_ids) < max_ans_length:
                        label_ids.append(0)
                        label_mask.append(0)

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

class FilterDataset:
    def __init__(self):
        pass

    def get_train_examples(self, data_dir, unused=False):
        train_examples = self.read_data(data_dir, is_training=True)
        return train_examples

    def get_dev_examples(self, data_dir, unused=False):
        dev_examples = self.read_data(data_dir, is_training=False)
        return dev_examples

    def get_test_examples(self, data_dir, unused=False):
        test_examples = self.read_data(data_dir, is_training=False)
        return test_examples

    def get_labels(self,datasets='ace2005'):
        if (datasets == 'conll04'):
            label_list = conll04_rel_label_list
        else:
            label_list = ace2005_rel_label_list
        return label_list
    def get_labeldict(self,datasets="ace2005"):
        return {rel:id for id,rel in enumerate(self.get_labels(datasets=datasets))}

    def read_data(self,data_dir, is_training=False,datasets='ace2005'):
        examples = []
        with open(data_dir, "r", encoding='utf-8') as reader:
            data = json.load(reader)
            labeldict=self.get_labeldict(datasets=datasets)
            print(labeldict)

            for i,res in tqdm(enumerate(data)):
                i = 0
                token_id_2_string_id = []
                for c in res["context"]:
                    # print(i,c)
                    token_id_2_string_id.append(i)
                    if (i == 0 or not c.startswith('##')):
                        i += 1
                token_id_2_string_id.append(i)
                # p = {"context": (' '.join(res["context"])).replace(' ##', ''), "qas": []}
                RC=res["RC"]
                label=[0]*len(list(labeldict.keys()))
                print(RC)
                for r in RC:
                    label[labeldict[r]]=1
                example=InputExample(guid=i,label=label,text_a=(' '.join(res["context"])).replace(' ##', ''))
                examples.append(example)
                print(example.text_a,example.label)
                print(example)
            print(len(examples))
        return examples

    def convert_examples_to_features(self, examples, tokenizer, label_list, max_seq_length, max_query_length, doc_stride,
                                 type=None, max_ans_length=None):
        """Loads a data file into a list of `InputBatch`s."""
        matrix_flag = True
        if (max_ans_length is None):
            matrix_flag = True
            max_ans_length = max_seq_length

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            if (type is not None and example.q_type != type):
                continue
            textlist = example.text_a
            labellist = example.label
            doc_tokens = []
            labels = []  # example.label
            doc_valid = []
            label_mask = []
            tok_to_orig_index = []
            orig_to_tok_index = []
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                orig_to_tok_index.append(len(doc_tokens))
                doc_tokens.extend(token)
                # label_1 = labellist[i]
                for m in range(len(token)):
                    # print(m,token[m])
                    tok_to_orig_index.append(i)
                    if m == 0:
                        labels.append(labellist[i])
                        doc_valid.append(1)
                        # label_mask.append(1)
                    else:
                        doc_valid.append(0)
            # if(len(labels)!=len(labellist)):
            #     print('==')
            #     # print(labels)
            #     # print(labellist)
            # # else:
            #     print('!=',len(labels),len(labellist))
            #     print(labels)
            #     print(labellist)
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
                    # labels = labels[0:max_doc_length]
                    doc_valid = doc_valid[0:max_doc_length]
                    # label_mask = label_mask[0:max_doc_length]
                    orig_seq_len = len(textlist)
                    seq_len = sum(doc_valid)
                    if (matrix_flag):
                        labels = self.get_clip_labels(labels, orig_seq_len, seq_len)
                    else:
                        labels = labels[:max_doc_length]
                if (matrix_flag):
                    label_ids = []
                    label_ids.append(label_map["[CLS]"])
                    label_ids.extend([label_map[l] for l in labels])
                    label_ids.append(label_map["[SEP]"])
                    label_mask = [1] * len(label_ids)
                    while len(label_ids) < max_ans_length:
                        label_ids.append(0)
                        label_mask.append(0)

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

if __name__=='__main__':
    FilterDataset().read_data("../datasets/ace2005/multiQA/train.json")