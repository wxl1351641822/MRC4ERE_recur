import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
from utils.relation_template import *

def get_ent_temp_dict(input_file):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    ent_temp_dict={}
    for entry in input_data:
        for doc_id,paragraph in enumerate(entry["paragraphs"]):
            for qa in paragraph["qas"]:
                questions = qa["questions"]
                q_type = qa["type"] # 问的是entity还是relation
                question_type = qa["entity_type"] # 问的是哪个类型的entity
                if q_type == "entity":
                    ent_temp_dict[question_type]=questions
    print(ent_temp_dict)
    return ent_temp_dict


# get_ent_temp_dict("mrc4ere/dev.json")
# get_ent_temp_dict("mrc4ere/train.json")
# get_ent_temp_dict("mrc4ere/test.json")

class conll04PreProcess:
    def __init__(self,args):
        self.output_dir = args.output_dir
        if (not os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)
        pickle.dump(args, open(self.output_dir + 'args', 'wb'))
        self.model=args.model
        self.mix_ent_rel=args.mix_ent_rel

        self.ent_matrix_label=args.ent_matrix_label
        self.rel_tail_head = args.rel_tail_head
        self.multi_decoder=args.multi_decoder

    def read_file(self,path):
        with open(path, 'r', encoding='utf-8') as f:
            id = ''
            text = []
            label = []
            rels = []
            rel_indexs = []
            beg = 0
            end = 0
            data = []
            for line in f.readlines():
                line = line.strip()
                if (line[0] == '#'):
                    if (len(text) > 0):
                        # break
                        data.append(
                            {"id": id, "text": text, "label": label, "rels": rels, "rel_indexes": rel_indexs})
                        # print(data[-1])
                    id = line
                    text = []
                    label = []
                    rels = []
                    rel_indexs = []

                else:
                    line = line.split('\t')
                    text.append(line[1])
                    label.append(line[2])
                    rels.append(eval(line[3]))
                    rel_indexs.append(eval(line[4]))

            if (len(text) > 0):
                data.append(
                    {"id": id, "text": text, "label": label, "rels": rels, "rel_indexes": rel_indexs})
        # print(data)
        return data

    def tongji(self,train):
        num_entities = []
        num_rel = []
        num_seq = []
        num_entity_len = []
        num_seo = []
        num_epo = []
        data2 = []
        entity_class_num_dict = defaultdict(int)
        entity_class_num_one_doc_dict = defaultdict(list)
        rel_class_num_one_doc_dict = defaultdict(list)
        rel_class_num_dict = defaultdict(int)
        head_rel_2_tail = defaultdict(list)
        head_2_rel = defaultdict(list)
        rel_2_tail = defaultdict(list)
        head_rel_one_doc_dict = defaultdict(list)
        head_name_rel_one_doc_dict = defaultdict(list)
        num_error = 0
        for d in train:
            id = d["id"]
            train_text, train_label, train_rels, train_rel_indexs = d["text"], d["label"], d["rels"], d["rel_indexes"]
            beg, end = 0, 0
            entities = {}
            entity_label = ''
            entity_class_num_one_doc_dict1 = defaultdict(int)
            for i, (text, label) in enumerate(zip(train_text, train_label)):
                if (len(entity_label) > 0 and label[0] == 'B'):
                    end = i
                    num_entity_len.append(end - beg)
                    entity_label = entity_label.lower()
                    entities[end - 1] = {"entity": ' '.join(train_text[beg:end]), "type": entity_label, "beg": beg,
                                         "end": end - 1}
                    entity_class_num_dict[entity_label] += 1
                    entity_class_num_one_doc_dict1[entity_label] += 1
                    entity_label = ''

                if (len(entity_label) == 0 and label[0] != 'O'):
                    beg = i
                    entity_label = label[2:]
                    if (label[0] != 'B'):
                        num_error += 1
                # if(label[0]=='B'):
                #     beg=i
                #     entity_label=label[2:]
                if (len(entity_label) > 0 and label[0] == 'O'):
                    end = i
                    num_entity_len.append(end - beg)
                    entity_label = entity_label.lower()
                    entities[end - 1] = {"entity": ' '.join(train_text[beg:end]), "type": entity_label, "beg": beg,
                                         "end": end - 1}
                    entity_class_num_dict[entity_label] += 1
                    entity_class_num_one_doc_dict1[entity_label] += 1
                    entity_label = ''

                # print(i,text,label,rel,rel_index,beg,end,entity_label)

            if (len(entity_label) > 0):
                end = len(train_text)
                num_entity_len.append(end - beg)
                entity_label = entity_label.lower()
                entities[end - 1] = {"entity": ' '.join(train_text[beg:end]), "type": entity_label, "beg": beg,
                                     "end": end - 1}
                entity_class_num_dict[entity_label] += 1
                entity_class_num_one_doc_dict1[entity_label] += 1
            for k, v in entity_class_num_one_doc_dict1.items():
                entity_class_num_one_doc_dict[k].append(v)
            relations = []
            RC = []

            seo = defaultdict(int)
            epo = defaultdict(int)
            rel_class_num_one_doc_dict1 = defaultdict(int)
            head_rel_one_doc_dict1 = defaultdict(int)
            head_name_rel_one_doc_dict1 = defaultdict(int)
            count_rel=defaultdict(list)
            count_relh = defaultdict(list)
            count_relt = defaultdict(list)
            for i, (rel, index) in enumerate(zip(train_rels, train_rel_indexs)):
                if (rel[0] == 'N'):
                    continue
                for r, ind in zip(rel, index):
                    r = r.lower()
                    if (r not in RC):
                        RC.append(r)
                    if (i in entities):
                        head = i
                    if (ind in entities):
                        tail = ind
                    if (entities[tail]["type"] not in head_rel_2_tail[r + '_' + entities[head]["type"]]):
                        head_rel_2_tail[r + '_' + entities[i]["type"]].append(entities[tail]["type"])
                    if (entities[tail]["type"] not in rel_2_tail[r]):
                        rel_2_tail[r].append(entities[tail]["type"])
                    if (r not in head_2_rel[entities[head]["type"]]):
                        head_2_rel[entities[head]["type"]].append(r)
                    relations.append({"rel_type": r, "head": head, "tail": tail})
                    count_rel[r].append([head,tail])
                    count_relh[r].append(head)
                    count_relt[r].append(tail)
                    rel_class_num_dict[r] += 1
                    seo[head] += 1
                    seo[tail] += 1
                    epo[str(head) + '_' + str(tail)] += 1
                    rel_class_num_one_doc_dict1[r] += 1
                    head_rel_one_doc_dict1[r + '_' + entities[head]["type"]] += 1
                    head_name_rel_one_doc_dict1[r + '_' + entities[head]["entity"]] += 1
            for r,v in count_rel.items():
                for h in count_relh[r]:
                    for t in count_relt[r]:
                        if([h,t] not in v):
                            print([h,t],r,v)

            for k, v in rel_class_num_one_doc_dict1.items():
                rel_class_num_one_doc_dict[k].append(v)
            for k, v in head_rel_one_doc_dict1.items():
                head_rel_one_doc_dict[k].append(v)
            for k, v in head_name_rel_one_doc_dict1.items():
                head_name_rel_one_doc_dict[k].append(v)
            num_s = 0
            # print(sum(seo.values()),len(seo.keys()))
            # # if(flag):
            # print(seo)
            # print(epo)
            num = 0
            if (sum(seo.values()) != len(seo.keys())):
                for r in relations:
                    if ((r["head"] in seo.keys() and seo[r["head"]] > 1) or (
                            r["tail"] in seo.keys() and seo[r["tail"]] > 1)):
                        num += 1
            num_seo.append(num)
            num = 0
            if (sum(epo.values()) != len(epo.keys())):
                for k, v in epo.items():
                    if (v > 1):
                        num += v
            seo_r = []
            epo_r = []
            normal = []
            for r in relations:

                k = str(r["head"]) + '_' + str(r["tail"])
                if (k in epo and epo[k] > 1):
                    epo_r.append(r)
                elif ((r["head"] in seo.keys() and seo[r["head"]] > 1) or (
                        r["tail"] in seo.keys() and seo[r["tail"]] > 1)):
                    seo_r.append(r)
                else:
                    normal.append(r)

            num_epo.append(num)
            data2.append(
                {"id": id, "text": train_text, "label": train_label, "entities": entities, "relations": relations,
                 "RC": RC, "epo": epo_r, "seo": seo_r, "normal": normal})
            num_entities.append(len(entities.keys()))
            num_rel.append(len(relations))
            num_seq.append(len(train_text))

        ##句子长度统计
        dic = dict(Counter(num_seq))
        seq_dic = {k: dic[k] for k in sorted(dic.keys())}
        print("total doc num:", len(train))
        print("seq_len:")
        print(seq_dic.keys())
        print(seq_dic.values())
        plt.bar(list(seq_dic.keys()), list(seq_dic.values()))
        plt.title("seq num")
        plt.xlabel("seq num")
        plt.ylabel("text num")
        plt.show()
        ##doc 内实体统计
        print("total entities num:", Counter(num_entities), sum(num_entities))
        ##doc内rel统计
        print("total rel num:", Counter(num_rel), sum(num_rel))
        dic = dict(Counter(num_seo))
        print("total  seo rel num:", dic, sum(num_seo))
        dic = dict(Counter(num_epo))
        print("total  epo rel num:", dic, sum(num_epo))
        print("各类别的实体各有多少：", entity_class_num_dict)
        print("各类别的关系各有多少：", rel_class_num_dict)
        for k, v in entity_class_num_one_doc_dict.items():
            dic = Counter(v)
            print(k, "在各个doc中的数量:", dic, sum(v), sum(dic.values()) - dic[1])
        for k, v in rel_class_num_one_doc_dict.items():
            dic = Counter(v)
            print(k, "在各个doc中的数量:", dic, sum(v), sum(dic.values()) - dic[1])
        print('**' * 30)
        for k, v in head_rel_2_tail.items():
            print(k, "对应的尾实体类型有", v)
        print('**' * 30)
        for k, v in rel_2_tail.items():
            print(k, "对应的尾实体类型有", v)
        print('**' * 30)
        for k, v in head_2_rel.items():
            print(k, "对应的关系类型有", v)
        print('**' * 30)
        for k, v in head_rel_one_doc_dict.items():
            dic = Counter(v)
            print(k, "对应尾实体的数量:", dic, sum(v), sum(dic.values()) - dic[1])
        print('**' * 30)
        for k, v in head_name_rel_one_doc_dict.items():
            dic = Counter(v)
            if (sum(dic.values()) - dic[1] > 0):
                print(k, "对应尾实体的数量:", dic, sum(v), sum(dic.values()) - dic[1])
        print("num error:", num_error)
        return data2, head_2_rel, rel_2_tail

    def get_json(self,data2, path, name="relations"):
        def indext2BIOES(label, beg=-1, end=-1):
            if (beg == end):
                label[beg] = 'S'
            else:
                label[beg] = 'B'
                for i in range(beg + 1, end):
                    label[i] = "I"
                label[end] = 'E'
            return label

        qas_num = 0
        ent_qas_num = 0
        rel_qas_num = 0
        ent_label_list = conll04_ent_label_list
        rel_label_list=conll04_rel_label_list
        entity_relation_map = conll04_entity_relation_map
        ent_q_temp = conll04_ent_question_templates
        rel_q_temp = conll04_question_templates

        paragraphs = []
        qas_id = 1

        for t in data2:
            id = t["id"]
            paragraph = {}
            seq_len = len(t["text"])
            context = " ".join(t["text"])
            paragraph["context"] = context
            entities = t["entities"]
            ent_label = {}
            for ent_type in ent_label_list:
                ent_label[ent_type] = ['O'] * seq_len
            rel_label = {}
            relations = []
            qas = []
            for ent in entities.values():
                entity_type = ent["type"]
                ent_label[entity_type] = indext2BIOES(ent_label[entity_type], ent["beg"], ent["end"])
                for rel_type in entity_relation_map[ent["type"]]:
                    if ((ent["entity"], rel_type) not in rel_label):
                        rel_label[(ent["entity"], rel_type)] = ['O'] * seq_len
            for rel in t[name]:
                # print(rel)
                head = entities[rel["head"]]
                # 1.构建head问题
                tail = entities[rel["tail"]]

                rel_label[(head["entity"], rel["rel_type"])] = indext2BIOES(
                    rel_label[(head["entity"], rel["rel_type"])],
                    tail["beg"], tail["end"])
                relations.append({"label": rel["rel_type"], "e1_ids": list(range(head["beg"], head["end"] + 1)),
                                  "e2_ids": list(range(tail["beg"], tail["end"] + 1)), "e1_type": head["type"],
                                  "e2_type": tail["type"], "e1_text": head["entity"], "e2_text": tail["entity"]})
                # print(relations[-1])

            type = 'entity'
            for ent_type, label in ent_label.items():
                idd = "qas_" + str(qas_id)
                qas_id += 1
                qas.append(
                    {"questions": ent_q_temp[ent_type], "label": label, "type": type, "entity_type": ent_type,
                     "id": idd})
                ent_qas_num += 1
            type = 'relation'
            for (head_entity, rel_type), label in rel_label.items():

                idd = "qas_" + str(qas_id)
                # print(idd,head_entity, rel_type, label)
                qas_id += 1
                questions = rel_q_temp[rel_type].copy()
                for i in range(3):
                    # print(questions[i],head_entity)
                    questions[i] = questions[i].format(head_entity)
                    # print(questions[i])
                rel_qas_num += 1
                # print(questions)
                qas.append({"questions": questions, "label": label, "type": type, "entity_type": rel_type, "id": idd})
                # print(qas[-1])
            qas_num += len(qas)
            # print(qas)
            # print(ent_label)
            # print(rel_label)
            # print(relations)
            paragraph["qas"] = qas
            paragraph["relations"] = relations
            paragraphs.append(paragraph)

        print("qas num:{},ent_qas_num:{},rel_qas_num:{}".format(qas_num, ent_qas_num, rel_qas_num))
        data = [{"paragraphs": paragraphs, "title": "conll04"}]
        print("save in ",path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"data": data}, f)

    def get_mrctp_json(self, data2, path, name="relations"):
        def indext2label(label, beg=-1, end=-1,type='entity'):
            #ent:1-实体头，2-实体结尾，3-单字实体
            #rel:头实体->尾实体
            if (not self.ent_matrix_label and type == 'entity'):
                    if (beg == end):
                        label[beg] = 'S'
                    else:
                        label[beg] = 'B'
                        for i in range(beg + 1, end):
                            label[i] = "I"
                        label[end] = 'E'
            else:#矩阵形式
                label_type = '[E]'
                if(beg<end):
                    if(type=='relation'):
                        label_type = '[+]'
                    index=beg*seq_len+end-int((beg+1)*beg/2)
                else:
                    if (type == 'relation'):
                        label_type = '[-]'
                    index = end * seq_len + beg - int((end + 1) * end / 2)
                if (type == 'entity'):
                        label[index]=label_type
                else:
                   label[index]=label_type

            return label

        qas_num = 0
        ent_qas_num = 0
        rel_qas_num = 0
        ent_label_list = conll04_ent_label_list
        rel_label_list = conll04_rel_label_list
        entity_relation_map = conll04_entity_relation_map
        ent_q_temp = conll04_ent_question_templates

        rel_q_temp=conll04_rel_question_templates

        paragraphs = []
        qas_id = 1

        for t in data2:
            id = t["id"]
            paragraph = {}
            seq_len = len(t["text"])
            context = " ".join(t["text"])
            paragraph["context"] = context
            entities = t["entities"]
            ent_label = {}
            for ent_type in ent_label_list:
                if (self.ent_matrix_label):
                    if (self.multi_decoder == 2 and self.mix_ent_rel):
                        ent_label[ent_type] = [['O'] * int(seq_len * (seq_len + 1) / 2),['O'] * int(seq_len * (seq_len + 1) / 2)]
                    else:
                        ent_label[ent_type] = ['O'] * int(seq_len * (seq_len + 1) / 2)
                else:
                    ent_label[ent_type] = ['O'] * seq_len

            relations = []

            for ent in entities.values():
                entity_type = ent["type"]
                if(self.multi_decoder==2 and self.mix_ent_rel):
                    ent_label[entity_type][0] = indext2label(ent_label[entity_type][0], ent["beg"], ent["end"])
                else:
                    ent_label[entity_type] = indext2label(ent_label[entity_type], ent["beg"], ent["end"])
            rel_label = {}
            for rel_type in rel_label_list:
                if (self.rel_tail_head):
                    if(self.multi_decoder==1):
                        rel_label[rel_type+"@head"]=['O']*int(seq_len*(seq_len+1)/2)
                        rel_label[rel_type + "@tail"] = ['O']*int(seq_len*(seq_len+1)/2)
                    else:
                        rel_label[rel_type]=[['O']*int(seq_len*(seq_len+1)/2),['O']*int(seq_len*(seq_len+1)/2)]
                else:
                    rel_label[rel_type] = ['O'] *int(seq_len * (seq_len + 1) / 2)

            for rel in t[name]:
                head = entities[rel["head"]]
                # 1.构建head问题
                tail = entities[rel["tail"]]

                if(self.rel_tail_head):
                    if (self.multi_decoder == 1):
                        rel_label[rel["rel_type"]+"@head"] = indext2label(
                            rel_label[rel["rel_type"]+"@head"],
                            head["beg"], tail["beg"],type = 'relation')
                        rel_label[rel["rel_type"] + "@tail"] = indext2label(
                            rel_label[rel["rel_type"] + "@tail"],
                            head["end"], tail["end"],type = 'relation')
                    else:
                        for i in range(2):
                            rel_label[rel["rel_type"]][i] = indext2label(
                                rel_label[rel["rel_type"]][i],
                                head["beg"], tail["beg"],type = 'relation')
                else:
                    rel_label[rel["rel_type"]] = indext2label(
                        rel_label[rel["rel_type"]],
                        head["beg"], tail["beg"],type = 'relation')

                relations.append({"label": rel["rel_type"], "e1_ids": list(range(head["beg"], head["end"] + 1)),
                                  "e2_ids": list(range(tail["beg"], tail["end"] + 1)), "e1_type": head["type"],
                                  "e2_type": tail["type"], "e1_text": head["entity"], "e2_text": tail["entity"]})
            qas = []
            type = 'entity'
            for ent_type, label in ent_label.items():
                idd = "qas_" + str(qas_id)
                qas_id += 1
                qas.append(
                    {"questions": ent_q_temp[ent_type], "label": label, "type": type, "entity_type": ent_type,
                     "id": idd})
                ent_qas_num += 1
            type = 'relation'
            for rel_type, label in rel_label.items():

                rel_type=rel_type.split('@')[0]
                idd = "qas_" + str(qas_id)
                qas_id += 1
                questions = rel_q_temp[rel_type]
                if (self.rel_tail_head and self.multi_decoder==1):
                    questions=['[{}]'.format(rel_type.split('@')[-1])+q for q in questions]
                rel_qas_num += 1
                qas.append({"questions": questions, "label": label, "type": type, "entity_type": rel_type, "id": idd})
                # print(qas[-1])

            qas_num += len(qas)
            # print(qas)
            # print(ent_label)
            # print(rel_label)
            # print(relations)
            paragraph["qas"] = qas
            paragraph["relations"] = relations
            paragraphs.append(paragraph)

        print("qas num:{},ent_qas_num:{},rel_qas_num:{}".format(qas_num, ent_qas_num, rel_qas_num))
        data = [{"paragraphs": paragraphs, "title": "conll04"}]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"data": data}, f)

    def process(self):
        conll04_origin_dir='../../../数据集/conll04/origin'
        file_name=["train","dev","test"]

        for name in file_name:
            path=os.path.join(conll04_origin_dir,name+'.txt')
            data=self.read_file(path)
            data,_,_=self.tongji(data)
            if(self.model=='mrc4ere'):
                self.get_json(data,self.output_dir+name+'.json')
                self.get_json(data, self.output_dir+ name + '_epo.json',name="epo")
                self.get_json(data, self.output_dir+ name + '_spo.json',name="seo")
                self.get_json(data, self.output_dir+ name + '_normal.json',name="normal")
            else:#mrc+tplink
                self.get_mrctp_json(data, self.output_dir + name + '.json')
                self.get_json(data, self.output_dir + name + '_epo.json', name="epo")
                self.get_json(data, self.output_dir + name + '_spo.json', name="seo")
                self.get_json(data, self.output_dir + name + '_normal.json', name="normal")
import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='./mymrc4ere/')
parser.add_argument("--model", default='mrc4ere')#mrc4ere,mrctplink
parser.add_argument("--mix_ent_rel", type=bool,default=True)
parser.add_argument("--ent_matrix_label",  type=bool,default=False)
parser.add_argument("--rel_tail_head",  type=bool,default=False)
parser.add_argument("--multi_decoder",  type=int,default=1)
args = parser.parse_args()


# args.model='mrctplink'
if(args.model=='mrctplink'):
    args.ent_matrix_label=True
    # args.martix_label=True
    args.output_dir=args.model+('_matrix' if args.ent_matrix_label else '')+('_rel_head_tail' if args.rel_tail_head else '')+('_'+str(args.multi_decoder))+'/'
print(args.output_dir)
conll04PreProcess(args).process()
# qas num:8776,ent_qas_num:3640,rel_qas_num:5136
# qas num:8776,ent_qas_num:3640,rel_qas_num:5136
# qas num:8776,ent_qas_num:3640,rel_qas_num:5136
# qas num:8776,ent_qas_num:3640,rel_qas_num:5136
# qas num:2146,ent_qas_num:972,rel_qas_num:1174
# qas num:2146,ent_qas_num:972,rel_qas_num:1174
# qas num:2146,ent_qas_num:972,rel_qas_num:1174
# qas num:2146,ent_qas_num:972,rel_qas_num:1174