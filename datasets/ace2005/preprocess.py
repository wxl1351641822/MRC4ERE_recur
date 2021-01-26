
import os
import json
import re
import argparse

from tqdm import tqdm
from utils.relation_template import *
from collections import defaultdict,Counter
import matplotlib.pyplot as plt


def aligment_ann(original, newtext, ann_file, offset):
    # Ensure that uncased tokenizers can also be aligned
    original = original.lower()
    newtext = newtext.lower()
    annotation = []
    terms = {}
    ends = {}
    for line in open(ann_file):
        if line.startswith('T'):
            # print(line)
            annots = line.rstrip().split("\t", 2)#[id,type beg end,entity]
            # print(annots)
            typeregion = annots[1].split(" ")#[type,beg,end]
            # print(typeregion)
            start = int(typeregion[1]) - offset
            end = int(typeregion[2]) - offset
            if not start in terms:
                terms[start] = []
            if not end in ends:
                ends[end] = []
            if len(annots) == 3:
                terms[start].append(
                    [start, end, annots[0], typeregion[0], annots[2]])#[beg,end,id,type,entity]
            else:
                terms[start].append([start, end, annots[0], typeregion[0], ""])#无entity名称
            ends[end].append(start)
        else:
            annotation.append(line)
    orgidx = 0
    newidx = 0
    orglen = len(original)#5543
    newlen = len(newtext)#5615--ACE文档贼长，不是词长，是字符长
    while orgidx < orglen and newidx < newlen:
        if original[orgidx] == newtext[newidx]:
            orgidx += 1
            newidx += 1
        elif newtext[newidx] == '\n':
            newidx += 1
        elif original[orgidx] == '\n':
            orgidx += 1
        elif newtext[newidx] == ' ':
            newidx += 1
        elif original[orgidx] == ' ':
            orgidx += 1
        elif newtext[newidx] == '\t':
            newidx += 1
        elif original[orgidx] == '\t':
            orgidx += 1
        elif newtext[newidx] == '.':
            # ignore extra "." for stanford
            newidx += 1
        else:
            assert False, "%d\t$%s$\t$%s$" % (
                orgidx, original[orgidx:orgidx + 20], newtext[newidx:newidx + 20])
        if orgidx in terms:#terns[start]=[[beg,end,id,type,entity],...]
            for l in terms[orgidx]:
                l[0] = newidx#--beg换为新的id
        if orgidx in ends:#换尾idx
            for start in ends[orgidx]:
                for l in terms[start]:
                    if l[1] == orgidx:
                        l[1] = newidx
            del ends[orgidx]#换后idx就不需要了
    entities = []
    relations = []
    dict1 = {}
    i = 0
    qiantao_count=0
    duoci_count=0
    ##实体存在同一个实体属于多个类别。
    for ts in terms.values():#ts=[[beg,end,id,type,entity],...]嵌套实体/嵌套类别
        end=ts[0][1]
        if(len(ts)>1):
            qiantao_count+=len(ts)
            # print("嵌套实体",ts)
        for term in ts:
            if term[4] == "":
                entities.append([term[2], term[3], term[0],
                                 term[1], newtext[term[0]:term[1]]])#[id,type,beg,end,entity]
            else:
                # assert newtext[term[0]:term[1]].replace("&AMP;", "&").replace("&amp;", "&").replace(" ", "").replace(
                #    "\n", "") == term[4].replace(" ", "").lower(), newtext[term[0]:term[1]] + "<=>" + term[4]
                assert newtext[term[0]:term[1]].replace(" ", "").replace('\n', "").replace("&AMP;", "&").replace("&amp;", "&") == \
                    term[4].replace(" ", "").lower(), newtext[term[0]:term[1]] + "<=>" + term[4]
                entities.append([term[2], term[3], term[0], term[1],
                                 newtext[term[0]:term[1]].replace("\n", " ")])#[id,type,beg,end,entity]

            if(' ' in entities[-1][4]):
                duoci_count+=1
                # print("多词实体：",entities[-1])

            dict1[term[2]] = i
            i += 1


    seo_head=defaultdict(list)
    seo_tail=defaultdict(list)
    epo=defaultdict(list)
    seo = defaultdict(list)
    #二元关系
    for rel in annotation:
        rel_id, rel_type, rel_e1, rel_e2 = rel.strip().split()
        rel_e1 = rel_e1[5:]
        rel_e2 = rel_e2[5:]
        relations.append([rel_id, rel_type, rel_e1, rel_e2])
        epo[rel_e1 + ';' + rel_e2].append(relations[-1])
        seo_head[rel_e1].append(relations[-1])
        seo_tail[rel_e2].append(relations[-1])
        seo[rel_e1].append(relations[-1])
        seo[rel_e2].append(relations[-1])
    relations1 = []
    for rel in relations:
        _, rel_type, rel_e1, rel_e2 = rel
        rel_e1_idx = dict1[rel_e1]
        rel_e2_idx = dict1[rel_e2]
        relations1.append([rel_type, rel_e1_idx, rel_e2_idx])
    entities1 = [[ent[1], ent[2], ent[3], ent[4]] for ent in entities]#[type,beg,end,entity]
    ##统计重叠实体数量
    seo_head_count = 0
    seo_tail_count = 0
    epo_count=0
    seo_count=0

    for e1_e2,rel in epo.items():
        e1,e2=e1_e2.split(';')
        if(len(rel)==1):
            if(len(seo_head[e1])>1):
                seo_head_count+=1
                # print("seo_head:",seo_head[e1])
            if (len(seo_tail[e2]) > 1):
                seo_tail_count += 1
                # print("seo_tail:", seo_tail[e2])
            if((len(seo[e1])>1) or len(seo[e2])>1):
                seo_count+=1
                # print("seo:",seo[e1],seo[e2])
        else:
            epo_count+=len(rel)
            # print("epo:",rel)
    # print(qiantao_count,duoci_count,epo_count,seo_count,seo_head_count,seo_tail_count)
    return entities1, relations1,qiantao_count,duoci_count,epo_count,seo_count,seo_head_count,seo_tail_count


def passage_blocks(txt, window_size, overlap):
    blocks = []
    regions = []
    for i in range(0, len(txt), window_size-overlap):
        b = txt[i:i+window_size]
        blocks.append(b)
        regions.append((i, i+window_size))
    return blocks, regions


def get_block_er(txt, entities, relations, window_size, overlap, tokenizer):
    """
    Get the block level annotation
    Args:
        txt: text to be processed, list of token
        entities: list of (entity_type, start, end,entity)
        relations: list of (relation_type,entity1_idx,entity2_idx)
        window_siez: sliding window size
        overlap: overlap between two adjacent windows
    Returns:
        ber: list of [block，entities, relations]
    """
    blocks, block_range = passage_blocks(txt, window_size, overlap)
    ber = [[[], [], []] for i in range(len(block_range))]
    e_dict = {}
    for i, (s, e) in enumerate(block_range):
        es = []
        for j, (entity_type, start, end, entity_str) in enumerate(entities):
            if start >= s and end <= e:
                nstart, nend = start-s, end-s
                if tokenizer.convert_tokens_to_string(blocks[i][nstart:nend]) == entity_str:
                    es.append((entity_type, nstart, nend, entity_str))#entity=[(entity_type, nstart, nend, entity_str),...]
                    e_dict[j] = e_dict.get(j, [])+[i]
                else:
                    print(
                        "The entity string and its corresponding index are inconsistent")
        ber[i][0] = blocks[i]
        ber[i][1].extend(es)
    for r, e1i, e2i in relations:
        if e1i not in e_dict or e2i not in e_dict:
            print("Entity lost due to sliding window")
            continue
        i1s, i2s = e_dict[e1i], e_dict[e2i]
        intersec = set.intersection(set(i1s), set(i2s))
        if intersec:
            for i in intersec:
                t1, s1, e1, es1 = entities[e1i][0], entities[e1i][1] - \
                    block_range[i][0], entities[e1i][2] - \
                    block_range[i][0], entities[e1i][3]
                t2, s2, e2, es2 = entities[e2i][0], entities[e2i][1] - \
                    block_range[i][0], entities[e2i][2] - \
                    block_range[i][0], entities[e2i][3]
                ber[i][2].append((r, (t1, s1, e1, es1), (t2, s2, e2, es2)))
        else:
            print("The two entities of the relationship are not on the same sentence")
    return ber#entity=[(entity_type, nstart, nend, entity_str),...],rel=[(r,(type1,start1,end1,entity1).(type1,start1,end1,entity1)),...]


def get_question(question_templates, head_entity, relation_type=None, end_entity_type=None,is_mq=False):
    """
    Args:
        head_entity: (entity_type,start_idx,end_idx,entity_string) or entity_type
    """


    if relation_type == None:
        question = question_templates['qa_turn1'][head_entity[0]] if isinstance(
            head_entity, tuple) else question_templates['qa_turn1'][head_entity]
    else:
        question = question_templates['qa_turn2'][str(
            (head_entity[0], relation_type, end_entity_type))]
        if(is_mq):
            question = [q.replace('XXX', head_entity[3]) for q in question]
        else:
            question = question.replace('XXX', head_entity[3])
    if(is_mq):
        question='\t'.join(question)
    # print(is_mq,question)
    return question


def block2qas(ber, dataset_tag, title="", threshold=1, max_distance=45,is_mq=False):
    """
    Args:
        ber: (block,entities,relations)
        dataset_tag: type of dataset
        title: title corresponding to the passage to which the block belongs
        threshold: only consider relationships where the frequency is greater than or equal to the threshold
        max_distance: used to filter relations by distance from the head entity
    """

    # if dataset_tag.lower() == "ace2004":
    #     entities = ace2004_entities
    #     relations = ace2004_relations
    #     idx1s = ace2004_idx1
    #     idx2s = ace2004_idx2
    #     dist = ace2004_dist
    #     if(is_mq):
    #         question_templates = ace2004_mq_question_templates
    #     else:
    #         question_templates = ace2004_question_templates
    if dataset_tag.lower() == 'ace2005':
        entities = ace2005_ent_label_list
        relations = ace2005_rel_label_list
        idx1s = ace2005_idx1
        idx2s = ace2005_idx2
        dist = ace2005_dist
        if (is_mq):##只有多问题
            question_templates = ace2005_qt
        else:
            question_templates = ace2005_qt
    else:
        raise Exception("this data set is not yet supported")
    block, ents, relas = ber#entity=[(entity_type, nstart, nend, entity_str),...],rel=[(r,(type1,start1,end1,entity1).(type1,start1,end1,entity1)),...]
    res = {'context': block, 'title': title}
    RC=[]
    # QA turn 1
    dict1 = {k: get_question(question_templates, k,is_mq=is_mq) for k in entities}
    qat1 = {dict1[k]: [] for k in dict1}
    q2e={q:k for k,q in dict1.items()}

    # print(ents)
    ent_list=[]
    if(len(ents)!=len(list(set(ents)))):
        print(len(ents),len(list(set(ents))))
    for en in ents:
        if(en in ent_list):
            print(en)
        else:
            ent_list.append(en)
        q = dict1[en[0]]
        qat1[q].append(en)
    qat1={"qas":qat1,"q2e":q2e}

    # QA turn 2
    mt2_over_num = 0
    mt2_q_len=[]

    mt2_len=[]
    for rel in relas:
        RC.append(rel[0])
    RC=list(set(RC))
    res["RC"]=RC
    if max_distance > 0:
        dict2 = {(rel[1], rel[0], rel[2][0]): [] for rel in relas}
        for rel in relas:
            if(rel[2] not in dict2[(rel[1], rel[0], rel[2][0])]):
                dict2[(rel[1], rel[0], rel[2][0])].append(rel[2])#dict2[((type1,start1,end1,entity1),r,type2]=[(type2,start2,end2,entity2),...]
            else:
                print(rel)


        qat2 = []
        ents1 = sorted(ents, key=lambda x: x[1])#start排序
        for i, ent1 in enumerate(ents1):#所有的头实体：ent1=(entity_type, nstart, nend, entity_str)
            start = ent1[1]
            qas = {}
            q2k={}
            for j, ent2 in enumerate(ents1[i+1:], i+1):
                if ent2[1] > start+max_distance:
                    break
                else:
                    head_type, end_type = ent1[0], ent2[0]##头实体，尾巴实体--枚举所有
                    for rel_type in relations:
                        idx1, idx2 = idx1s[head_type], idx2s[(
                            rel_type, end_type)]
                        if dist[idx1][idx2] >= threshold:
                            k = (ent1, rel_type, end_type)
                            q = get_question(
                                question_templates, ent1, rel_type, end_type,is_mq=is_mq)
                            qas[q] = dict2.get(k, [])
                            q2k[q] = k
            qat2.append({'head_entity': ent1, "qas": qas, "q2k": q2k})

            for q, el in qas.items():
                if (len(el) > 1):
                    mt2_over_num += 1
                mt2_len.append(len(el))
            mt2_q_len.append(len(qas))
    else:
        dict2 = {(rel[1], rel[0], rel[2][0]): [] for rel in relas}
        for rel in relas:
            dict2[(rel[1], rel[0], rel[2][0])].append(rel[2])

        qat2 = []
        relations=ace2005_entity_relation_map
        for ent in ents:
            qas = {}
            q2k={}
            for rel in relations[ent[0]]:
                _,rel_type,ent_type=eval(rel)
                k = (ent, rel_type, ent_type)
                idx1, idx2 = idx1s[ent[0]], idx2s[(rel_type, ent_type)]
                if dist[idx1][idx2] >= threshold:
                    q = get_question(question_templates,
                                     ent, rel_type, ent_type,is_mq=is_mq)
                    qas[q] = dict2.get(k, [])##qas[q]=[(type2,start2,end2,entity2),...]-tail_ent
                    q2k[q]=k
            qat2.append({'head_entity': ent, "qas": qas,"q2k":q2k})
            for q, el in qas.items():
                mt2_len.append(len(el))
                if (len(el) > 1):
                    mt2_over_num += 1
            mt2_q_len.append(len(qas.keys()))
            # print(len(qas.keys()))
    mt1_over_num=0
    mt1_len = []
    for q,el in qat1.items():
        mt1_len.append(len(el))
        if(len(el)>1):
            mt1_over_num+=1
    # print(mt2_q_len)
    qas = [qat1, qat2]
    res["qa_pairs"] = qas
    return res,mt1_over_num,mt2_over_num,mt1_len,mt2_len,mt2_q_len


def char_to_wordpiece(passage, entities, tokenizer):
    entities1 = []
    tpassage = tokenizer.tokenize(passage)
    for ent in entities:
        ent_type, start, end, ent_str = ent
        s = tokenizer.tokenize(passage[:start])
        start1 = len(s)
        ent_str1 = tokenizer.tokenize(ent_str)
        end1 = start1 + len(ent_str1)
        ent_str2 = tokenizer.convert_tokens_to_string(ent_str1)
        assert tpassage[start1:end1] == ent_str1
        entities1.append((ent_type, start1, end1, ent_str2))#[(type,start,end,ent)]
    return entities1


def process(data_dir, output_dir, tokenizer, is_test, window_size, overlap, dataset_tag, threshold=1, max_distance=45,is_mq=False):
    """
    Args:
        data_dir: data directory
        output_dir: output directory
        tokenizer: tokenizer of pretrained model
        is_test: whether it is test data
        window_size: sliding window size
        overlap: overlap between two adjacent windows
        dataset_tag: type of dataset
        threshold： only consider relationships where the frequency is greater than or equal to the threshold
        max_distance: used to filter relations by distance from the head entity
    """
    ann_files = []
    txt_files = []
    data = []
    for f in os.listdir(data_dir):
        if f.endswith('.txt'):
            txt_files.append(os.path.join(data_dir, f))
        elif f.endswith('.ann'):
            ann_files.append(os.path.join(data_dir, f))
    ann_files = sorted(ann_files)
    txt_files = sorted(txt_files)
    mt1_over_num, mt2_over_num = 0,0
    qiantao_count, duoci_count, epo_count, seo_count, seo_head_count, seo_tail_count=0,0,0,0,0,0
    seq_num=[]
    rel_num=[]
    ent_num=[]
    ent_num=[]
    mt1_len, mt2_len, mt2_q_len=[],[],[]
    for ann_path, txt_path in tqdm(zip(ann_files, txt_files), total=len(ann_files)):
        with open(txt_path, encoding='utf-8') as f:
            raw_txt = f.read()
            txt = [t for t in raw_txt.split('\n') if t.strip()]
        # get the title information, the title will be added to all windows of a passage
        title = re.search('[A-Za-z_]+[A-Za-z]', txt[0]
                          ).group().split('-')+txt[1].strip().split()
        title = " ".join(title)
        title = tokenizer.tokenize(title)
        ntxt = ' '.join(txt[3:])
        ntxt1 = tokenizer.tokenize(ntxt)
        ntxt2 = tokenizer.convert_tokens_to_string(ntxt1)#ntxt.lower()
        seq_num.append(len(ntxt2))
        offset = raw_txt.index(txt[3])#51
        # print(offset)
        entities, relations,qiantao_count1,duoci_count1,epo_count1,seo_count1,seo_head_count1,seo_tail_count1 = aligment_ann(
            raw_txt[offset:], ntxt2, ann_path, offset)
        rel_num.append(len(relations))
        ent_num.append(len(entities))
        # entities=[[type,beg,end,entity],...]
        # rel=[[type,e1_idx,e2_idx],..]
        qiantao_count+=qiantao_count1
        duoci_count+= duoci_count1
        epo_count += epo_count1
        seo_count += seo_count1
        seo_head_count+=seo_head_count1
        seo_tail_count+=seo_tail_count1


        # convert entitiy index from char level to wordpiece level
        entities = char_to_wordpiece(ntxt2, entities, tokenizer)##[(type,start,end,ent),...]
        if is_test and not is_mq:
            data.append({"title": title, "passage": ntxt1,
                         "entities": entities, "relations": relations})
        else:
            block_er = get_block_er(
                ntxt1, entities, relations, window_size, overlap, tokenizer)

            for ber in block_er:
                res, mt1_over_num1, mt2_over_num1,mt1_len1,mt2_len1,mt2_q_len1 = block2qas(ber, dataset_tag,
                                                            title, threshold, max_distance,is_mq)
                mt1_len+=mt1_len1
                mt2_len+=mt2_len1
                mt2_q_len+=mt2_q_len1
                mt1_over_num+=mt1_over_num1
                mt2_over_num+=mt2_over_num1
                data.append(res)




    print("嵌套情况：{}，多词实体情况：{}，epo:{},seo:{},seo_head:{},seo_tail:{},第一轮多回答数量：{}，第二轮多回答数量：{}".format(qiantao_count, duoci_count, epo_count, seo_count, seo_head_count, seo_tail_count, mt1_over_num, mt2_over_num))

    rel_dic=Counter(rel_num)
    ent_dic=Counter(ent_num)
    seq_dic=Counter(seq_num)
    print("block sum:",len(data))
    print("seq len:",sum(seq_num),seq_dic)
    print("rel:", sum(rel_num),rel_dic)
    print("ent:",sum(ent_num),ent_dic)
    mt1_dic=Counter(mt1_len)
    mt2_dic=Counter(mt2_len)
    mt2_q_dic=Counter(mt2_q_len)
    print("第1轮问题的答案数量:",len(mt1_len),sum(mt1_len), mt1_dic)
    print("第2轮问题的答案数量:", sum(mt2_len), mt2_dic)
    print("第2轮问题的问题数量:", len(mt2_q_len),sum(mt2_q_len), mt2_q_dic)
    # plt.bar(list(seq_dic.keys()), list(seq_dic.values()))
    # plt.plot(list(rel_dic.keys()),list(rel_dic.values()))
    # plt.plot(list(ent_dic.keys()), list(ent_dic.values()))
    # plt.title("seq num")
    # plt.xlabel("seq num")
    # plt.ylabel("text num")
    # plt.show()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if(is_mq):
        mq_data=get_mqdata(data_dir,data)
        save_path = os.path.join(output_dir, os.path.split(data_dir)[-1] + "_mq.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(mq_data, f, sort_keys=True, indent=4)
    save_path = os.path.join(output_dir, os.path.split(data_dir)[-1] + ".json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, indent=4)


def get_mqdata(data_dir,data):
    mq_data = {"data": [{"title": data_dir.split('/')[-2], "paragraphs": []}]}
    qas_i = 0
    qas_len=[]
    for res in tqdm(data):
        i = 0
        token_id_2_string_id = []
        for c in res["context"]:
            # print(i,c)
            token_id_2_string_id.append(i)
            if (i==0 or not c.startswith('##')):
                i += 1
        token_id_2_string_id.append(i)
        p = {"context": (' '.join(res["context"])).replace(' ##', ''), "qas": []}
        qas = []
        relations = []
        context = p["context"].split(' ')
        seq_length = len(context)
        q2e=res["qa_pairs"][0]["q2e"]
        for q, ans in res["qa_pairs"][
            0]["qas"].items():  # ans=[(entity_type, nstart, nend, entity_str),...],rel=[(r,(type1,start1,end1,entity1).(type1,start1,end1,entity1)),...]
            ent_type=q2e[q]
            qs = q.split("\t")

            qas.append({"questions": qs, "type": "entity", "entity_type": ent_type, "id": "qas_" + str(qas_i),
                        "label": ["O"] * seq_length})
            if (len(ans) == 0):
                continue
            qas_i += 1
            for type, s, e, ent in ans:  # (entity_type, nstart, nend, entity_str)
                # print(res["context"][s:e],ent,s,e)
                s = token_id_2_string_id[s]
                e = token_id_2_string_id[e]
                # if(' '.join(context[s:e]) != ent):
                #     print(s,e,context[s:e])
                #     print(context)
                assert ' '.join(context[s:e]) == ent
                if (s == e - 1):
                    qas[-1]["label"][s] = 'S'
                else:
                    qas[-1]["label"][s] = 'B'
                    for i in range(s + 1, e - 1):
                        qas[-1]["label"][i] = 'I'
                    qas[-1]["label"][e - 1] = 'E'
                # print(qas[-1])
        q_list=[]
        for triple in res["qa_pairs"][1]:  # {"head_entity": ent1, "qas": qas}
            head = triple["head_entity"]# (entity_type, nstart, nend, entity_str)
            head=(head[0],token_id_2_string_id[head[1]],token_id_2_string_id[head[2]],head[3])
            if(' '.join(context[head[1]:head[2]])!=head[3]):
                print(head)
            q2k=triple["q2k"]#q2k[q] = (ent, rel_type, ent_type)
            for q, ans in triple["qas"].items():  # qas[q]=[(type2,start2,end2,entity2),...]\
                tail_type=q2k[q][-1]
                qs = q.split("\t")
                qas.append(
                    {"questions": qs, "type": "relation", "entity_type": triple["q2k"][q][1], "tail_type": tail_type,
                     "id": "qas_" + str(qas_i),"head":head,
                     "label": ["O"] * seq_length})
                qas_i += 1
                if (len(ans) < 1):
                    continue
                for type, s, e, ent in ans:  # (entity_type, nstart, nend, entity_str)
                    s = token_id_2_string_id[s]
                    e = token_id_2_string_id[e]
                    assert ' '.join(context[s:e]) == ent
                    relations.append({"label": triple["q2k"][q][1], "e1_ids": list(
                        range(head[1],head[2])),
                                      "e2_ids": list(range(s, e)), "e1_type": head[0], "e2_type": type,
                                      "e1_text": head[3], "e2_text": ent})
                    if (s == e - 1):
                        qas[-1]["label"][s] = 'S'
                    else:
                        qas[-1]["label"][s] = 'B'
                        for i in range(s + 1, e - 1):
                            qas[-1]["label"][i] = 'I'
                        qas[-1]["label"][e - 1] = 'E'
                # if('\t'.join(qs) not in q_list):
                #
                #     q_list.append('\t'.join(qs))
                # else:
                #     print(head, triple["q2k"][q][1],ans[0][0])
                #     print(qas[-1])
        p["relations"] = relations
        p["qas"] = qas
        qas_len.append(len(qas))
        mq_data["data"][0]["paragraphs"].append(p)
        # print(p)
    print(sum(qas_len),qas_len)
    return mq_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir='/data/home/wuyuming/wxl/数据集/ACE2005/multiQA_raw_data/'
    parser.add_argument("--data_dir", default=dir+'train')
    parser.add_argument(
        "--dataset_tag", choices=["ace2004", 'ace2005'], default='ace2005')
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=15)
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--output_base_dir",
                        default="./")
    parser.add_argument("--pretrained_model_path",
                        default='/data/home/wuyuming/wxl/pretrained_models/bert-base-uncased')
    parser.add_argument("--max_distance", type=int, default=45,
                        help="used to filter relations by distance from the head entity")
    parser.add_argument("--is_mq", type=bool, default=True,
                        help="多问题吗？")
    parser.add_argument("--is_bidirect", type=bool, default=False,
                        help="仅从前向后链接则为False；")
    args = parser.parse_args()
    # args.max_distance=0
    # datadir='../../../数据集/ACE2005/multiQA_raw_data'
    if not args.is_test or args.is_mq:
        output_dir = "{}/{}_overlap_{}_window_{}_threshold_{}_max_distance_{}_is_mq_{}".format(args.output_base_dir, os.path.split(
            args.pretrained_model_path)[-1], args.overlap, args.window_size, args.threshold, args.max_distance,args.is_mq)
    else:
        output_dir = args.output_base_dir
    from transformers import BertTokenizer
    # print(args.pretrained_model_path)
    print("***train***"*10)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    process(args.data_dir, output_dir, tokenizer, args.is_test,
            args.window_size, args.overlap, args.dataset_tag, args.threshold, args.max_distance,args.is_mq)
    args.data_dir=dir+'dev'
    print("***dev***"*10)

    process(args.data_dir, output_dir, tokenizer, args.is_test,
            args.window_size, args.overlap, args.dataset_tag, args.threshold, args.max_distance, args.is_mq)
    args.data_dir = dir + 'test'
    print("***test***" * 10)
    process(args.data_dir, output_dir, tokenizer, args.is_test,
            args.window_size, args.overlap, args.dataset_tag, args.threshold, args.max_distance, args.is_mq)
