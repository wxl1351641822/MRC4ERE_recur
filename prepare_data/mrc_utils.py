import sys

sys.path.append("..")

import json
import collections
from .mrc_example import MRCExample, InputFeature, GroupFeature

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer


def read_squad_examples(input_file, is_training,unused=True):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    unused=['[unused0]','[unused1]']
    examples = []
    for entry in input_data:
        print(len(entry["paragraphs"]))
        for doc_id,paragraph in enumerate(entry["paragraphs"]):
            # if(doc_id==1):
            #     break
            paragraph_text = paragraph["context"]
            all_relations = paragraph["relations"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            # q=[]
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            # print("****"*30)
            # print(paragraph_text,all_relations)
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                questions = qa["questions"]
                label = qa["label"]
                q_type = qa["type"] # 问的是entity还是relation
                question_type = qa["entity_type"] # 问的是哪个类型的entity
                # print(len(doc_tokens),len(label))
                # if(len(doc_tokens)!=len(label)):
                #     print(doc_tokens)
                #     print(label)
                if("head" in qa and unused):
                    # print(qa["head"])# (entity_type, nstart, nend, entity_str)
                    # print(doc_tokens[qa["head"][1]:qa["head"][2]],paragraph_text.split(' ')[qa["head"][1]:qa["head"][2]])
                    doc_tokens1=doc_tokens[:qa["head"][1]]+[unused[0]]+doc_tokens[qa["head"][1]:qa["head"][2]]+[unused[1]]+doc_tokens[qa["head"][2]:]
                    label1 = label[:qa["head"][1]] + ['O'] + label[qa["head"][1]:qa["head"][2]] + [
                        'O'] + label[qa["head"][2]:]
                else:
                    doc_tokens1=doc_tokens
                    label1=label


                relations = []
                if q_type == "entity":
                    for relation in all_relations:
                        # if relation["label"] == question_type and relation["e1_text"] in questions[0]: #  比如问题是问work for的， relation中有work_for的关系，则加进来
                        tmp_relation = relation.copy()
                        relations.append(tmp_relation)
                # else:
                # if('\t'.join(questions) not in q):
                #     q.append('\t'.join(questions))
                if(len(label)==0):
                    print(doc_tokens1)
                example = MRCExample(
                    qas_id=qas_id,
                    doc_id=doc_id,
                    question_text=questions,
                    doc_tokens=doc_tokens1,
                    label=label1,
                    q_type=q_type,
                    entity_type = question_type,
                    relations = relations # list of dict
                )
                examples.append(example)

                # if len(examples) == 20:
                #     return examples
            # print(len(paragraph["qas"]),len(q),len(list(set(q))))
    return examples


def convert_relation_examples_to_features(examples, tokenizer, label_list, max_seq_length, max_query_length, doc_stride):
    # load a data file into a list of "InputBatch"
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        group_features = []
        for query in example.question_text: # question text is not tokenized
            query_tokens = tokenizer.tokenize(query)
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[: max_query_length]

            all_doc_tokens = example.doc_tokens
            all_doc_labels = example.label # original label
            tok_to_orig_index = list(range(len(all_doc_tokens)))

            assert len(example.doc_tokens) == len(example.label)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            _DocSpan = collections.namedtuple(
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:  # add tokens
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                orig_all_doc_labels = all_doc_labels
                input_labels= ["[CLS]"] + ["O"] * len(query_tokens) + ["[SEP]"] \
                            + all_doc_labels + ["[SEP]"]
                label_ids = [label_map[tmp] for tmp in input_labels]

                assert len(label_ids) == len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    label_ids.append(label_map["O"])


                # while len(all_doc_labels) < max_tokens_for_doc:
                #     all_doc_labels.append("O")
                # label_ids = [label_map[tmp] for tmp in all_doc_labels]
                # label_ids = [label_map["[CLS]"]] + [label_map["O"]] * len(query_tokens) + [label_map["[SEP]"]] \
                #            + label_ids + [label_map["[SEP]"]]

                if len(label_ids) != max_seq_length:
                    print(len(orig_all_doc_labels))

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(label_ids) == max_seq_length

                group_features.append(
                    InputFeature(

                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_ids,
                        label_mask=label_mask,
                        valid_id=valid
                    )
                )

        features.append(group_features)

    return features





def _divide_group(example_list, n_groups):
    grouped_list = []
    for i in range(0, len(example_list), n_groups):
        grouped_list.append(example_list[i:i + n_groups])
    return grouped_list


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def iob2_to_iobes(tags):
    """
    checked
    IOB2 -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B', 'S'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I', 'E'))
        else:
            raise Exception('Invalid IOB format!')

    assert len(new_tags) == len(tags)

    return new_tags



if __name__=='__main__':
    read_squad_examples('../datasets/conll04/mrc4ere/train.json', is_training=True)