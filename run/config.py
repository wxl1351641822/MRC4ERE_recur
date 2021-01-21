#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import argparse
import json

from configparser import SafeConfigParser

class Configurable:

    def __init__(self, config_file, extra_args, logger,id):

        config = SafeConfigParser()
        config.read(config_file)
        self.id=id

        if extra_args:
            extra_args = { k[2:] : v for k, v in zip(extra_args[0::2], extra_args[1::2]) }

        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
                    logger.info(section + "-" + k + "-" + v)
        self._config = config
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, "minibatch"))
        config.write(open(self.config_file, "w", encoding="utf8"))
        logger.info("Loaded config file successful.")
        for section in config.sections():
            for k, v in config.items(section):
                logger.info(k + ": " + v)

        bert_config_file = self._config.get('Bert', 'bert_config')
        with open(bert_config_file, "r", encoding='utf-8') as reader:
            self.bert_config_json = json.load(reader)
        self.logger=logger
    def get_alllist(self):
        lis=[]
        klis=[]
        for section in self._config.sections():
            for k, v in  self._config.items(section):
                lis.append(self._config.get(section,k))
                klis.append(k)
        return [klis,lis]


    @property
    def bert_model(self):
        return self._config.get('Bert', 'bert_model')

    @property
    def bert_config(self):
        return self.bert_config_json["bert_config"]

    @property
    def bert_frozen(self):
        return self.bert_config_json["bert_frozen"]

    @property
    def hidden_size(self):
        return self.bert_config_json["hidden_size"]

    @property
    def hidden_dropout_prob(self):
        return self.bert_config_json["hidden_dropout_prob"]

    @property
    def classifier_sign(self):
        return self.bert_config_json["classifier_sign"]

    @property
    def clip_grad(self):
        return self.bert_config_json["clip_grad"]

    @property
    def use_cuda(self):
        return self._config.getboolean('Run', 'use_cuda')

    @property
    def loss_type(self):
        return self._config.get('Run', 'loss_type')

    @property
    def task_name(self):
        return self._config.get('Run', 'task_name')

    @property
    def do_train(self):
        return self._config.getboolean('Run', 'do_train')

    @property
    def do_eval(self):
        return self._config.getboolean('Run', 'do_eval')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def dev_batch_size(self):
        return self._config.getint('Run', 'dev_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def checkpoint(self):
        return self._config.getint('Run', 'checkpoint')

    @property
    def learning_rate(self):
        return self._config.getfloat('Run', 'learning_rate')

    @property
    def epochs(self):
        return self._config.getfloat('Run', 'epochs')

    @property
    def warmup_proportion(self):
        return self._config.getfloat('Run', 'warmup_proportion')

    @property
    def local_rank(self):
        return self._config.getint('Run', 'local_rank')

    @property
    def gradient_accumulation_steps(self):
        return self._config.getint('Run', 'gradient_accumulation_steps')

    @property
    def seed(self):
        return self._config.getint('Run', 'seed')

    @property
    def threshold(self):
        return self._config.getfloat('Run', 'threshold')

    @property
    def export_model(self):
        return self._config.getboolean('Run', 'export_model')

    @property
    def use_filter_flag(self):
        return self._config.getint('Run', 'use_filter_flag')#0-不用filter,1-rel filter,2-rel-tail filter

    @property
    def use_gen_rel(self):
        return self._config.getboolean('Run', 'use_gen_rel')

    @property
    def train(self):
        return self._config.getboolean('Run', 'train')

    @property
    def predict(self):
        return self._config.getboolean('Run', 'predict')

    @property
    def use_train_weight(self):
        return self._config.getboolean('Run', 'use_train_weight')

    @property
    def loss_ent_weight(self):
        if(self._config.has_option('Run', 'loss_ent_weight')):
            return self._config.getfloat('Run', 'loss_ent_weight')
        else:
            return 0.5

    @property
    def loss_rel_weight(self):
        if (self._config.has_option('Run', 'loss_rel_weight')):
            return self._config.getfloat('Run', 'loss_rel_weight')
        else:
            return 0.5

    #
    @property
    def model(self):
        # model = 'mrctplink'  # mrc4ere,mrctplink
        return self._config.get('Run', 'model')



    @property
    def ent_matrix_label(self):
        # ent_matrix_label = True  # mrc4ere不起作用，但mrctplink中均为matrix,如果关，则ent为BIOES的序列标注，不mix才有作用
        return self._config.getboolean('Run', 'ent_matrix_label')

    @property
    def mix_ent_rel(self):
        # mix_ent_rel = True  # 是否共用解码器fc
        return self._config.getboolean('Run', 'mix_ent_rel')

    @property
    def rel_tail_head(self):
        # rel_tail_head = False  # True:rel区分beg,end,False：rel仅有beg预测
        return self._config.getboolean('Run', 'rel_tail_head')

    @property
    def multi_decoder(self):
        # multi_decoder = 1  # 1:只有一个解码器,问题区分head/tail(如果有的话）,2:[beg_label,end_label],ent的第二个解码器不用全O
        return self._config.getint('Run', 'multi_decoder')

    @property
    def pool_output(self):
        return self._config.get('Run', 'pool_output')

    @property
    def before_relcls_avg(self):
        if (self._config.has_option('Run', 'before_relcls_avg')):
            return self._config.getboolean('Run', 'before_relcls_avg')
        else:
            return False

    @property
    def filter_use_last_bert(self):
        if (self._config.has_option('Run', 'filter_use_last_bert')):
            return self._config.getboolean('Run', 'filter_use_last_bert')
        else:
            return False

    @property
    def dataname(self):
        return self._config.get('Data', 'dataname')

    @property
    def max_seq_length(self):
        return self._config.getint('Data', 'max_seq_length')

    @property
    def max_query_length(self):
        return self._config.getint('Data', 'max_query_length')

    @property
    def doc_stride(self):
        return self._config.getint('Data', 'doc_stride')#

    @property
    def unused_flag(self):
        if (self._config.has_option('Data', 'unused_flag')):
            return self._config.getboolean('Data', 'unused_flag')
        else:
            return False




    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')



    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')


    def set_dev_file(self,dev_file):
        self._config.set('Data', 'dev_file',dev_file)

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    def set_test_file(self,test_file):
        self._config.set('Data', 'test_file',test_file)

    @property
    def output_dir(self):
        return self._config.get('Save', 'output_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def result_dir(self):
        return self._config.get('Save', 'result_dir')

    @property
    def tb_log_dir(self):
        return self._config.get('Save', 'tb_log_dir')

    @property
    def predict_model_path(self):
        return self._config.get('Save', 'predict_model_path')

    def copy_config(self,dir,path):
        self._config.write(open(dir+'/'+path, "w", encoding="utf8"))
        self._config.set('Run', 'predict','True')
        self._config.set('Run', 'train', 'False')
        self._config.write(open(dir + '/predict_' + path, "w", encoding="utf8"))
        self._config.set('Run', 'predict', 'False')
        self._config.set('Run', 'train','True')






if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    print(config.dev_file)
    config.set_dev_file('../datasets')
    print(config.dev_file)
