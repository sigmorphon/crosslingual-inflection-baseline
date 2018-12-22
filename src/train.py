'''
train
'''
import argparse
import glob
import os
import random
import re
from functools import partial
from math import ceil

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import dataloader
import model
import util
from model import decode_greedy

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


class Data(util.NamedEnum):
    sigmorphon19task1 = 'sigmorphon19task1'


class Arch(util.NamedEnum):
    soft = 'soft'  # soft attention without input-feeding
    hard = 'hard'  # hard attention with dynamic programming without input-feeding
    hmm = 'hmm'  # 0th-order hard attention without input-feeding
    hmmfull = 'hmmfull'  # 1st-order hard attention without input-feeding


DEV = 'dev'
TEST = 'test'


def get_args():
    '''
    get_args
    '''
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', required=True, type=Data, choices=list(Data))
    parser.add_argument('--train', required=True, nargs='+')
    parser.add_argument('--dev', required=True)
    parser.add_argument('--test', default=None, type=str)
    parser.add_argument('--model', required=True, help='dump model filename')
    parser.add_argument('--load', default='', help='load model and continue training; with `smart`, recover training automatically')
    parser.add_argument('--bs', default=20, type=int, help='training batch size')
    parser.add_argument('--epochs', default=20, type=int, help='maximum training epochs')
    parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adadelta', 'Adam'])
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD')
    parser.add_argument('--estop', default=1e-8, type=float, help='early stopping criterion')
    parser.add_argument('--cooldown', default=0, type=int, help='cooldown of `ReduceLROnPlateau`')
    parser.add_argument('--max_norm', default=0, type=float, help='gradient clipping max norm')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout prob')
    parser.add_argument('--embed_dim', default=100, type=int, help='embedding dimension')
    parser.add_argument('--src_layer', default=1, type=int, help='source encoder number of layers')
    parser.add_argument('--trg_layer', default=1, type=int, help='target decoder number of layers')
    parser.add_argument('--src_hs', default=200, type=int, help='source encoder hidden dimension')
    parser.add_argument('--trg_hs', default=200, type=int, help='target decoder hidden dimension')
    parser.add_argument('--arch', required=True, type=Arch, choices=list(Arch))
    parser.add_argument('--wid_siz', default=11, type=int, help='maximum transition in 1st-order hard attention')
    parser.add_argument('--loglevel', default='info', choices=['info', 'debug'])
    parser.add_argument('--saveall', default=False, action='store_true', help='keep all models')
    parser.add_argument('--mono', default=False, action='store_true', help='enforce monotonicity')
    parser.add_argument('--bestacc', default=False, action='store_true', help='select model by accuracy only')
    # yapf: enable
    return parser.parse_args()


class Trainer(object):
    '''docstring for Trainer.'''

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.data = None
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.min_lr = 0
        self.scheduler = None
        self.evaluator = None
        self.last_devloss = float('inf')
        self.models = list()

    def load_data(self, dataset, train, dev, test=None):
        assert self.data is None
        logger = self.logger
        # yapf: disable
        if dataset == Data.sigmorphon19task1:
            assert isinstance(train, list) and len(train) == 2
            self.data = dataloader.TagSIGMORPHON2019Task1(train, dev, test)
        else:
            raise ValueError
        # yapf: enable
        logger.info('src vocab size %d', self.data.source_vocab_size)
        logger.info('trg vocab size %d', self.data.target_vocab_size)
        logger.info('src vocab %r', self.data.source[:500])
        logger.info('trg vocab %r', self.data.target[:500])

    def build_model(self, opt):
        assert self.model is None
        params = dict()
        params['src_vocab_size'] = self.data.source_vocab_size
        params['trg_vocab_size'] = self.data.target_vocab_size
        params['embed_dim'] = opt.embed_dim
        params['dropout_p'] = opt.dropout
        params['src_hid_size'] = opt.src_hs
        params['trg_hid_size'] = opt.trg_hs
        params['src_nb_layers'] = opt.src_layer
        params['trg_nb_layers'] = opt.trg_layer
        params['nb_attr'] = self.data.nb_attr
        params['wid_siz'] = opt.wid_siz
        params['src_c2i'] = self.data.source_c2i
        params['trg_c2i'] = self.data.target_c2i
        params['attr_c2i'] = self.data.attr_c2i
        mono = True
        # yapf: disable
        model_classfactory = {
            (Arch.soft, not mono): model.TagTransducer,
            (Arch.hard, not mono): model.TagHardAttnTransducer,
            (Arch.hmm, mono): model.MonoTagHMMTransducer,
            (Arch.hmmfull, not mono): model.TagFullHMMTransducer,
            (Arch.hmmfull, mono): model.MonoTagFullHMMTransducer
        }
        # yapf: enable
        model_class = model_classfactory[(opt.arch, opt.mono)]
        self.model = model_class(**params)
        self.logger.info('number of attribute %d', self.model.nb_attr)
        self.logger.info('dec 1st rnn %r', self.model.dec_rnn.layers[0])
        self.logger.info('number of parameter %d',
                         self.model.count_nb_params())
        self.model = self.model.to(self.device)

    def load_model(self, model):
        assert self.model is None
        self.logger.info('load model in %s', model)
        self.model = torch.load(open(model, mode='rb'), map_location=self.device)
        self.model = self.model.to(self.device)
        epoch = int(model.split('_')[-1])
        return epoch

    def smart_load_model(self, model_prefix):
        assert self.model is None
        models = []
        for model in glob.glob(f'{model_prefix}.nll*'):
            res = re.findall(r'\w*_\d+\.?\d*', model)
            loss_, evals_, epoch_ = res[0].split('_'), res[1:-1], res[-1].split('_')
            assert loss_[0] == 'nll' and epoch_[0] == 'epoch'
            loss, epoch = float(loss_[1]), int(epoch_[1])
            evals = []
            for ev in evals_:
                ev = ev.split('_')
                evals.append(util.Eval(ev[0], ev[0], float(ev[1])))
            models.append((epoch, (model, loss, evals)))
        self.models = [x[1] for x in sorted(models)]
        return self.load_model(self.models[-1][0])

    def setup_training(self, optimizer, lr, min_lr, momentum, cooldown):
        assert self.model is not None
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr, momentum=momentum)
        elif optimizer == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        else:
            raise ValueError
        self.min_lr = min_lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            'min',
            patience=0,
            cooldown=cooldown,
            factor=0.5,
            min_lr=min_lr)
        self.setup_evalutator()

    def save_training(self, model_fp):
        save_objs = (self.optimizer.state_dict(), self.scheduler.state_dict())
        torch.save(save_objs, open(f'{model_fp}.progress', 'wb'))

    def load_training(self, model_fp):
        assert self.model is not None
        optimizer_state, scheduler_state = torch.load(
            open(f'{model_fp}.progress', 'rb'))
        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(scheduler_state)

    def setup_evalutator(self):
        self.evaluator = util.BasicEvaluator()

    def train(self, epoch_idx, batch_size, max_norm):
        logger, model, data = self.logger, self.model, self.data
        logger.info('At %d-th epoch with lr %f.', epoch_idx,
                    self.optimizer.param_groups[0]['lr'])
        model.train()
        nb_train_batch = ceil(data.nb_train / batch_size)
        for src, src_mask, trg, _ in tqdm(
                data.train_batch_sample(batch_size), total=nb_train_batch):
            out = model(src, src_mask, trg)
            loss = model.loss(out, trg[1:])
            self.optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            logger.debug('loss %f with total grad norm %f', loss,
                         util.grad_norm(model.parameters()))
            self.optimizer.step()

    def iterate_batch(self, mode, batch_size):
        if mode == 'dev':
            return self.data.dev_batch_sample, ceil(
                self.data.nb_dev / batch_size)
        elif mode == 'test':
            return self.data.test_batch_sample, ceil(
                self.data.nb_test / batch_size)
        else:
            raise ValueError(f'wrong mode: {mode}')

    def calc_loss(self, mode, batch_size, epoch_idx=-1):
        self.model.eval()
        sampler, nb_batch = self.iterate_batch(mode, batch_size)
        loss, cnt = 0, 0
        for src, src_mask, trg, _ in tqdm(sampler(batch_size), total=nb_batch):
            out = self.model(src, src_mask, trg)
            loss += self.model.loss(out, trg[1:]).item()
            cnt += 1
        loss = loss / cnt
        self.logger.info(
            'Average %s loss value per instance is %f at the end of epoch %d',
            mode, loss, epoch_idx)
        return loss

    def iterate_instance(self, mode):
        if mode == 'dev':
            return self.data.dev_sample, self.data.nb_dev
        elif mode == 'test':
            return self.data.test_sample, self.data.nb_test
        else:
            raise ValueError(f'wrong mode: {mode}')

    def evaluate(self, mode, epoch_idx=-1, decode_fn=decode_greedy):
        self.model.eval()
        sampler, nb_instance = self.iterate_instance(mode)
        results = self.evaluator.evaluate_all(sampler, nb_instance, self.model,
                                              decode_fn)
        for result in results:
            self.logger.info('%s %s is %f at the end of epoch %d', mode,
                             result.long_desc, result.res, epoch_idx)
        return results

    def decode(self, mode, write_fp, decode_fn=decode_greedy):
        self.model.eval()
        cnt = 0
        sampler, nb_instance = self.iterate_instance(mode)
        with open(f'{write_fp}.{mode}.guess', 'w') as out_fp, \
             open(f'{write_fp}.{mode}.gold', 'w') as trg_fp:
            for src, trg in tqdm(sampler(), total=nb_instance):
                pred, _ = decode_fn(self.model, src)
                trg = self.data.decode_target(trg)[1:-1]
                pred = self.data.decode_target(pred)
                out_fp.write(f'{"".join(pred)}\n')
                trg_fp.write(f'{"".join(trg)}\n')
                cnt += 1
        self.logger.info(f'finished decoding {cnt} {mode} instance')

    def update_lr_and_stop_early(self, epoch_idx, devloss, estop):
        prev_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(devloss)
        curr_lr = self.optimizer.param_groups[0]['lr']

        stop_early = True
        if (self.last_devloss - devloss) < estop and \
            prev_lr == curr_lr == self.min_lr:
            self.logger.info(
                'Early stopping triggered with epoch %d (previous dev loss: %f, current: %f)',
                epoch_idx, self.last_devloss, devloss)
            stop_status = stop_early
        else:
            stop_status = not stop_early
        self.last_devloss = devloss
        return stop_status

    def save_model(self, epoch_idx, devloss, eval_res, model_fp):
        eval_tag = '.'.join(['{}_{}'.format(e.desc, e.res) for e in eval_res])
        fp = model_fp + '.nll_{:.4f}.{}.epoch_{}'.format(
            devloss, eval_tag, epoch_idx)
        torch.save(self.model, open(fp, 'wb'))
        self.models.append((fp, devloss, eval_res))

    def reload_and_test(self, model_fp, batch_size, best_acc):
        best_fp, _, best_res = self.models[0]
        best_acc_fp, _, best_acc = self.models[0]
        best_devloss_fp, best_devloss, _ = self.models[0]
        for fp, devloss, res in self.models:
            # [acc, edit distance ]
            if res[0].res >= best_res[0].res and res[1].res <= best_res[1].res:
                best_fp, best_res = fp, res
            if res[0].res >= best_acc[0].res:
                best_acc_fp, best_acc = fp, res
            if devloss <= best_devloss:
                best_devloss_fp, best_devloss = fp, devloss
        self.model = None
        if best_acc:
            best_fp = best_acc_fp
        self.logger.info(f'loading {best_fp} for testing')
        self.load_model(best_fp)
        self.logger.info('decoding dev set')
        self.decode(DEV, f'{model_fp}.decode')
        if self.data.test_file is not None:
            self.calc_loss(TEST, batch_size)
            self.logger.info('decoding test set')
            self.decode(TEST, f'{model_fp}.decode')
            results = self.evaluate(TEST)
            results = ' '.join([f'{r.desc} {r.res}' for r in results])
            self.logger.info(f'TEST {model_fp.split("/")[-1]} {results}')
        return set([best_fp])

    def cleanup(self, saveall, save_fps, model_fp):
        if not saveall:
            for fp, _, _ in self.models:
                if fp in save_fps:
                    continue
                os.remove(fp)
        os.remove(f'{model_fp}.progress')


def main():
    '''
    main
    '''
    opt = get_args()
    util.maybe_mkdir(opt.model)
    logger = util.get_logger(opt.model + '.log', log_level=opt.loglevel)
    for key, value in vars(opt).items():
        logger.info('command line argument: %s - %r', key, value)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    trainer = Trainer(logger)
    trainer.load_data(opt.dataset, opt.train, opt.dev, test=opt.test)
    if opt.load and opt.load != '0':
        if os.path.isfile(opt.load):
            start_epoch = trainer.load_model(opt.load) + 1
        elif opt.load == 'smart':
            start_epoch = trainer.smart_load_model(opt.model) + 1
        else:
            raise ValueError
        logger.info('continue training from epoch %d', start_epoch)
        trainer.setup_training(opt.optimizer, opt.lr, opt.min_lr, opt.momentum,
                               opt.cooldown)
        trainer.load_training(opt.model)
    else:
        start_epoch = 0
        trainer.build_model(opt)
        trainer.setup_training(opt.optimizer, opt.lr, opt.min_lr, opt.momentum,
                               opt.cooldown)

    for epoch_idx in range(start_epoch, start_epoch + opt.epochs):
        trainer.train(epoch_idx, opt.bs, opt.max_norm)
        with torch.no_grad():
            devloss = trainer.calc_loss(DEV, opt.bs, epoch_idx)
            eval_res = trainer.evaluate(DEV, epoch_idx)
        if trainer.update_lr_and_stop_early(epoch_idx, devloss, opt.estop):
            break
        trainer.save_model(epoch_idx, devloss, eval_res, opt.model)
        trainer.save_training(opt.model)
    save_fps = trainer.reload_and_test(opt.model, opt.bs, opt.bestacc)
    trainer.cleanup(opt.saveall, save_fps, opt.model)


if __name__ == '__main__':
    main()
