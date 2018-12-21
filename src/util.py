import logging
import os
import random
import string
import sys
from collections import namedtuple
from enum import Enum
from functools import partial

import numpy as np
from tqdm import tqdm

from dataloader import BOS_IDX, EOS_IDX

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


class NamedEnum(Enum):
    def __str__(self):
        return self.value


def log_grad_norm(self, grad_input, grad_output, logger=None):
    try:
        logger.debug('')
        logger.debug('Inside %r backward', self.__class__.__name__)
        logger.debug('grad_input size: %r', grad_input[0].size())
        logger.debug('grad_output size: %r', grad_output[0].size())
        logger.debug('grad_input norm: %r', grad_input[0].detach().norm())
    except:
        pass


def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.detach().norm(norm_type)
            total_norm += param_norm**norm_type
        total_norm = total_norm**(1. / norm_type)
    return total_norm


def maybe_mkdir(filename):
    '''
    maybe mkdir
    '''
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def get_logger(log_file, log_level='info'):
    '''
    create logger and output to file and stdout
    '''
    assert log_level in ['info', 'debug']
    fmt = '%(asctime)s %(levelname)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logger = logging.getLogger()
    log_level = {'info': logging.INFO, 'debug': logging.DEBUG}[log_level]
    logger.setLevel(log_level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(stream)
    filep = logging.FileHandler(log_file, mode='a')
    filep.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(filep)
    return logger


def get_temp_log_filename(prefix='exp', dir='scratch/explog'):
    id = id_generator()
    fp = f'{dir}/{prefix}-{id}'
    maybe_mkdir(fp)
    return fp


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


Eval = namedtuple('Eval', 'desc long_desc res')


class Evaluator(object):
    pass


class BasicEvaluator(Evaluator):

    def evaluate(self, predict, ground_truth):
        '''
        evaluate single instance
        '''
        correct = 1
        if len(predict) == len(ground_truth):
            for elem1, elem2 in zip(predict, ground_truth):
                if elem1 != elem2:
                    correct = 0
                    break
        else:
            correct = 0
        dist = edit_distance(predict, ground_truth)
        return correct, dist

    def evaluate_all(self, data_iter, nb_data, model, decode_fn):
        '''
        evaluate all instances
        '''
        correct, distance, nb_sample = 0, 0, 0
        for src, trg in tqdm(data_iter(), total=nb_data):
            pred, _ = decode_fn(model, src)
            nb_sample += 1
            trg = trg.view(-1).tolist()
            trg = [x for x in trg if x != BOS_IDX and x != EOS_IDX]
            corr, dist = self.evaluate(pred, trg)
            correct += corr
            distance += dist
        acc = round(correct / nb_sample * 100, 4)
        distance = round(distance / nb_sample, 4)
        return [
            Eval('acc', 'accuracy', acc),
            Eval('dist', 'average edit distance', distance)
        ]


def edit_distance(str1, str2):
    '''Simple Levenshtein implementation for evalm.'''
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + dg)
    return int(table[len(str2)][len(str1)])