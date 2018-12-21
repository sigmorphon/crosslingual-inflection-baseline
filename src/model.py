'''
all model
'''
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import BOS_IDX, EOS_IDX, PAD_IDX

EPSILON = 1e-7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StackedLSTM(nn.Module):
    '''
    step-by-step stacked LSTM
    '''

    def __init__(self, input_siz, rnn_siz, nb_layers, dropout):
        '''
        init
        '''
        super().__init__()
        self.nb_layers = nb_layers
        self.rnn_siz = rnn_siz
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(nb_layers):
            self.layers.append(nn.LSTMCell(input_siz, rnn_siz))
            input_siz = rnn_siz

    def get_init_hx(self, batch_size):
        '''
        initial h0
        '''
        h_0_s, c_0_s = [], []
        for _ in range(self.nb_layers):
            h_0 = torch.zeros((batch_size, self.rnn_siz), device=DEVICE)
            c_0 = torch.zeros((batch_size, self.rnn_siz), device=DEVICE)
            h_0_s.append(h_0)
            c_0_s.append(c_0)
        return (h_0_s, c_0_s)

    def forward(self, input, hidden):
        '''
        dropout after all output except the last one
        '''
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = self.dropout(h_1_i)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Attention(nn.Module):
    '''
    attention with mask
    '''

    def forward(self, ht, hs, mask, weighted_ctx=True):
        '''
        ht: batch x ht_dim
        hs: (seq_len x batch x hs_dim, seq_len x batch x ht_dim)
        mask: seq_len x batch
        '''
        hs, hs_ = hs
        # seq_len, batch, _ = hs.size()
        hs = hs.transpose(0, 1)
        hs_ = hs_.transpose(0, 1)
        # hs: batch x seq_len x hs_dim
        # hs_: batch x seq_len x ht_dim
        # hs_ = self.hs2ht(hs)
        # Alignment/Attention Function
        # batch x ht_dim x 1
        ht = ht.unsqueeze(2)
        # batch x seq_len
        score = torch.bmm(hs_, ht).squeeze(2)
        # attn = F.softmax(score, dim=-1)
        attn = F.softmax(score, dim=-1) * mask.transpose(0, 1) + EPSILON
        attn = attn / attn.sum(-1, keepdim=True)

        # Compute weighted sum of hs by attention.
        # batch x 1 x seq_len
        attn = attn.unsqueeze(1)
        if weighted_ctx:
            # batch x hs_dim
            weight_hs = torch.bmm(attn, hs).squeeze(1)
        else:
            weight_hs = None

        return weight_hs, attn


class Transducer(nn.Module):
    '''
    seq2seq with soft attention baseline
    '''

    def __init__(self, *, src_vocab_size, trg_vocab_size, embed_dim,
                 src_hid_size, src_nb_layers, trg_hid_size, trg_nb_layers,
                 dropout_p, src_c2i, trg_c2i, attr_c2i, **kwargs):
        '''
        init
        '''
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embed_dim = embed_dim
        self.src_hid_size = src_hid_size
        self.src_nb_layers = src_nb_layers
        self.trg_hid_size = trg_hid_size
        self.trg_nb_layers = trg_nb_layers
        self.dropout_p = dropout_p
        self.src_c2i, self.trg_c2i, self.attr_c2i = src_c2i, trg_c2i, attr_c2i
        self.src_embed = nn.Embedding(
            src_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.trg_embed = nn.Embedding(
            trg_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.enc_rnn = nn.LSTM(
            embed_dim,
            src_hid_size,
            src_nb_layers,
            bidirectional=True,
            dropout=dropout_p)
        self.dec_rnn = StackedLSTM(embed_dim, trg_hid_size, trg_nb_layers,
                                   dropout_p)
        self.out_dim = trg_hid_size + src_hid_size * 2
        self.scale_enc_hs = nn.Linear(src_hid_size * 2, trg_hid_size)
        self.attn = Attention()
        self.linear_out = nn.Linear(self.out_dim, self.out_dim)
        self.final_out = nn.Linear(self.out_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def encode(self, src_batch):
        '''
        encoder
        '''
        enc_hs, _ = self.enc_rnn(self.dropout(self.src_embed(src_batch)))
        scale_enc_hs = self.scale_enc_hs(enc_hs)
        return enc_hs, scale_enc_hs

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        '''
        decode step
        '''
        h_t, hidden = self.dec_rnn(input_, hidden)
        ctx, attn = self.attn(h_t, enc_hs, enc_mask)
        # Concatenate the ht and ctx
        # weight_hs: batch x (hs_dim + ht_dim)
        ctx = torch.cat((ctx, h_t), dim=1)
        # ctx: batch x out_dim
        ctx = self.linear_out(ctx)
        ctx = torch.tanh(ctx)
        word_logprob = F.log_softmax(self.final_out(ctx), dim=-1)
        return word_logprob, hidden, attn

    def decode(self, enc_hs, enc_mask, trg_batch):
        '''
        enc_hs: tuple(enc_hs, scale_enc_hs)
        '''
        trg_seq_len = trg_batch.size(0)
        trg_bat_siz = trg_batch.size(1)
        trg_embed = self.dropout(self.trg_embed(trg_batch))
        output = []
        hidden = self.dec_rnn.get_init_hx(trg_bat_siz)
        for idx in range(trg_seq_len - 1):
            input_ = trg_embed[idx, :]
            word_logprob, hidden, _ = self.decode_step(enc_hs, enc_mask,
                                                       input_, hidden)
            output += [word_logprob]
        return torch.stack(output)

    def forward(self, src_batch, src_mask, trg_batch):
        '''
        only for training
        '''
        # trg_seq_len, batch_size = trg_batch.size()
        enc_hs = self.encode(src_batch)
        # output: [trg_seq_len-1, batch_size, vocab_siz]
        output = self.decode(enc_hs, src_mask, trg_batch)
        return output

    def count_nb_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def loss(self, predict, target):
        '''
        compute loss
        '''
        return F.nll_loss(
            predict.view(-1, self.trg_vocab_size),
            target.view(-1),
            ignore_index=PAD_IDX)


HMMState = namedtuple('HMMState', 'init trans emiss')


class HMM(object):
    def __init__(self, nb_states, nb_tokens, initial, transition, emission):
        assert isinstance(initial, torch.Tensor)
        assert isinstance(transition, torch.Tensor)
        assert isinstance(emission, torch.Tensor)
        assert initial.shape[-1] == nb_states
        assert transition.shape[-2:] == (nb_states, nb_states)
        assert emission.shape[-2:] == (nb_states, nb_tokens)
        self.ns = nb_states
        self.V = nb_tokens
        self.initial = initial
        self.transition = transition
        self.emission = emission

    def emiss(self, T, idx, ignore_index=None):
        assert len(idx.shape) == 1
        bs = idx.shape[0]
        idx = idx.view(-1, 1).expand(bs, self.ns).unsqueeze(-1)
        emiss = torch.gather(self.emission[T], -1, idx).view(bs, 1, self.ns)
        if ignore_index is None:
            return emiss
        else:
            idx = idx.view(bs, 1, self.ns)
            mask = (idx != ignore_index).float()
            return emiss * mask

    def p_x(self, seq, ignore_index=None):
        T, bs = seq.shape
        assert self.initial.shape == (bs, 1, self.ns)
        assert self.transition.shape == (T - 1, bs, self.ns, self.ns)
        assert self.emission.shape == (T, bs, self.ns, self.V)
        # fwd = pi * b[:, O[0]]
        # fwd = self.initial * self.emiss(0, seq[0])
        fwd = self.initial + self.emiss(0, seq[0], ignore_index=ignore_index)
        #induction:
        for t in range(T - 1):
            # fwd[t + 1] = np.dot(fwd[t], a) * b[:, O[t + 1]]
            # fwd = torch.bmm(fwd, self.transition[t]) * self.emiss(
            #     t + 1, seq[t + 1])
            fwd = fwd + self.transition[t].transpose(1, 2)
            fwd = fwd.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
            fwd = fwd + self.emiss(
                t + 1, seq[t + 1], ignore_index=ignore_index)
        return fwd


class HMMTransducer(Transducer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.attn

    def loss(self, predict, target):
        assert isinstance(predict, HMMState)
        seq_len = target.shape[0]
        hmm = HMM(predict.init.shape[-1], self.trg_vocab_size, predict.init,
                  predict.trans, predict.emiss)
        loss = hmm.p_x(target, ignore_index=PAD_IDX)
        return -torch.logsumexp(loss, dim=-1).mean() / seq_len

    def decode(self, enc_hs, enc_mask, trg_batch):
        trg_seq_len = trg_batch.size(0)
        trg_bat_siz = trg_batch.size(1)
        trg_embed = self.dropout(self.trg_embed(trg_batch))
        hidden = self.dec_rnn.get_init_hx(trg_bat_siz)

        initial, transition, emission = None, list(), list()
        for idx in range(trg_seq_len - 1):
            input_ = trg_embed[idx, :]
            trans, emiss, hidden = self.decode_step(enc_hs, enc_mask, input_,
                                                    hidden)
            if idx == 0:
                initial = trans[:, 0].unsqueeze(1)
                emission += [emiss]
            else:
                transition += [trans]
                emission += [emiss]
        transition = torch.stack(transition)
        emission = torch.stack(emission)
        return HMMState(initial, transition, emission)

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        src_seq_len, bat_siz = enc_mask.shape
        h_t, hidden = self.dec_rnn(input_, hidden)

        # Concatenate the ht and hs
        # ctx_*: batch x seq_len x (trg_hid_siz+src_hid_size*2)
        ctx_curr = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), enc_hs[0].transpose(
                0, 1)),
            dim=2)

        hs_ = enc_hs[1].transpose(0, 1)
        h_t = h_t.unsqueeze(2)
        score = torch.bmm(hs_, h_t).squeeze(2)
        trans = F.softmax(score, dim=-1) * enc_mask.transpose(0, 1) + EPSILON
        trans = trans / trans.sum(-1, keepdim=True)
        trans = trans.unsqueeze(1).log()
        trans = trans.expand(bat_siz, src_seq_len, src_seq_len)

        ctx = torch.tanh(self.linear_out(ctx_curr))
        # emiss: batch x seq_len x nb_vocab
        emiss = F.log_softmax(self.final_out(ctx), dim=-1)

        return trans, emiss, hidden


class FullHMMTransducer(HMMTransducer):
    def __init__(self, wid_siz, **kwargs):
        super().__init__(**kwargs)
        assert wid_siz % 2 == 1
        self.wid_siz = wid_siz
        self.trans = nn.Linear(self.trg_hid_size * 2, self.wid_siz)

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        src_seq_len, bat_siz = enc_mask.shape
        h_t, hidden = self.dec_rnn(input_, hidden)

        # Concatenate the ht and hs
        # ctx_trans: batch x seq_len x (trg_hid_siz*2)
        ctx_trans = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), enc_hs[1].transpose(
                0, 1)),
            dim=2)
        trans = F.softmax(self.trans(ctx_trans), dim=-1)
        trans_list = trans.split(1, dim=1)
        ws = (self.wid_siz - 1) // 2
        trans_shift = [
            F.pad(t, (-ws + i, src_seq_len - (ws + 1) - i))
            for i, t in enumerate(trans_list)
        ]
        trans = torch.cat(trans_shift, dim=1)
        trans = trans * enc_mask.transpose(0, 1).unsqueeze(1) + EPSILON
        trans = trans / trans.sum(-1, keepdim=True)
        trans = trans.log()

        # Concatenate the ht and hs
        # ctx_emiss: batch x seq_len x (trg_hid_siz+src_hid_size*2)
        ctx_emiss = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), enc_hs[0].transpose(
                0, 1)),
            dim=2)
        ctx = torch.tanh(self.linear_out(ctx_emiss))
        # emiss: batch x seq_len x nb_vocab
        emiss = F.log_softmax(self.final_out(ctx), dim=-1)

        return trans, emiss, hidden


class MonoHMMTransducer(HMMTransducer):
    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        trans, emiss, hidden = super().decode_step(enc_hs, enc_mask, input_,
                                                   hidden)
        trans_mask = torch.ones_like(trans[0]).triu().unsqueeze(0)
        trans_mask = (trans_mask - 1) * -np.log(EPSILON)
        trans = trans + trans_mask
        trans = trans - trans.logsumexp(-1, keepdim=True)
        return trans, emiss, hidden


class HardAttnTransducer(Transducer):
    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        '''
        enc_hs: tuple(enc_hs, scale_enc_hs)
        '''
        src_seq_len = enc_hs[0].size(0)
        h_t, hidden = self.dec_rnn(input_, hidden)

        # ht: batch x trg_hid_dim
        # enc_hs: seq_len x batch x src_hid_dim*2
        # attns: batch x 1 x seq_len
        _, attns = self.attn(h_t, enc_hs, enc_mask, weighted_ctx=False)

        # Concatenate the ht and hs
        # ctx: batch x seq_len x (trg_hid_siz+src_hid_size*2)
        ctx = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), enc_hs[0].transpose(
                0, 1)),
            dim=2)
        # ctx: batch x seq_len x out_dim
        ctx = self.linear_out(ctx)
        ctx = torch.tanh(ctx)

        # word_prob: batch x seq_len x nb_vocab
        word_prob = F.softmax(self.final_out(ctx), dim=-1)
        # word_prob: batch x nb_vocab
        word_prob = torch.bmm(attns, word_prob).squeeze(1)
        return torch.log(word_prob), hidden, attns


class TagTransducer(Transducer):
    def __init__(self, *, nb_attr, **kwargs):
        super().__init__(**kwargs)
        self.nb_attr = nb_attr + 1 if nb_attr > 0 else 0
        if self.nb_attr > 0:
            attr_dim = self.embed_dim // 5
            self.src_embed = nn.Embedding(
                self.src_vocab_size - nb_attr,
                self.embed_dim,
                padding_idx=PAD_IDX)
            # padding_idx is a part of self.nb_attr, so need to +1
            self.attr_embed = nn.Embedding(
                self.nb_attr + 1, attr_dim, padding_idx=PAD_IDX)
            self.merge_attr = nn.Linear(attr_dim * self.nb_attr, attr_dim)
            self.dec_rnn = StackedLSTM(self.embed_dim + attr_dim,
                                       self.trg_hid_size, self.trg_nb_layers,
                                       self.dropout_p)
        else:
            self.dec_rnn = StackedLSTM(self.embed_dim, self.trg_hid_size,
                                       self.trg_nb_layers, self.dropout_p)

    def encode(self, src_batch):
        '''
        encoder
        '''
        if self.nb_attr > 0:
            assert isinstance(src_batch, tuple) and len(src_batch) == 2
            src, attr = src_batch
            bs = src.shape[1]
            new_idx = torch.arange(1, self.nb_attr + 1).expand(bs, -1)
            attr = (
                (attr > 1).float() * new_idx.to(attr.device).float()).long()
            enc_attr = F.relu(
                self.merge_attr(self.attr_embed(attr).view(bs, -1)))
        else:
            src = src_batch
            enc_attr = None
        enc_hs = super().encode(src)
        return enc_hs, enc_attr

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        '''
        decode step
        '''
        enc_hs_, attr = enc_hs
        if attr is not None:
            input_ = torch.cat((input_, attr), dim=1)
        return super().decode_step(enc_hs_, enc_mask, input_, hidden)


class TagHMMTransducer(TagTransducer, HMMTransducer):
    pass


class TagFullHMMTransducer(TagTransducer, FullHMMTransducer):
    pass


class MonoTagHMMTransducer(TagTransducer, MonoHMMTransducer):
    pass


class MonoTagFullHMMTransducer(TagTransducer, MonoHMMTransducer,
                               FullHMMTransducer):
    pass


class TagHardAttnTransducer(TagTransducer, HardAttnTransducer):
    pass


def dummy_mask(seq):
    '''
    create dummy mask (all 1)
    '''
    if isinstance(seq, tuple):
        seq = seq[0]
    assert len(seq.size()) == 1 or (len(seq.size()) == 2 and seq.size(1) == 1)
    return torch.ones_like(seq, dtype=torch.float)


def decode_greedy(transducer,
                  src_sentence,
                  max_len=100,
                  trg_bos=BOS_IDX,
                  trg_eos=EOS_IDX):
    '''
    src_sentence: [seq_len]
    '''
    if isinstance(transducer, HMMTransducer):
        return decode_greedy_hmm(
            transducer,
            src_sentence,
            max_len=max_len,
            trg_bos=BOS_IDX,
            trg_eos=EOS_IDX)
    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)

    output, attns = [], []
    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    for _ in range(max_len):
        word_logprob, hidden, attn = transducer.decode_step(
            enc_hs, src_mask, input_, hidden)
        word = torch.max(word_logprob, dim=1)[1]
        attns.append(attn)
        if word == trg_eos:
            break
        input_ = transducer.dropout(transducer.trg_embed(word))
        output.append(word.item())
    return output, attns


def decode_greedy_hmm(transducer,
                      src_sentence,
                      max_len=100,
                      trg_bos=BOS_IDX,
                      trg_eos=EOS_IDX):
    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)
    T = src_mask.shape[0]

    output, attns = [], []
    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    for idx in range(max_len):
        trans, emiss, hidden = transducer.decode_step(enc_hs, src_mask, input_,
                                                      hidden)
        if idx == 0:
            initial = trans[:, 0].unsqueeze(1)
            attns.append(initial)
            forward = initial
        else:
            attns.append(trans)
            # forward = torch.bmm(forward, trans)
            forward = forward + trans.transpose(1, 2)
            forward = forward.logsumexp(dim=-1, keepdim=True).transpose(1, 2)

        # wordprob = torch.bmm(forward, emiss)
        log_wordprob = forward + emiss.transpose(1, 2)
        log_wordprob = log_wordprob.logsumexp(dim=-1)
        word = torch.max(log_wordprob, dim=-1)[1]
        if word == trg_eos:
            break
        input_ = transducer.dropout(transducer.trg_embed(word))
        output.append(word.item())
        word_idx = word.view(-1, 1).expand(1, T).unsqueeze(-1)
        word_emiss = torch.gather(emiss, -1, word_idx).view(1, 1, T)
        forward = forward + word_emiss
    return output, attns


Beam = namedtuple('Beam', 'seq_len log_prob hidden input partial_sent attn')


def decode_beam_search(transducer,
                       src_sentence,
                       max_len=50,
                       nb_beam=5,
                       norm=True,
                       trg_bos=BOS_IDX,
                       trg_eos=EOS_IDX):
    '''
    src_sentence: [seq_len]
    '''
    assert not isinstance(transducer, HMMTransducer)

    def score(beam):
        '''
        compute score based on logprob
        '''
        assert isinstance(beam, Beam)
        if norm:
            return -beam.log_prob / beam.seq_len
        return -beam.log_prob

    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)

    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    start = Beam(1, 0, hidden, input_, '', [])
    beams = [start]
    finish_beams = []
    for _ in range(max_len):
        next_beams = []
        for beam in sorted(beams, key=score)[:nb_beam]:
            word_logprob, hidden, attn = transducer.decode_step(
                enc_hs, src_mask, beam.input, beam.hidden)
            topk_log_prob, topk_word = word_logprob.topk(nb_beam)
            topk_log_prob = topk_log_prob.view(nb_beam, 1)
            topk_word = topk_word.view(nb_beam, 1)
            for log_prob, word in zip(topk_log_prob, topk_word):
                if word == trg_eos:
                    beam = Beam(beam.seq_len + 1,
                                beam.log_prob + log_prob.item(), None, None,
                                beam.partial_sent, beam.attn + [attn])
                    finish_beams.append(beam)
                else:
                    beam = Beam(
                        beam.seq_len + 1, beam.log_prob + log_prob.item(),
                        hidden, transducer.dropout(transducer.trg_embed(word)),
                        ' '.join([beam.partial_sent,
                                  str(word.item())]), beam.attn + [attn])
                    next_beams.append(beam)
        beams = next_beams
    finish_beams = finish_beams if finish_beams else next_beams
    max_output = sorted(finish_beams, key=score)[0]
    return list(map(int, max_output.partial_sent.split())), max_output.attn