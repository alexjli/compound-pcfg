#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import BertDataset
from torch.utils.data import DataLoader
from utils import *
from models import CompBertPCFG
from torch.nn.init import xavier_uniform_
import datetime


from parallel import BetterDataParallel

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--save_path', default='compound-pcfg.pt', help='where to save the model')

# Model options
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')
parser.add_argument('--prior', type = str, default = "normal", help='prior from which z is drawn')
parser.add_argument('--vpost', type = str, default = "normal", help='distribution used to approximate variational posterior')
parser.add_argument('--dummy', type = int, default = 0, help="used for output checking")
def main(args):
  if args.seed != -1:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
  train_data = BertDataset(args.train_file)
  val_data = BertDataset(args.val_file)
  #train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=2)
  #val_loader = DataLoader(dataset=val_data, shuffle=True, batch_size=2)
  train_sents = train_data.batch_size.sum()
  vocab_size = int(train_data.vocab_size)
  max_len = max(val_data.sents.size(1), train_data.sents.size(1))
  print('Train: %d sents / %d batches, Val: %d sents / %d batches' %
        (train_data.sents.size(0), len(train_data), val_data.sents.size(0), len(val_data)))
  print('Vocab size: %d, Max Sent Len: %d' % (vocab_size, max_len))
  print('Save Path', args.save_path)
  #cuda.set_device(args.gpu)
  model = CompBertPCFG(vocab = vocab_size,
                   state_dim = args.state_dim,
                   t_states = args.t_states,
                   nt_states = args.nt_states,
                   h_dim = args.h_dim,
                   w_dim = args.w_dim,
                   z_dim = args.z_dim,
                   prior = args.prior,
                   vpost = args.vpost)
  # model parallelize
  base_gpu = torch.device('cuda:0')
  model.to(base_gpu)
  model = BetterDataParallel(model)


  for name, param in model.named_parameters():
    if param.dim() > 1:
      xavier_uniform_(param)
  print("model architecture")
  print(model)
  model.train()
  # model.cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))
  best_val_ppl = 1e5
  best_val_f1 = 0
  epoch = 0
  num_gpus = torch.cuda.device_count()
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    train_kl = 0.
    num_sents = 0.
    num_words = 0.
    all_stats = [[0., 0., 0.]]
    b = 0
    n_gpus = torch.cuda.device_count()
    for i in np.random.permutation(len(train_data)):
      b += 1
      #current_device = next(model.parameters()).device
      sents, bert, length, batch_size, _, gold_spans, gold_binary_trees, _ = train_data[i]
      if length > args.max_length or length == 1: #length filter based on curriculum
        continue
      if batch_size < n_gpus or batch_size % n_gpus != 0:
        continue
      with open("batchsize.log", 'a') as fp:
          fp.write(str(batch_size) + "\n")
      #if batch_size == 0 or batch_size % num_gpus != 0:   #gpu paraellization filter
      #  continue
      # sents = sents.cuda()
      #sents = sents.to(base_gpu)
      optimizer.zero_grad()
      nll, kl, binary_matrix, argmax_spans = model(sents, bert, argmax=True)
      (nll+kl).mean().backward()
      train_nll += nll.sum().item()
      train_kl += kl.sum().item()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      optimizer.step()
      num_sents += batch_size
      num_words += batch_size * (length + 1) # we implicitly generate </s> so we explicitly count it
      for bb in range(batch_size):
        span_b = [(a[0].item(), a[1].item()) for a in argmax_spans[bb]] #ignore labels
        span_b_set = set(span_b[:-1])
        update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
      if b % args.print_every == 0:
        all_f1 = get_f1(all_stats)
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        gparam_norm = sum([p.grad.norm()**2 for p in model.parameters()
                           if p.grad is not None]).item()**0.5
        log_str = 'Epoch: %d, Batch: %d/%d, |Param|: %.6f, |GParam|: %.2f,  LR: %.4f, ' + \
                  'ReconPPL: %.2f, KL: %.4f, PPLBound: %.2f, ValPPL: %.2f, ValF1: %.2f, ' + \
                  'CorpusF1: %.2f, Throughput: %.2f examples/sec'
        print(log_str %
              (epoch, b, len(train_data), param_norm, gparam_norm, args.lr,
               np.exp(train_nll / num_words), train_kl /num_sents,
               np.exp((train_nll + train_kl)/num_words), best_val_ppl, best_val_f1,
               all_f1[0], num_sents / (time.time() - start_time)))
        # print an example parse
        tree = get_tree_from_binary_matrix(binary_matrix[0], length)
        action = get_actions(tree)
        sent_str = [train_data.idx2word[word_idx] for word_idx in list(sents[0].cpu().numpy())]
        print("Pred Tree: %s" % get_tree(action, sent_str))
        print("Gold Tree: %s" % get_tree(gold_binary_trees[0], sent_str))
        with open("dummy_output_bert%d.log" % args.dummy, 'a') as fp:
          fp.write(log_str %
              (epoch, b, len(train_data), param_norm, gparam_norm, args.lr,
               np.exp(train_nll / num_words), train_kl /num_sents,
               np.exp((train_nll + train_kl)/num_words), best_val_ppl, best_val_f1,
               all_f1[0], num_sents / (time.time() - start_time)) + "\n")
          e = datetime.datetime.now()
          fp.write("The time is now: = %s:%s:%s" % (e.hour, e.minute, e.second) + "\n")



    args.max_length = min(args.final_max_length, args.max_length + args.len_incr)
    print('--------------------------------')
    print('Checking validation perf...')
    val_ppl, val_f1 = eval(val_data, model)
    print('--------------------------------')

    if val_ppl < best_val_ppl:
      best_val_ppl = val_ppl
      best_val_f1 = val_f1
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu(),
        'word2idx': train_data.word2idx,
        'idx2word': train_data.idx2word
      }
      print('Saving checkpoint to %s' % args.save_path)
      torch.save(checkpoint, args.save_path)
      # model.cuda()
      model.to(base_gpu)

def eval(data, model):
  model.eval()
  # no need to parallelize because input model should be parallel
  num_sents = 0
  num_words = 0
  total_nll = 0.
  total_kl = 0.
  corpus_f1 = [0., 0., 0.]
  sent_f1 = []
  n_gpus = torch.cuda.device_count()
  with torch.no_grad():
    for i in range(len(data)):
      sents, bert, length, batch_size, _, gold_spans, gold_binary_trees, other_data = data[i]
      if length == 1:
        continue
      if batch_size < n_gpus or batch_size % n_gpus != 0:
        continue
      #if batch_size == 0 or batch_size % num_gpus != 0:   #gpu paraellization filter
      #  continue
      #sents = sents.cuda()
      # note that for unsuperised parsing, we should do model(sents, argmax=True, use_mean = True)
      # but we don't for eval since we want a valid upper bound on PPL for early stopping
      # see eval.py for proper MAP inference
      nll, kl, binary_matrix, argmax_spans = model(sents, bert, argmax=True)
      total_nll += nll.sum().item()
      total_kl  += kl.sum().item()
      num_sents += batch_size
      num_words += batch_size*(length +1) # we implicitly generate </s> so we explicitly count it
      for b in range(batch_size):
        span_b = [(a[0].item(), a[1].item()) for a in argmax_spans[b]] #ignore labels
        span_b_set = set(span_b[:-1])
        gold_b_set = set(gold_spans[b][:-1])
        tp, fp, fn = get_stats(span_b_set, gold_b_set)
        corpus_f1[0] += tp
        corpus_f1[1] += fp
        corpus_f1[2] += fn
        # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py

        model_out = span_b_set
        std_out = gold_b_set
        overlap = model_out.intersection(std_out)
        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)
        if len(std_out) == 0:
          reca = 1.
          if len(model_out) == 0:
            prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        sent_f1.append(f1)
  tp, fp, fn = corpus_f1
  prec = tp / (tp + fp)
  recall = tp / (tp + fn)
  corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
  sent_f1 = np.mean(np.array(sent_f1))
  recon_ppl = np.exp(total_nll / num_words)
  ppl_elbo = np.exp((total_nll + total_kl)/num_words)
  kl = total_kl /num_sents
  print('ReconPPL: %.2f, KL: %.4f, PPL (Upper Bound): %.2f' %
        (recon_ppl, kl, ppl_elbo))
  print('Corpus F1: %.2f, Sentence F1: %.2f' %
        (corpus_f1*100, sent_f1*100))
  model.train()
  return ppl_elbo, sent_f1*100

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
