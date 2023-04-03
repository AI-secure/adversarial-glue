#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import logging
import json
import re
from random import random
from functools import reduce, partial

import numpy as np
import logging
#from visualdl import LogWriter

from pathlib import Path
import paddle as P
from paddle.nn import functional as F
from propeller import log
import propeller.paddle as propeller

#from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
#from ernie.optimization import AdamW, LinearDecay
from utils import create_if_not_exists, get_warmup_and_linear_decay

from datasets import load_dataset, load_metric

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

parser = propeller.ArgumentParser('classify model with ERNIE')
parser.add_argument(
    '--task',
    type=str,
    default='sst2',
    choices=['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli'],
    help='task to do')
parser.add_argument(
    '--from_pretrained',
    type=Path,
    required=True,
    help='pretrained model directory or tag')
parser.add_argument(
    '--max_seqlen',
    type=int,
    default=128,
    help='max sentence length, should not greater than 512')
parser.add_argument('--bsz', type=int, default=32, help='batchsize')
parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='data directory includes train / develop data')
parser.add_argument('--epoch', type=int, default=3, help='epoch')
parser.add_argument(
    '--max_steps',
    type=int,
    # required=True,
    help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
parser.add_argument('--warmup_proportion', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument(
    '--save_dir', type=Path, required=True, help='model output directory')
parser.add_argument(
    '--wd', type=float, default=0.01, help='weight decay, aka L2 regularizer')
parser.add_argument(
    '--init_checkpoint',
    type=str,
    default=None,
    help='checkpoint to warm start from')

parser.add_argument(
    '--use_amp',
    action='store_true',
    help='only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices'
)

args = parser.parse_args()
env = P.distributed.ParallelEnv()

tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)
#tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

sentence1_key, sentence2_key = task_to_keys[args.task]
datasets = load_dataset("glue", args.task)
metric = load_metric("glue", args.task)
is_regression = args.task == "stsb"
if not is_regression:
    label_list = datasets["train"].features["label"].names
    num_labels = len(label_list)
else:
    num_labels = 1

args.max_steps = int(args.epoch * datasets['train'].num_rows / args.bsz)
num_gpus = 4
args.bsz /= num_gpus
args.bsz = int(args.bsz)
log.debug('max_steps = ' + str(args.max_steps))

if sentence2_key is None:
    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn(
            'seg_a',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
        propeller.data.LabelColumn('label'),
    ])
else:
    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn(
            'seg_a',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
        propeller.data.TextColumn(
            'seg_b',
            unk_id=tokenizer.unk_id,
            vocab_dict=tokenizer.vocab,
            tokenizer=tokenizer.tokenize),
        propeller.data.LabelColumn('label', is_regression=is_regression),
    ])


def map_fn_1s(seg_a, label):
    seg_a, _ = tokenizer.truncate(seg_a, [], seqlen=args.max_seqlen)
    sentence, segments = tokenizer.build_for_ernie(seg_a, [])
    return sentence, segments, label


def map_fn_2s(seg_a, seg_b, label):
    seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=args.max_seqlen)
    sentence, segments = tokenizer.build_for_ernie(seg_a, seg_b)
    return sentence, segments, label


train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'),
                                            shuffle=True, repeat=True, use_gz=False, shard=True) \
                               .map(map_fn_1s if sentence2_key is None else map_fn_2s) \
                               .padded_batch(args.bsz)

dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'),
                                        shuffle=False, repeat=False, use_gz=False) \
                               .map(map_fn_1s if sentence2_key is None else map_fn_2s) \
                               .padded_batch(args.bsz)

if args.task == 'mnli':
    dev_mm_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir.replace('mnli', 'mnli-mm'), 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn_1s if sentence2_key is None else map_fn_2s) \
                                   .padded_batch(args.bsz)

shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen], [-1])
types = ('int64', 'int64', 'int64')

P.distributed.init_parallel_env()
model = ErnieModelForSequenceClassification.from_pretrained(
    args.from_pretrained, num_labels=num_labels, name='')

if args.init_checkpoint is not None:
    log.info('loading checkpoint from %s' % args.init_checkpoint)
    sd = P.load(args.init_checkpoint)
    model.set_state_dict(sd)

model = P.DataParallel(model)

g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
param_name_to_exclue_from_weight_decay = re.compile(
    r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

lr_scheduler = P.optimizer.lr.LambdaDecay(
    args.lr,
    get_warmup_and_linear_decay(args.max_steps,
                                int(args.warmup_proportion * args.max_steps)))
opt = P.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
    weight_decay=args.wd,
    grad_clip=g_clip)
scaler = P.amp.GradScaler(enable=args.use_amp)
step = 0
create_if_not_exists(args.save_dir)

#with LogWriter(logdir=str(create_if_not_exists(args.save_dir / 'vdl-%d' % env.dev_id))) as log_writer:
with P.amp.auto_cast(enable=args.use_amp):
    for ids, sids, label in P.io.DataLoader(
            train_ds, places=P.CUDAPlace(env.dev_id), batch_size=None):
        step += 1
        loss, _ = model(ids, sids, labels=label)
        if is_regression:
            loss = F.mse_loss(_, P.cast(label.reshape((-1, 1)), dtype='float32'))
        loss = scaler.scale(loss)
        loss.backward()
        scaler.minimize(opt, loss)
        model.clear_gradients()
        lr_scheduler.step()

        # do logging
        if step % 10 == 0:
            _lr = lr_scheduler.get_lr()
            if args.use_amp:
                _l = (loss / scaler._scale).numpy()
                msg = '[rank-%d][step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                    env.dev_id, step, _l, _lr, scaler._scale.numpy())
            else:
                _l = loss.numpy()
                msg = '[rank-%d][step-%d] train loss %.5f lr %.3e' % (
                    env.dev_id, step, _l, _lr)
            log.debug(msg)
            #log_writer.add_scalar('loss', _l, step=step)
            #log_writer.add_scalar('lr', _lr, step=step)

        # do saving
        if step % 100 == 0 and env.dev_id == 0:
            if args.task == 'mnli':
                eval_tasks = {args.task: dev_ds, 'mnli-mm': dev_mm_ds}
            else:
                eval_tasks = {args.task: dev_ds}
            for task, ds in eval_tasks.items():
                preds = []
                labels = []
                with P.no_grad():
                    model.eval()
                    for d in P.io.DataLoader(
                            dev_ds, places=P.CUDAPlace(env.dev_id),
                            batch_size=None):
                        ids, sids, label = d
                        loss, logits = model(ids, sids, labels=label)
                        # print('\n'.join(map(str, logits.numpy().tolist())))
                        labels += label.numpy().tolist()
                        preds += np.squeeze(logits.numpy()).tolist() if is_regression else logits.argmax(
                            -1).numpy().tolist()
                    model.train()
                eval_result = metric.compute(predictions=preds, references=labels)
                for key, value in sorted(eval_result.items()):
                    # log_writer.add_scalar(task + '-eval/' + key, value, step=step)
                    log.debug(task + '-' + key + ' %.4f' % value)
            if args.save_dir is not None:
                P.save(model.state_dict(), args.save_dir / 'ckpt.bin')
        # exit 
        if step > args.max_steps:
            if env.dev_id == 0:
                if args.task == 'mnli':
                    eval_tasks = {args.task: dev_ds, 'mnli-mm': dev_mm_ds}
                else:
                    eval_tasks = {args.task: dev_ds}
                for task, ds in eval_tasks.items():
                    preds = []
                    labels = []
                    with P.no_grad():
                        model.eval()
                        for d in P.io.DataLoader(
                                dev_ds, places=P.CUDAPlace(env.dev_id),
                                batch_size=None):
                            ids, sids, label = d
                            loss, logits = model(ids, sids, labels=label)
                            # print('\n'.join(map(str, logits.numpy().tolist())))
                            labels += label.numpy().tolist()
                            preds += np.squeeze(logits.numpy()).tolist() if is_regression else logits.argmax(
                                -1).numpy().tolist()
                        model.train()
                    eval_result = metric.compute(predictions=preds, references=labels)
                    for key, value in sorted(eval_result.items()):
                        # log_writer.add_scalar(task + '-eval/' + key, value, step=step)
                        log.debug(task + '-' + key + ' %.4f' % value)
            break

if args.save_dir is not None and env.dev_id == 0:
    P.save(model.state_dict(), args.save_dir / 'ckpt.bin')
log.debug('done')
