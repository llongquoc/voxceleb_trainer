#!/usr/bin/python
#-*- coding: utf-8 -*-


import sys
sys.path.insert(0,'..')

from tuneThreshold import *
from SpeakerNet import *
from utils import *
from DatasetLoader import get_data_loader
import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import torch
import glob
import zipfile
import datetime
import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.nn.functional as F
from flask import Flask, request, jsonify


app = Flask(__name__)


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = 'Prepare Data');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=400,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default='softmaxproto',     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default='adam', help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default='steplr', help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
parser.add_argument('--lr_decay',       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument('--hard_prob',      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument('--hard_rank',      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=2,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=400,   help='Number of speakers in the softmax layer, only for softmax-based losses');

## Load
parser.add_argument('--model_path',      type=str,   default='model000000500.model', help='Path for model and logs');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=64,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=True,  help='Log input features')
parser.add_argument('--model',          type=str,   default='ResNetSE34V2',     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default='ASP',  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default='8888', help='Port for distributed training, input as text');
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

parser.add_argument('--threshold',           type=float,   default=-1.0831763744354248,    help='Threshold');
parser.add_argument('--feats_path',     type=str,   default='feats.npy', help='Path for feats file');

args = parser.parse_args();


args.model_save_path = args.model_path + '/model'
n_gpus = torch.cuda.device_count()
args.gpu = gpu

## Load models
s = SpeakerNet(**vars(args));

if args.distributed:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

    torch.cuda.set_device(args.gpu)
    s.cuda(args.gpu)

    s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)
else:
    s = WrappedModel(s).cpu()

## Initialise trainer and data loader
trainer = ModelTrainer(s, **vars(args))

## Load model weights
try:
    trainer.loadParameters(args.model_path);
except:
    raise Exception('Model path is wrong!')
print('Model %s loaded from previous state!'%args.model_path);


def main_worker(file_path):
    if os.path.isfile(file_path):
        raise Exception('File path is wrong!')
    
    feature_vector = create_feature_vectors(trainer, '', [file_path], args.nDataLoaderThread, args.eval_frames))[file_path]
    normalized_vector = F.normalize(feature_vector, p=2, dim=1)
    feats = np.load(args.feats_path, allow_pickle=True)[()]

    max_score = args.threshold
    speaker = ''
    for key, value in feats.items():
        com_feat = F.normalize(value, p=2, dim=1)
        dist = F.pairwise_distance(normalized_vector.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
        score = -1 * np.mean(dist);

        if score >= max_score:
            max_score = score
            speaker = key.split('/')[0]
    return speaker


@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)
    speaker = main_worker(file_name)
    os.remove(file_name)

    result = {'speaker': speaker}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8008', debug=True)