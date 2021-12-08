#!/usr/bin/python
#-*- coding: utf-8 -*-


from SpeakerNet import *
from utils import create_feature_vectors, loadParameters
from DatasetLoader import loadWAV
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
import subprocess
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.nn.functional as F


random.seed(5)


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = 'Prepare Data');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=400,    help='Input length to the network for testing; 0 uses the whole files');

## Training details
parser.add_argument('--trainfunc',      type=str,   default='angleproto',     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default='adam', help='sgd or adam');

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
parser.add_argument('--model',          type=str,   default='ResNetSE34L',     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default='ASP',  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=256,    help='Embedding size in the last FC layer');

## Data_preparing 's params
parser.add_argument('--preprocess',           dest='preprocess', action='store_true', help='Use when preprocess data');
parser.add_argument('--gpu',           dest='gpu', action='store_true', help='Use GPU');
parser.add_argument('--append',           dest='append', action='store_true', help='Append feats.npy');
parser.add_argument('--dataset_path',     type=str,   default='dataset/train-set', help='Absolute path to the dataset');
parser.add_argument('--feats_path',     type=str,   default='feats.npy', help='Path for feats file');

args = parser.parse_args();


def main_worker(args):
    dataset_path = args.dataset_path
    feats_path = args.feats_path

    if args.preprocess == True:
        folder_list = os.listdir(dataset_path)
        for folder in folder_list:
            file_list = os.listdir(dataset_path + '/' + folder)
            for file in file_list:
                file_path = dataset_path + '/' + folder + '/' + file
                outfile = dataset_path + '/' + folder + '/' + file.replace(file.split('.')[-1], 'wav')
                out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(file_path, outfile), shell=True)
                if out != 0:
                    raise ValueError('Conversion failed %s.'%file_path)
                os.remove(file_path)

        quit()

    ## Load models
    if args.gpu == True:
        s = SpeakerNet(**vars(args));
        s = WrappedModel(s).cuda(0)
    else:
        s = SpeakerNetCPU(**vars(args));
        s = WrappedModel(s).cpu()

    ## Load model weights
    try:
        loadParameters(args.model_path, s, args.gpu);
    except:
        raise Exception('Model path is wrong!')
    print('Model %s loaded from previous state!'%args.model_path);

    files_path = []
    file_list = os.listdir(dataset_path)
    for file in file_list:
        files_path.append(file)

    files_path.sort()

    feats = create_feature_vectors(s, dataset_path, files_path, args.eval_frames)
    
    if args.append == True:
        temp = np.load(args.feats_path, allow_pickle=True)[()]
        for key, value in temp.items():
            feats[key] = value

    np.save(feats_path, feats)


if __name__ == '__main__':
    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)

    main_worker(args)