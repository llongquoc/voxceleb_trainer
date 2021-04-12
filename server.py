#!/usr/bin/python
#-*- coding: utf-8 -*-


from SpeakerNet import *
from utils import *
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
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = 'Prepare Data');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=400,    help='Input length to the network for testing; 0 uses the whole files');

## Training details
parser.add_argument('--trainfunc',      type=str,   default='softmaxproto',     help='Loss function');

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
parser.add_argument('--model',          type=str,   default='ResNetSE34V2',     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default='ASP',  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

##Server 's params
parser.add_argument('--threshold',           type=float,   default=-1.0831763744354248,    help='Threshold');
parser.add_argument('--feats_path',     type=str,   default='feats.npy', help='Path for feats file');

args = parser.parse_args();


## Load models
s = SpeakerNetCPU(**vars(args));
s = WrappedModel(s).cpu()

## Load model weights
try:
    loadParameters(args.model_path, s);
except:
    raise Exception('Model path is wrong!')
print('Model %s loaded from previous state!'%args.model_path);

feats = np.load(args.feats_path, allow_pickle=True)[()]


def main_worker(file_path):
    data = create_data(file_path, args.eval_frames)
    feature_vector = s(data).detach().cpu()
    normalized_vector = F.normalize(feature_vector, p=2, dim=1)

    max_score = args.threshold
    speaker = ''
    for key, value in feats.items():
        dist = F.pairwise_distance(normalized_vector.unsqueeze(-1), value.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
        score = -1 * np.mean(dist);

        if score >= max_score:
            max_score = score
            speaker = key.split('/')[-2]
    return speaker


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    file_name_1 = str(random.randint(0, 100000)) + secure_filename(audio_file.filename).split('.')[-1]
    audio_file.save(file_name_1)

    file_name_2 = str(random.randint(0, 100000)) + '.wav'
    out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(file_name_1, file_name_2), shell=True)
    if out != 0:
        return 'Invalid format!'

    speaker = main_worker(file_name_2)
    os.remove(file_name_1)
    os.remove(file_name_2)

    result = {'speaker': speaker}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080', debug=False)