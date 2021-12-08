#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import glob
import zipfile
import datetime
import os
import random
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import loadWAV
from DatasetLoader import get_data_loader
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_data(file_path, eval_frames):
    return torch.FloatTensor(loadWAV(file_path, eval_frames, evalmode=True))


def loadParameters(path, model, gpu):
        self_state = model.module.state_dict();
        if gpu == True:
            loaded_state = torch.load(path, map_location="cuda:%d"%0);
        else:
            loaded_state = torch.load(path, map_location="cpu");
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);


def create_feature_vectors(model, dataset_path, files_path, eval_frames):
    feats = {}
    
    for file_path in tqdm(files_path):
        path = os.path.join(dataset_path, file_path)
        data = create_data(path, eval_frames)
        feature_vector = model(data).detach().cpu()
        normalized_vector = F.normalize(feature_vector, p=2, dim=1)
        feats[file_path] = normalized_vector

    return feats


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


def score_normalization(ref, com, cohorts, top=-1):
    """
    Adaptive symmetric score normalization using cohorts from eval data
    """
    def ZT_norm(ref, com, top=-1):
        """
        Perform Z-norm or T-norm depending on input order
        """
        S = np.mean(np.inner(cohorts, ref), axis=1)
        S = np.sort(S, axis=0)[::-1][:top]
        mean_S = np.mean(S)
        std_S = np.std(S)
        score = np.inner(ref, com)
        score = np.mean(score)
        return (score - mean_S) / std_S

    def S_norm(ref, com, top=-1):
        """
        Perform S-norm
        """
        return (ZT_norm(ref, com, top=top) + ZT_norm(com, ref, top=top)) / 2

    ref = ref.cpu().numpy()
    com = com.cpu().numpy()
    return S_norm(ref, com, top=top)