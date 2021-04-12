#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
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
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import get_data_loader
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np


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


def create_feature_vectors(trainer, dataset_path, files_path, nDataLoaderThread, eval_frames):
    feats = {}
    tstart = time.time()

    trainer.__model__.eval();

    test_dataset = test_dataset_loader(files_path, dataset_path, num_eval=10, eval_frames=eval_frames)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
    
    for idx, data in enumerate(test_loader):
            inp1                = data[0][0].cpu()
            ref_feat            = trainer.__model__(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % 100 == 0:
                sys.stdout.write('\rReading %d of %d: %.2f Hz, embedding size %d'%(idx,len(files_path),idx/telapsed,ref_feat.size()[1]));
    
    return feats


def loadParameters(path, model):

        self_state = model.module.state_dict();
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