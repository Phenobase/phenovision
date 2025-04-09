import math
import shutil
import sys
from typing import Iterable, Optional
import os
import os.path as osp
# import pandas as pd
import numpy as np

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


@torch.no_grad()
def evaluate(data_loader, model, device):
    
    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []

    for i, batch in enumerate(data_loader):
        images = batch[0]
        
        target = batch[-1]
        
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
        
        outputs.append(output.cpu())
        targets.append(target.cpu())

    return outputs, targets

@torch.no_grad()
def infer(data_loader, model, device):
    
    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []
    
    dat_len = len(data_loader)

    for i, batch in enumerate(data_loader):
        images = batch[0]
        
        target = batch[-1]
        
        #images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
        
        outputs.append(output.cpu())
        targets.append(target.cpu())
        
        print(f'* Done iteration {i} of {dat_len}')

    return outputs, targets

@torch.no_grad()
def infer_hfhub(data_loader, model, device):
    
    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []
    
    dat_len = len(data_loader)

    for i, batch in enumerate(data_loader):
        images = batch[0]
        
        target = batch[-1]
        
        #images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
        
        outputs.append(output[0].cpu())
        targets.append(target.cpu())
        
        print(f'* Done iteration {i} of {dat_len}')

    return outputs, targets


@torch.no_grad()
def get_codes(data_loader, model, device, intermediates=False):
    
    # switch to evaluation mode
    model.eval()
    outputs = []
    intermediate_outputs = [] if intermediates else None
    
    dat_len = len(data_loader)
    for i, batch in enumerate(data_loader):
        images = batch[0]
        
        target = batch[-1]
        
        #images = images.to(device, non_blocking=True)
        #target = target.to(device, non_blocking=True)
        
        # compute output
        with torch.cuda.amp.autocast():
            # Extract standard features
            output = model.forward_features(images)
            
            # If intermediates requested, also extract those
            if intermediates:
                intermediate_output = model.forward_intermediates(images, return_prefix_tokens=True)
                intermediate_outputs.append(intermediate_output.cpu())
        
        outputs.append(output.cpu())
        #targets.append(target)
        
        print(f'* Done iteration {i} of {dat_len}')
    
    # Return appropriate structure based on intermediates flag
    if intermediates:
        return {
            'features': outputs,
            'intermediates': intermediate_outputs
        }
    else:
        return outputs
