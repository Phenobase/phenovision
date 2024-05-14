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
