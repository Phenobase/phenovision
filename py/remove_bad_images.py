import io
import logging
import os
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class PhenoDatasetDeleter(data.Dataset):

    def __init__(
            self,
            img,
            target,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        
        self.target = target
        self.img = img
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
      
        img = self.img[index]
        bad = 0
        
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.img[index]}). {str(e)}')
            bad = 1
            
        if self.input_img_mode and not self.load_bytes:
            try:
                img = img.convert(self.input_img_mode)
            except Exception as e:
                _logger.warning(f'Skipped sample (index {index}, file {self.img[index]}). {str(e)}')
                bad = 1
        
        if self.transform is not None:
            try:
                img = self.transform(img)
            except Exception as e:
                _logger.warning(f'Skipped sample (index {index}, file {self.img[index]}). {str(e)}')
                bad = 1
            
        
        return self.img[index], bad

    def __len__(self):
        return len(self.img)

    def filename(self, index, basename=False, absolute=False):
        return self.img[index]

    def filenames(self, basename=False, absolute=False):
        return self.img


