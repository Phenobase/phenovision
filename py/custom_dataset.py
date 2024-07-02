import io
import logging
import os
import rocksdb
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class PhenoDataset(data.Dataset):

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
        self.skipped = []

    def __getitem__(self, index):
      
        img = self.img[index]
        
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.img[index]}). {str(e)}')
            self._consecutive_errors += 1
            self.skipped.append(self.img[index])
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.img))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(img)
            
        target = self.target[index, ]

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img)

    def filename(self, index, basename=False, absolute=False):
        return self.img[index]

    def filenames(self, basename=False, absolute=False):
        return self.img
      
class PhenoDatasetRocksDB(data.Dataset):

    def __init__(
            self,
            img,
            target,
            db,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None
    ):
        
        #rdb_options = rocksdb.Options(create_if_missing = True)
        
        self.target = target
        self.img = img
        self.db = db
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.skipped = []

    def __getitem__(self, index):
      
        img = self.img[index]
        
        img_data = self.db.get(img)
        #key_str = self.db.img.decode('utf-8')
        raw_data = io.BytesIO(img_data)
        
        try:
            img = raw_data.read() if self.load_bytes else Image.open(raw_data)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.img[index]}). {str(e)}')
            self._consecutive_errors += 1
            self.skipped.append(self.img[index])
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.img))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(img)
            
        target = self.target[index, ]

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img)

    def filename(self, index, basename=False, absolute=False):
        return self.img[index]

    def filenames(self, basename=False, absolute=False):
        return self.img
