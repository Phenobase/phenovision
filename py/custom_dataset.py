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


_ERROR_RETRY = 1000


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

class PhenoDatasetInf(data.Dataset):
    def __init__(
            self,
            img,
            target,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
            inference_mode=False,
    ):
        
        self.target = target
        self.img = img
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.skipped = []
        
        # Inference mode configuration
        self.inference_mode = inference_mode
        self.valid_indices = []
        self.invalid_indices = []
        
        # Use an extreme value that won't be confused with actual logits
        self.SENTINEL_VALUE = -999999999
            
    def __getitem__(self, index):
        img_path = self.img[index]
        
        try:
            img = img_path.read() if self.load_bytes else Image.open(img_path)
        except Exception as e:
            if self.inference_mode:
                # During inference, create a dummy image
                _logger.warning(f'Invalid image at index {index}, file {img_path}. {str(e)}')
                self.invalid_indices.append(index)
                
                # Create black dummy image
                dummy_img = Image.new(self.input_img_mode, (224, 224), color=(0, 0, 0))
                if self.transform is not None:
                    dummy_img = self.transform(dummy_img)
                
                # Create sentinel target with same shape as normal target
                # Get sentinel target shape from a normal target
                sample_target = self.target[0, ]  # Shape should be [d]
                # Create sentinel target filled with extreme value
                sentinel_target = np.full_like(sample_target, self.SENTINEL_VALUE)
                
                return dummy_img, sentinel_target
            else:
                # Original behavior - skip invalid images
                _logger.warning(f'Skipped sample (index {index}, file {img_path}). {str(e)}')
                self._consecutive_errors += 1
                self.skipped.append(img_path)
                if self._consecutive_errors < _ERROR_RETRY:
                    return self.__getitem__((index + 1) % len(self.img))
                else:
                    raise e
                    
        self._consecutive_errors = 0
        
        # Process valid image
        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
            
        if self.transform is not None:
            img = self.transform(img)
            
        target = self.target[index, ]
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
            
        # Track valid indices during inference
        if self.inference_mode:
            self.valid_indices.append(index)
            
        return img, target
    
    def __len__(self):
        return len(self.img)
    
    def filename(self, index, basename=False, absolute=False):
        return self.img[index]
    
    def filenames(self, basename=False, absolute=False):
        return self.img
    
    def is_sentinel_target(self, target):
        """Check if a target is a sentinel value"""
        # Check if all values are the sentinel value
        return np.all(target == self.SENTINEL_VALUE)
    
    def get_valid_indices(self):
        """Return indices of valid images during inference"""
        if not self.inference_mode:
            _logger.warning('get_valid_indices() called but inference_mode is False')
        return self.valid_indices.copy()
    
    def get_invalid_indices(self):
        """Return indices of invalid images during inference"""
        if not self.inference_mode:
            _logger.warning('get_invalid_indices() called but inference_mode is False')
        return self.invalid_indices.copy()

def prepare_inference_dataset(all_images, all_targets, **kwargs):
    """
    Pre-filter images and create a dataset with only valid images.
    
    Args:
        all_images: List of all image paths
        all_targets: Array or list of all targets
        **kwargs: Any additional keyword arguments to pass to PhenoDataset
        
    Returns:
        dataset: PhenoDataset containing only valid images
        valid_indices: List mapping dataset indices back to original indices
    """
    valid_indices = []
    valid_images = []
    valid_targets = []
    
    # Pre-scan to find only valid images
    for i, img_path in enumerate(all_images):
        try:
            # Verify image can be opened
            Image.open(img_path)
            valid_images.append(img_path)
            valid_targets.append(all_targets[i])
            valid_indices.append(i)
        except Exception as e:
            print(f"Skipping invalid image at index {i}: {img_path} - {str(e)}")
    
    # Create dataset with only valid images, passing all keyword arguments
    dataset = PhenoDataset(
        img=valid_images,
        target=valid_targets,
        **kwargs  # This unpacks all additional arguments
    )
    
    print(f"Created dataset with {len(valid_images)} valid images out of {len(all_images)} total")
    
    # Return both the dataset and the mapping to original indices
    return dataset, valid_indices
