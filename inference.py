#!/usr/bin/env python3
"""
PhenoVision Inference Module
============================

A clean, optimized implementation for plant phenology classification using PhenoVision.
Predicts flowering and fruiting probabilities from plant images.

Model: https://huggingface.co/phenobase/phenovision
Paper: https://github.com/Phenobase/phenovision

Installation:
    pip install torch transformers pillow numpy

Usage:
    from phenovision_inference import PhenoVisionClassifier
    
    classifier = PhenoVisionClassifier()
    results = classifier.predict('path/to/image.jpg')
    print(f"Flowering: {results['flowering']:.2%}, Fruiting: {results['fruiting']:.2%}")

Author: DeepEarth Contributors
License: MIT
"""

import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhenoVisionClassifier:
    """
    PhenoVision classifier for plant phenology stages.
    
    This model classifies plant images into flowering and fruiting stages.
    It uses a Vision Transformer (ViT) architecture fine-tuned on plant phenology data.
    
    Attributes:
        model_path: Path to the model weights and configuration
        device: Torch device for computation (cuda/cpu)
        model: The loaded ViT model
        processor: Image preprocessor for ViT
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the PhenoVision classifier.
        
        Args:
            model_path: Path to model directory. If None, uses HuggingFace Hub.
            device: Device to run on ('cuda', 'cpu', or None for auto-detect).
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing PhenoVision on {self.device}")
        
        # Load model
        if model_path:
            # Load from local path
            self.model_path = Path(model_path)
            logger.info(f"Loading model from {self.model_path}")
            self.model = ViTForImageClassification.from_pretrained(
                self.model_path,
                local_files_only=True
            )
        else:
            # Load from HuggingFace Hub
            logger.info("Loading model from HuggingFace Hub: phenobase/phenovision")
            self.model = ViTForImageClassification.from_pretrained(
                "phenobase/phenovision"
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize image processor with standard ViT settings
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224",
            size={"height": 224, "width": 224}
        )
        
        # Class labels - determined through empirical testing
        # Index 0: Fruiting probability
        # Index 1: Flowering probability
        self.class_names = ['fruiting', 'flowering']
        
        logger.info("PhenoVision classifier ready")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image tensor ready for model input
            
        Raises:
            ValueError: If image format is not supported
            FileNotFoundError: If image file doesn't exist
        """
        # Handle different input types
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a file path or PIL Image")
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image using ViT processor
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)
    
    def predict(self, image: Union[str, Path, Image.Image]) -> Dict[str, float]:
        """
        Predict flowering and fruiting probabilities for a single image.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Dictionary containing:
                - 'flowering': Probability of flowering (0-1)
                - 'fruiting': Probability of fruiting (0-1)
                
        Note:
            Both probabilities are independent and can sum to more than 1,
            as plants can exhibit both flowering and fruiting simultaneously.
        """
        # Preprocess image
        pixel_values = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(pixel_values)
            logits = outputs.logits
            
            # Apply sigmoid for multi-label classification
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Return as dictionary with corrected indices
        results = {
            'flowering': float(probabilities[1]),  # Index 1 is flowering
            'fruiting': float(probabilities[0])    # Index 0 is fruiting
        }
        
        return results
    
    def batch_predict(self, 
                     images: List[Union[str, Path, Image.Image]], 
                     batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Predict phenology for multiple images in batches.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Number of images to process simultaneously
            
        Returns:
            List of prediction dictionaries for each image
            
        Note:
            Failed images will have -1.0 for probabilities and an 'error' key.
        """
        results = []
        total_images = len(images)
        
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch = images[batch_start:batch_end]
            batch_tensors = []
            batch_indices = []
            
            # Preprocess batch
            for idx, img in enumerate(batch):
                try:
                    tensor = self.preprocess_image(img)
                    batch_tensors.append(tensor)
                    batch_indices.append(batch_start + idx)
                except Exception as e:
                    logger.error(f"Failed to process image {batch_start + idx}: {e}")
                    results.append({
                        'flowering': -1.0,
                        'fruiting': -1.0,
                        'error': str(e)
                    })
            
            # Process valid images
            if batch_tensors:
                # Stack tensors
                pixel_values = torch.cat(batch_tensors, dim=0)
                
                # Run batch inference
                with torch.no_grad():
                    outputs = self.model(pixel_values)
                    logits = outputs.logits
                    probabilities = torch.sigmoid(logits).cpu().numpy()
                
                # Add results in correct order
                for prob in probabilities:
                    results.append({
                        'flowering': float(prob[1]),  # Index 1 is flowering
                        'fruiting': float(prob[0])    # Index 0 is fruiting
                    })
        
        return results
    
    def classify(self, 
                image: Union[str, Path, Image.Image],
                threshold: float = 0.5) -> Dict[str, Union[List[str], Dict[str, float]]]:
        """
        Classify an image into phenological stages using threshold.
        
        Args:
            image: Path to image file or PIL Image
            threshold: Probability threshold for classification (default: 0.5)
            
        Returns:
            Dictionary containing:
                - 'stages': List of detected stages (e.g., ['flowering', 'fruiting'])
                - 'probabilities': Raw probability values
        """
        probs = self.predict(image)
        
        stages = []
        if probs['flowering'] >= threshold:
            stages.append('flowering')
        if probs['fruiting'] >= threshold:
            stages.append('fruiting')
        
        if not stages:
            stages = ['vegetative']  # Neither flowering nor fruiting
        
        return {
            'stages': stages,
            'probabilities': probs
        }


def main():
    """
    Example usage and testing of the PhenoVision classifier.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='PhenoVision Plant Phenology Classifier')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to local model directory (optional)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                       help='Device to use (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = PhenoVisionClassifier(
        model_path=args.model_path,
        device=args.device
    )
    
    # Classify image
    try:
        result = classifier.classify(args.image, threshold=args.threshold)
        
        print("\n" + "="*50)
        print("PHENOVISION CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Image: {args.image}")
        print(f"Detected stages: {', '.join(result['stages']).upper()}")
        print("\nProbabilities:")
        print(f"  Flowering: {result['probabilities']['flowering']:.1%}")
        print(f"  Fruiting:  {result['probabilities']['fruiting']:.1%}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())