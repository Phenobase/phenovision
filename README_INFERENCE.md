# PhenoVision Inference Module

## Quick Start Guide

This document describes the optimized inference module (`inference.py`) for production use of PhenoVision.

### Installation

```bash
pip install torch transformers pillow numpy
```

### Basic Usage

```python
from inference import PhenoVisionClassifier

# Initialize
classifier = PhenoVisionClassifier()

# Predict
results = classifier.predict('plant_image.jpg')
print(f"Flowering: {results['flowering']:.1%}")
print(f"Fruiting: {results['fruiting']:.1%}")
```

### Command Line

```bash
python inference.py path/to/image.jpg
```

### Batch Processing

```python
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = classifier.batch_predict(images, batch_size=32)
```

## API Reference

### `PhenoVisionClassifier`

Main classifier class for plant phenology detection.

#### Methods

- `predict(image)` - Single image prediction
- `batch_predict(images, batch_size)` - Batch prediction
- `classify(image, threshold)` - Classification with threshold

#### Returns

Dictionary with:
- `flowering`: Probability (0-1)
- `fruiting`: Probability (0-1)

## Model Outputs

The model provides independent probabilities for:
- **Flowering**: Plant is in flowering stage
- **Fruiting**: Plant is bearing fruit

Both can be high simultaneously (e.g., plants that flower and fruit together).

## Performance

- GPU recommended for optimal performance
- Batch processing for multiple images
- ~50-100ms per image on GPU
- ~200-500ms per image on CPU

## Integration Example

```python
# Production pipeline integration
from inference import PhenoVisionClassifier
import pandas as pd

classifier = PhenoVisionClassifier()

# Process dataset
df = pd.read_csv('plant_observations.csv')
results = []

for idx, row in df.iterrows():
    pred = classifier.predict(row['image_path'])
    results.append({
        'id': row['id'],
        'flowering_prob': pred['flowering'],
        'fruiting_prob': pred['fruiting']
    })

pd.DataFrame(results).to_csv('phenology_predictions.csv')
```

## Citation

See main README for citation information.