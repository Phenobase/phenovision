---
library_name: transformers
pipeline_tag: image-classification
license: mit
tags:
  - vision
  - image-classification
  - biology
  - ecology
  - phenology
  - plants
  - vit
  - plant-phenology
  - iNaturalist
datasets:
  - iNaturalist
metrics:
  - accuracy
  - f1
language:
  - en
model-index:
  - name: PhenoVision
    results:
      - task:
          type: image-classification
          name: Plant Reproductive Structure Detection
        metrics:
          - type: accuracy
            value: 98.02
            name: Flower Accuracy (buffer-filtered)
          - type: accuracy
            value: 97.01
            name: Fruit Accuracy (buffer-filtered)
---

# PhenoVision: Automated Plant Reproductive Phenology from Field Images

PhenoVision is a Vision Transformer (ViT-Large) model fine-tuned to detect **flowers** and **fruits** in plant photographs. It was trained on 1.5 million human-annotated iNaturalist images and has been used to generate over 30 million new phenology records across 119,000+ plant species, vastly expanding global coverage of plant reproductive phenology data.

| | Flower | Fruit |
|---|---|---|
| **Accuracy** | 98.0% | 97.0% |
| **Sensitivity** | 98.5% | 84.2% |
| **Specificity** | 97.2% | 99.4% |
| **Expert validation** | 98.6% | 90.4% |

## Model Details

- **Model type:** Multi-label image classification (sigmoid outputs)
- **Architecture:** Vision Transformer Large (ViT-L/16), ~304M parameters
- **Input:** 224 x 224 RGB images
- **Output:** 2 logits (flower, fruit) — apply sigmoid for probabilities
- **Pretraining:** PlantCLEF 2022 checkpoint ("virtual taxonomist" — trained on 2.9M plant species images)
- **Current version:** v1.1.0
- **Model DOI:** [10.57967/hf/7952](https://doi.org/10.57967/hf/7952)
- **Developer:** [Phenobase](https://phenobase.org/)
- **Repository:** [github.com/Phenobase/phenovision](https://github.com/Phenobase/phenovision)
- **License:** MIT

### Key Innovation: Virtual Taxonomist Pretraining

Instead of standard ImageNet pretraining, PhenoVision uses a ViT-Large checkpoint pretrained on the PlantCLEF 2022 dataset (2.9 million plant images for species classification). Since species classification relies heavily on recognizing reproductive structures (flowers, fruits), this domain-specific pretraining provides a strong initialization for phenology detection. Compared to ImageNet pretraining, PlantCLEF pretraining achieved:

- Higher accuracy: TSS = 0.864 vs. 0.835
- Faster convergence: Best epoch at 4 vs. 11

## Intended Uses

**Primary use:** Detecting the presence of flowers and/or fruits in field photographs of plants.

**Suitable for:**
- Automated phenology annotation of iNaturalist and other community science images
- Large-scale phenology monitoring and climate change research
- Generating presence-only reproductive phenology datasets
- Integration with phenology databases (e.g., [Phenobase](https://phenobase.org/), USA-NPN)

**Out of scope:**
- Counting individual flowers or fruits
- Distinguishing flower developmental stages (buds vs. open vs. senescent)
- Detecting leaf phenology (use [PhenoVisionL](https://huggingface.co/phenobase/phenovisionL) instead)
- Identifying plant species (this is a phenology model, not a taxonomic classifier)

## How to Use

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load model and processor
processor = ViTImageProcessor.from_pretrained("phenobase/phenovision")
model = ViTForImageClassification.from_pretrained("phenobase/phenovision")
model.eval()

# Run inference
image = Image.open("plant_photo.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]

flower_prob = probs[0].item()
fruit_prob = probs[1].item()

print(f"Flower: {flower_prob:.3f}")
print(f"Fruit:  {fruit_prob:.3f}")
```

### Applying Thresholds

Raw probabilities should be converted to detection calls using the optimized thresholds and uncertainty buffers provided as companion files. Predictions falling within the buffer zone are classified as "Equivocal" and should be excluded for research-quality outputs.

| Class | Threshold | Buffer Lower | Buffer Upper | Equivocal Range |
|-------|-----------|--------------|--------------|-----------------|
| Flower | 0.48 | 0.325 | 0.385 | 0.155 - 0.865 |
| Fruit | 0.60 | 0.405 | 0.305 | 0.195 - 0.905 |

- Probability **above** (threshold + buffer_upper) → **Detected** (high certainty)
- Probability **below** (threshold - buffer_lower) → **Not Detected** (high certainty)
- Probability **within** buffer zone → **Equivocal** (exclude from analysis)

## Training Data

- **Source:** [iNaturalist](https://www.inaturalist.org/) open data (research-grade observations)
- **Size:** 1,535,930 images from 119,340 species across 10,406 genera and 408 plant families
- **Splits:** 60% train (921,720) / 20% validation (307,291) / 20% test (306,919), stratified by genus
- **Annotations:** Human phenology annotations from iNaturalist platform (reproductiveCondition field)
- **Licensing:** Images under CC-0, CC-BY, or CC-BY-NC licenses
- **Note:** Approximately 1-5% of training annotations are marked "unknown" due to annotation difficulty

## Training Procedure

- **Optimizer:** AdamW
- **Learning rate:** 5e-4 (base), with layer-wise decay factor 0.65
- **Batch size:** 384
- **Weight decay:** 0.05
- **Data augmentation:** RandAugment
- **Epochs:** 10 (best model selected at epoch 7 by average Data Quality Index)
- **Hardware:** NVIDIA A100 GPU
- **Loss:** Binary cross-entropy (multi-label)
- **v1.1.0 training:** Fine-tuned from v1.0.0 checkpoint on updated data snapshot (2025-10-27)

## Evaluation Results

### Test Set Performance (v1.1.0)

| Class | Filter | N | Accuracy | Sensitivity | Specificity | PPV | NPV | J-Index | F1 | DQI |
|-------|--------|---|----------|-------------|-------------|-----|-----|---------|-----|-----|
| Flower | All data | 713,698 | 95.77% | 96.93% | 93.72% | 96.45% | 94.54% | 0.907 | 96.69% | 0.934 |
| Flower | Buffer filtered | 663,738 | 98.02% | 98.47% | 97.19% | 98.48% | 97.19% | 0.957 | 98.48% | 0.970 |
| Fruit | All data | 713,698 | 94.33% | 77.33% | 98.04% | 89.64% | 95.18% | 0.754 | 83.03% | 0.670 |
| Fruit | Buffer filtered | 651,791 | 97.01% | 84.16% | 99.37% | 96.11% | 97.16% | 0.835 | 89.74% | 0.803 |

### Expert Validation

Independent expert review of model predictions:
- **Flower presence:** 98.6% agreement
- **Fruit presence:** 90.4% agreement

### Taxonomic Coverage

- **Species:** 119,340 from 10,406 genera and 408 families
- **Genera with 10+ records:** 7,409 (flowers), 5,240 (fruits)
- **Median records per genus:** 184 (flowers), 85 (fruits)
- **New geographic grid cells:** 3,798 (flowers), 4,147 (fruits) with no prior phenology data

## Companion Files

The following files are uploaded alongside the model weights:

| File | Description |
|------|-------------|
| `final_buffer_params.csv` | Decision thresholds and uncertainty buffer parameters per class. Used to convert probabilities to Detected/Not Detected/Equivocal calls. |
| `family_stats.csv` | Per-family (706 families) accuracy statistics. Useful for assessing model reliability for specific taxonomic groups. |

## Limitations and Biases

### Design Limitations
- **Presence-only:** The model reports detections but NOT absences. A non-detection does not mean the plant lacks flowers/fruits — it may simply not be visible in the image.
- **Partial plant coverage:** Images typically show only part of a plant. Reproductive structures may exist on non-photographed parts.
- **Buffer zone data loss:** Applying uncertainty thresholds removes ~7-9% of predictions as equivocal, trading completeness for accuracy.

### Known Failure Modes
- Inconspicuous reproductive structures (grasses, sedges) are harder to detect
- Flower buds may be confused with open flowers
- Background plants with flowers/fruits can cause false positives for the focal plant
- Some families show lower accuracy (e.g., Haloragaceae ~79%)

### Data Biases
- Reflects iNaturalist's geographic biases: overrepresentation of urban areas, developed countries, and coastal regions
- Taxonomic bias toward common, conspicuous species
- Limited coverage in biodiversity-rich tropical regions

### Annotation Quality
- Training labels come from community science annotations with inherent variability
- Some iNaturalist annotations are incomplete (e.g., flower present but only fruit annotated)
- Family-level accuracy statistics (in `family_stats.csv`) should be consulted when interpreting results for specific taxonomic groups

## Citation

If you use PhenoVision in your research, please cite:

```bibtex
@article{dinnage2025phenovision,
  title={PhenoVision: A framework for automating and delivering research-ready plant phenology data from field images},
  author={Dinnage, Russell and Grady, Erin and Neal, Nevyn and Deck, Jonn and Denny, Ellen and Walls, Ramona and Seltzer, Carrie and Guralnick, Robert and Li, Daijiang},
  journal={Methods in Ecology and Evolution},
  volume={16},
  pages={1763--1780},
  year={2025},
  doi={10.1111/2041-210X.14346}
}
```

## Acknowledgments

- **Funding:** National Science Foundation (NSF)
- **Data:** [iNaturalist](https://www.inaturalist.org/) community and platform
- **Infrastructure:** [Phenobase](https://phenobase.org/) — a global plant phenology database
- **Integration:** Plant Phenology Ontology (PPO), USA National Phenology Network (USA-NPN)
