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
  - leaf-phenology
  - iNaturalist
datasets:
  - iNaturalist
base_model: phenobase/phenovision
metrics:
  - accuracy
language:
  - en
model-index:
  - name: PhenoVisionL
    results:
      - task:
          type: image-classification
          name: Plant Leaf Phenology Detection
        metrics:
          - type: accuracy
            value: 98.6
            name: Green Leaves Accuracy (expert validation)
          - type: accuracy
            value: 99.4
            name: Colored Leaves Accuracy (expert validation)
          - type: accuracy
            value: 87.0
            name: Breaking Buds Accuracy (expert validation)
---

# PhenoVisionL: Automated Leaf Phenology Detection from Field Images

PhenoVisionL is a Vision Transformer (ViT-Large) model fine-tuned to detect **leaf phenological states** in plant photographs: green leaves, colored (senescent) leaves, and breaking leaf buds. It was trained on 165,988 iNaturalist records of deciduous woody plants using a two-stage semi-supervised approach, and has generated 5.6 million leaf phenology observations across 6,500+ species, filling major geographic gaps in global leaf phenology data.

| | Green Leaves | Colored Leaves | Breaking Buds |
|---|---|---|---|
| **Expert validation accuracy** | 98.6% | 99.4% | 87.0% |
| **False positive rate** | 1.2% | 0.6% | 9.4% |

## Model Details

- **Model type:** Multi-label image classification (sigmoid outputs)
- **Architecture:** Vision Transformer Large (ViT-L/16), ~304M parameters
- **Input:** 224 x 224 RGB images
- **Output:** 3 logits (green leaves, colored leaves, breaking buds) — apply sigmoid for probabilities
- **Pretraining:** Initialized from trained [PhenoVision](https://huggingface.co/phenobase/phenovision) reproductive model (transfer learning from flower/fruit detection)
- **Current version:** v1.0.0
- **Model DOI:** [10.57967/hf/5785](https://doi.org/10.57967/hf/5785)
- **Developer:** [Phenobase](https://phenobase.org/)
- **Repository:** [github.com/Phenobase/phenovision](https://github.com/Phenobase/phenovision)
- **License:** MIT

### Transfer Learning from Reproductive Model

PhenoVisionL is initialized from the trained PhenoVision reproductive structures model rather than from ImageNet or PlantCLEF directly. This leverages the reproductive model's learned representations of plant structure and morphology, providing a strong initialization for leaf phenology tasks. A new randomly initialized classification head replaces the original 2-class output with a 3-class output.

## Intended Uses

**Primary use:** Detecting leaf phenological states in field photographs of deciduous woody plants.

**Suitable for:**
- Automated annotation of leaf phenology in iNaturalist and community science images
- Climate change research on phenological shifts (spring leaf-out, autumn senescence)
- Large-scale monitoring of deciduous forest phenology
- Integration with phenology databases (e.g., [Phenobase](https://phenobase.org/), USA-NPN)

**Out of scope:**
- **Evergreen plants** — the model was trained on deciduous woody plants only
- **Herbaceous plants** — not included in training data
- Quantifying leaf area or canopy cover
- Detecting reproductive structures (use [PhenoVision](https://huggingface.co/phenobase/phenovision) instead)
- Species identification

## How to Use

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load model and processor
processor = ViTImageProcessor.from_pretrained("phenobase/phenovisionL")
model = ViTForImageClassification.from_pretrained("phenobase/phenovisionL")
model.eval()

# Run inference
image = Image.open("plant_photo.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]

green_prob = probs[0].item()
colored_prob = probs[1].item()
breaking_buds_prob = probs[2].item()

print(f"Green leaves:   {green_prob:.3f}")
print(f"Colored leaves: {colored_prob:.3f}")
print(f"Breaking buds:  {breaking_buds_prob:.3f}")
```

### Applying Thresholds

Raw probabilities should be converted to detection calls using the optimized thresholds and uncertainty buffers provided as companion files. Predictions falling within the buffer zone are classified as "Equivocal" and should be excluded for research-quality outputs.

- Probability **above** (threshold + buffer_upper) → **Detected** (high certainty)
- Probability **below** (threshold - buffer_lower) → **Not Detected** (high certainty)
- Probability **within** buffer zone → **Equivocal** (exclude from analysis)

See the companion file `epoch_1_threshold_buffers.csv` for the specific threshold and buffer values for each class.

## Training Data

- **Source:** [iNaturalist](https://www.inaturalist.org/) open data
- **Taxonomic scope:** 145 genera of deciduous woody plants, covering 6,501 species from 57 plant families
- **Size:** 165,988 records containing 326,128 images
- **Annotations:**
  - **Green leaves & colored leaves:** iNaturalist user annotations (dynamicProperties field)
  - **Breaking leaf buds:** Expert annotations only — iNaturalist user annotations for this class were found to be unreliable and were excluded from training
- **Licensing:** Images under CC-0, CC-BY, or CC-BY-NC licenses

## Training Procedure

PhenoVisionL uses a **two-stage semi-supervised training approach**:

### Stage 1: Single-Image Training
- **Data:** 88,184 single-image observations with verified annotations
- **Epochs:** 4
- **Class balancing:** Upsampling of minority classes (colored leaves, breaking buds) to address imbalance

### Stage 2: Multi-Image Semi-Supervised Fine-Tuning
- **Data:** 77,804 multi-image observations
- **Epochs:** 4
- **Confidence filtering:** Only predictions with >0.95 probability that matched the original iNaturalist annotation were used; lower-confidence predictions were excluded

### Hyperparameters (both stages)
- **Optimizer:** AdamW
- **Learning rate:** 5e-4 (base), with layer-wise decay factor 0.65
- **Batch size:** 384
- **Weight decay:** 0.05
- **Data augmentation:** RandAugment
- **Hardware:** NVIDIA A100 GPU
- **Loss:** Binary cross-entropy (multi-label)

## Evaluation Results

### Expert Validation

Independent expert review of high-confidence (unequivocal) model predictions:

| Phenophase | Accuracy | False Positive Rate |
|------------|----------|-------------------|
| Green leaves | 98.6% | 1.2% |
| Colored leaves | 99.4% | 0.6% |
| Breaking leaf buds | 87.0% | 9.4% |

Breaking buds have lower accuracy due to inherent task difficulty — morphological similarity between breaking leaf buds and flower buds, and limited expert-only training data for this class.

### Coverage

- **Observations generated:** 5.6 million from 26+ million iNaturalist images
- **Species covered:** 6,501 across 145 genera and 57 families
- **Geographic reach:** 8,515 grid cells (100 km x 100 km) globally
- **New coverage:** 4,342 grid cells received green leaf phenology data where none existed before
- **Regions with new data:** Temperate Eurasia, boreal and arctic regions previously lacking coverage

## Companion Files

The following files are uploaded alongside the model weights:

| File | Description |
|------|-------------|
| `epoch_1_threshold_buffers.csv` | Decision thresholds and uncertainty buffer parameters per class. Used to convert probabilities to Detected/Not Detected/Equivocal calls. **Note:** Despite the `.csv` extension, this file is in RDS format and should be read with `readRDS()` in R. |
| `family_stats.csv` | Per-family (57 families) accuracy statistics for each leaf class. |

## Limitations and Biases

### Taxonomic Restrictions
- **Deciduous woody plants only:** The model was trained exclusively on 145 genera of deciduous woody plants. It is **not suitable** for herbaceous plants, evergreen species, or non-vascular plants.
- Performance varies by family — consult `family_stats.csv` for family-level accuracy.

### Design Limitations
- **Presence-only:** The model reports detections but NOT absences. A non-detection does not mean leaves are absent.
- **Breaking buds are harder:** 87% accuracy and 9.4% false positive rate, driven by morphological similarity to flower buds and limited training data (expert annotations only).
- **Partial plant coverage:** Images typically show only part of a plant.

### Known Failure Modes
- Breaking leaf buds confused with flower buds in some taxa
- Taxa-specific leaf morphology can affect detection (unusual leaf forms)
- Background vegetation may contribute to false detections
- Very early or late phenological stages may be ambiguous

### Data Biases
- Reflects iNaturalist's geographic biases: overrepresentation of urban areas, developed countries, and coastal regions
- Taxonomic bias toward common, conspicuous deciduous species
- Breaking bud annotations limited to a single expert annotator

### Annotation Quality
- User-contributed iNaturalist annotations for green and colored leaves have variable quality
- Breaking bud annotations are expert-only due to reliability concerns with user annotations
- Family-level accuracy statistics should be consulted when interpreting results

## Citation

If you use PhenoVisionL in your research, please cite:

```bibtex
@article{grady2025phenovisionL,
  title={PhenoVision: A framework for automating and delivering research-ready plant phenology data from field images},
  author={Grady, Erin L. and Denny, Ellen G. and Seltzer, Carrie E. and Deck, John and Li, Daijiang and Dinnage, Russell and Guralnick, Robert P.},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.09.26.678778}
}
```

Also cite the original PhenoVision framework paper:

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
- **Data archive:** [Zenodo](https://doi.org/10.5281/zenodo.17107251)
- **Integration:** Plant Phenology Ontology (PPO), USA National Phenology Network (USA-NPN)
