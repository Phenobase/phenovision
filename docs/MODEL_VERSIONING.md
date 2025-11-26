# PhenoVision Model Versioning

**Last Updated:** 2025-11-17

## Overview

PhenoVision uses semantic versioning adapted for machine learning models to track the evolution of training data, model architecture, and hyperparameters.

## Versioning Scheme

**Format:** `vMAJOR.MINOR.PATCH`

```
v1.1.0
│ │ │
│ │ └─ PATCH: Data-only updates
│ └─── MINOR: Model logic/architecture changes
└───── MAJOR: New functionality
```

### Version Component Definitions

#### MAJOR (vX.0.0)

Increment for **new functionality or capabilities**:

- Adding a new phenological category (e.g., leaf phenology in addition to reproductive)
- Major architectural changes (e.g., switching from ViT to a completely different architecture)
- Breaking changes to output format or API

**Examples:**
- `v1.0.0` → `v2.0.0`: Added leaf phenology detection
- `v2.0.0` → `v3.0.0`: Added flower color classification

#### MINOR (v1.X.0)

Increment for **model logic or hyperparameter changes**:

- Adding/modifying data augmentation strategies
- Changing loss functions
- Modifying learning rate schedules
- Architectural tweaks (e.g., adding dropout layers)
- Changes to training procedure that affect model weights

**Examples:**
- `v1.0.0` → `v1.1.0`: Added RandAugment and Random Erasing
- `v1.1.0` → `v1.2.0`: Changed from BCE loss to Focal loss
- `v1.2.0` → `v1.3.0`: Added attention pooling layer

#### PATCH (v1.1.X)

Increment for **data-only updates**:

- Retraining on new annotated images (same model logic)
- Updating to new version of iNaturalist data
- Adjusting train/val/test split (same underlying data)
- Bug fixes in data processing that don't change model logic

**Examples:**
- `v1.1.0` → `v1.1.1`: Retrained on 6 months of new iNaturalist annotations
- `v1.1.1` → `v1.1.2`: Fixed data processing bug and retrained
- `v1.1.2` → `v1.1.3`: Updated to iNaturalist 2025-06 export

## Version History

### Reproductive Structure Detection

| Version  | Date       | Description                                    | Model DOI         | Training Data |
|----------|------------|------------------------------------------------|-------------------|---------------|
| v1.0.0   | 2024-11-13 | Initial release (implicit version)             | 10.57967/hf/2763  | iNat 2024-11  |
| v1.1.0   | 2025-11-17 | Added RandAugment + Random Erasing             | TBD               | iNat 2024-11 (bad images filtered) |

### Leaf Phenology Detection

| Version  | Date       | Description                                    | Model DOI         | Training Data |
|----------|------------|------------------------------------------------|-------------------|---------------|
| v1.0.0   | 2024-12-02 | Initial release (implicit version)             | 10.57967/hf/5785  | Leaves 2024   |

## Version Metadata

Each training run creates metadata documenting:

- **Version**: The semantic version (e.g., `v1.1.0`)
- **Split timestamp**: When the train/val/test split was created
- **Split parameters**: Proportions (60/20/20), seed, stratification variables
- **Data statistics**:
  - Total images in dataset
  - Good images (post-filtering)
  - Bad images (filtered out)
  - Train/val/test counts
- **File paths**: Locations of versioned CSV files

**Metadata file:** `data/inat/split_metadata_v1.1.0.json`

## Triggering a New Version

### To create a new minor version (model changes):

1. **Update version number** in `_targets_train_repro.R`:
   ```r
   model_version = "v1.2.0",  # Increment MINOR
   ```

2. **Make model changes** in `R/vit_model_train.R`:
   - Modify augmentation strategy
   - Change loss function
   - Adjust hyperparameters
   - Update architecture

3. **Run training pipeline**:
   ```bash
   Rscript run_pipeline.R --pipeline=train_repro
   ```

4. **Document changes** in this file's Version History table

### To create a new patch version (data update):

1. **Update version number** in `_targets_train_repro.R`:
   ```r
   model_version = "v1.1.1",  # Increment PATCH
   ```

2. **Update data source** (if needed):
   - Run download pipeline with new data
   - Or use existing data with different split

3. **Run training pipeline**:
   ```bash
   Rscript run_pipeline.R --pipeline=train_repro
   ```

4. **Document changes** in this file's Version History table

## Versioned Artifacts

Each version creates the following artifacts:

### Training Data
- `data/inat/train_v1.1.0.csv` - Training split
- `data/inat/val_v1.1.0.csv` - Validation split
- `data/inat/test_v1.1.0.csv` - Test split
- `data/inat/split_metadata_v1.1.0.json` - Split metadata

### Model Outputs
- `output/reproductive/v1.1.0/` - Model directory
- `output/reproductive/v1.1.0/checkpoints/phenovision_v1.1.0_epoch1.pt` - Epoch 1 checkpoint
- `output/reproductive/v1.1.0/checkpoints/phenovision_v1.1.0_epoch2.pt` - Epoch 2 checkpoint
- `output/reproductive/v1.1.0/checkpoints/phenovision_v1.1.0_epochN.pt` - Final checkpoint

### Guild AI Runs
- `.guild/runs/<run_id>/` - Complete experiment tracking
- Includes: metrics, parameters, code snapshot, outputs

## Data Provenance

### Bad Image Filtering

Starting with `v1.1.0`, all versions include automated bad image detection and filtering:

1. **Detection**: `detect_bad_images()` tests all images before splitting
2. **Filtering**: Only images marked `good == TRUE` are included in train/val/test
3. **Logging**: Bad images are logged for investigation
4. **Metadata**: Number of good/bad images recorded in split metadata

**Benefits:**
- No corrupt images in training data
- Faster training (no dataloader errors)
- Reproducible image quality standards
- Trackable bad image rate by batch/source

### Stratified Splitting

Train/val/test splits are stratified to ensure balanced representation:

**Reproductive model stratification:**
- Taxonomic order
- Flowering state (0/1)
- Fruiting state (0/1)

**Leaf model stratification:**
- Taxonomic family
- Green leaves state (0/1)
- Colored leaves state (0/1)
- Breaking buds state (0/1)

**Parameters:**
- Train: 60%
- Validation: 20%
- Test: 20%
- Seed: 234987 (for reproducibility)

## Linking Versions to DOIs

After training and validation:

1. **Upload model to HuggingFace Hub**:
   - Repository: `phenobase/phenovision` (reproductive) or `phenobase/phenovisionL` (leaf)
   - Checkpoint: `output/reproductive/v1.1.0/checkpoints/phenovision_v1.1.0_epochN.pt`
   - Include threshold buffers and family stats

2. **Create DOI via HuggingFace**:
   - Go to model repository settings
   - Click "Create DOI"
   - DOI format: `10.57967/hf/{model_id}`

3. **Update version history**:
   - Add DOI to Version History table above
   - Note which version corresponds to which DOI

4. **Update inference pipeline**:
   - Edit `_targets_common.R`:
     ```r
     config <- list(
       model_doi_repro = "10.57967/hf/XXXX",  # Update to new DOI
       ...
     )
     ```

## Best Practices

### 1. Version Bumping Guidelines

- **MAJOR**: Reserve for major milestones (new phenology types, API changes)
- **MINOR**: Use liberally for any model logic changes
- **PATCH**: Use for any data-only retraining

**When in doubt, increment MINOR** rather than PATCH if there's any code change.

### 2. Documentation

For each new version, document:
- What changed (in this file's Version History)
- Why the change was made
- Expected impact on model performance
- Any breaking changes to downstream pipelines

### 3. Testing

Before releasing a new version:
1. Run full training pipeline
2. Validate on test set
3. Compare metrics to previous version
4. Test inference pipeline with new model
5. Spot-check predictions on sample images

### 4. Git Workflow

Tag git commits with version numbers:
```bash
git tag -a v1.1.0 -m "Add RandAugment and Random Erasing"
git push origin v1.1.0
```

### 5. Backward Compatibility

Maintain backward compatibility for:
- Inference pipeline (should work with any model version)
- Output format (CSV columns, parquet schema)
- Metadata format (JSON structure)

**If breaking changes are needed**, increment MAJOR version.

## Questions & Troubleshooting

### Q: Can I retrain an old version with new data?

**A:** Yes! The version is tied to the model logic, not the data. To retrain `v1.1.0` with new data:

1. Keep `model_version = "v1.1.0"` (model logic unchanged)
2. Update to new data source in download pipeline
3. Increment PATCH when saving: `v1.1.1`
4. Document in Version History that `v1.1.1` is `v1.1.0` logic with new data

### Q: What if I forget to bump the version?

**A:** The pipeline will **overwrite** the previous version's files. This is by design - bumping the version is how you trigger a fresh run. If you accidentally overwrote a version:

1. Check `.guild/runs/` - the old run is still there with all artifacts
2. Manually copy the checkpoint to a new version folder
3. Update version history accordingly

### Q: How do I compare two versions?

Use Guild AI:
```r
library(guildai)
runs_info() %>% filter(tag == "reproductive")
runs_compare(runs = c("run1_id", "run2_id"))
```

Or compare CSVs:
```r
v1 <- read_csv("data/inat/split_metadata_v1.0.0.json")
v2 <- read_csv("data/inat/split_metadata_v1.1.0.json")
```

## References

- **Semantic Versioning**: https://semver.org/
- **ML Model Versioning Best Practices**: https://neptune.ai/blog/version-control-for-ml-models
- **Guild AI Documentation**: https://my.guild.ai/docs/
- **PhenoVision Paper**: Dinnage et al. (2025). Methods in Ecology and Evolution 16:1763-1780
