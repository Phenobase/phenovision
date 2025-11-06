# Multi-Pipeline Targets Refactor - Detailed Implementation Plan

**Project:** PhenoVision Multi-Pipeline Architecture
**Branch:** targets-refactor
**Status:** In Progress
**Started:** 2025-01-06

## Table of Contents
1. [Overview](#overview)
2. [Guild AI Integration Learnings](#guild-ai-integration-learnings)
3. [Architecture Design](#architecture-design)
4. [Implementation Phases](#implementation-phases)
5. [Function Naming Conventions](#function-naming-conventions)
6. [Git Workflow](#git-workflow)
7. [Testing Strategy](#testing-strategy)

---

## Overview

### Goal
Create 5 separate targets pipeline files that share common functions organized by THEME (not by pipeline). Use Guild AI to run existing training scripts unchanged through the targets framework.

### Key Requirements
1. **Minimize redundant work**: Only process new images or when model versions change
2. **Independent model tracking**: Leaf and reproductive models update separately
3. **Preserve existing code**: Training scripts remain unchanged
4. **Easy execution**: Single wrapper script for local or SLURM execution
5. **Proper git workflow**: Work in feature branch, merge only when tested

### Five Pipelines
1. **Download Images** (future - not in current scope, user has script from collaborator)
2. **Download Annotations** (_targets_download_annots.R)
3. **Train Reproductive Model** (_targets_train_repro.R)
4. **Train Leaf Model** (_targets_train_leaf.R)
5. **Inference** (_targets_inference.R)

---

## Guild AI Integration Learnings

### From bioclim_intrinsic_dimension Project

**Repository:** https://github.com/rdinnager/bioclim_intrinsic_dimension

#### How Guild AI Works with R

1. **Automatic Flag Detection**
   - Guild AI automatically detects any scalar literal assigned at the top level of an R script
   - No code changes needed in the training scripts!
   - Parameters are "flags" that Guild tracks automatically

2. **The Wrapper Function Pattern**

   From `R/target_functions.R`:
   ```r
   run_model_guildai <- function(script = "bioclim_ivae_lat_long.R",
                                 label, tag, flags, comment = "",
                                 GUILD_HOME = "/path/to/project/.guild") {
     # WORKAROUND: Fix Guild AI discovery bug
     assignInNamespace("find_guild", guildai:::find_r_guildai_guild, ns = "guildai")

     # Set Guild home directory
     Sys.setenv(GUILD_HOME = GUILD_HOME)

     # Run the script with Guild AI
     guild_run(script, label = label, tag = tag, as_job = FALSE,
               flags = flags, comment = comment)

     # Get run information
     run_info <- runs_info(label = label)
     return(run_info)
   }
   ```

3. **Integration with Targets**

   From `_targets.R`:
   ```r
   tar_target(model_runs_2.5,
              run_model_guildai(script = ivae_lat_long_script,
                                label = model_run_labels_2.5,
                                tag = "ident_lat_lon_weighted_scratch",
                                flags = list(loggamma_init = -5.0,
                                             num_epochs = 10000,
                                             lr = 0.0005,
                                             res = "2.5")),
              pattern = map(model_run_labels_2.5))
   ```

4. **Key Insights**
   - **No script modification needed**: Existing training scripts work as-is
   - **Flags override defaults**: Pass parameters via `flags` argument
   - **Metadata tracking**: Guild AI automatically tracks runs, outputs, metrics
   - **Bug workaround needed**: `assignInNamespace` call fixes Guild discovery issue
   - **Returns run info**: Can extract checkpoint paths, metrics from `runs_info()`

5. **Files to Ignore** (from .guildignore)
   ```
   _targets
   data
   ```

#### Guild AI R Package Status
- **CRAN version**: Available (as of 2024-2025)
- **Documentation**: https://guildai.github.io/guildai-r/
- **Known Issues**: Discovery bug (workaround shown above)
- **Installation**: `install.packages("guildai")` or `remotes::install_github("t-kalinowski/guildai-r")`

---

## Architecture Design

### File Structure

```
phenovision/
├── run_pipeline.R                          # NEW: Wrapper script
├── _targets_common.R                       # NEW: Shared configuration
├── _targets_download_annots.R              # NEW: Pipeline 2
├── _targets_train_repro.R                  # NEW: Pipeline 3
├── _targets_train_leaf.R                   # NEW: Pipeline 4
├── _targets_inference.R                    # NEW: Pipeline 5 (refactored)
├── _targets.R                              # EXISTING: Will keep for backward compatibility
│
├── R/                                      # Function files (one per file)
│   ├── *_common.R                          # Shared functions
│   ├── *_inference.R                       # Inference-specific
│   ├── *_training.R                        # Training-specific
│   ├── *_download.R                        # Download-specific
│   │
│   # Examples of new files:
│   ├── load_image_metadata_common.R
│   ├── load_model_from_doi_common.R
│   ├── create_batch_groups_common.R
│   ├── annotate_batch_inference.R
│   ├── threshold_annotations_inference.R
│   ├── aggregate_by_obs_inference.R
│   ├── run_training_guild_training.R       # Guild AI wrapper
│   ├── compute_thresholds_training.R
│   └── upload_to_hf_training.R
│
├── scripts/
│   ├── submit_download_annots.sh           # NEW
│   ├── submit_train_repro.sh               # NEW
│   ├── submit_train_leaf.sh                # NEW
│   └── submit_inference.sh                 # NEW
│
└── .claude_notes_and_utils/
    ├── TARGETS_REFACTOR_PLAN.md            # THIS FILE
    └── TARGETS_REFACTOR_PROGRESS.md        # Progress checklist
```

### Function Naming Convention

**Pattern:** `{function_name}_{pipeline_suffix}.R`

**Suffixes:**
- `_common`: Used by multiple pipelines
- `_inference`: Inference pipeline only
- `_training`: Training pipelines only
- `_download`: Download pipeline only

**Rules:**
1. One function per file (except very closely related helpers)
2. Function name in file matches filename (minus suffix)
3. Clear, descriptive names
4. Consistent with existing codebase style

**Sourcing Strategy:**
```r
# In _targets_inference.R:
purrr::walk(list.files("R", pattern = "_common\\.R$", full.names = TRUE), source)
purrr::walk(list.files("R", pattern = "_inference\\.R$", full.names = TRUE), source)
```

---

## Implementation Phases

### Phase 0a: Git Branch Setup
**Status:** Pending

1. Check git status to verify clean state
2. Create feature branch: `git checkout -b targets-refactor`
3. All subsequent work happens on this branch
4. Regular commits after each phase

### Phase 0b: Setup Guild AI & Test
**Status:** In Progress

1. ✅ **COMPLETED:** Retrieved Guild AI examples from bioclim_intrinsic_dimension
2. ✅ **COMPLETED:** Documented Guild AI integration patterns
3. **TODO:** Install guildai R package
   ```r
   install.packages("guildai")
   # OR if issues:
   remotes::install_github("t-kalinowski/guildai-r")
   ```
4. **TODO:** Test Guild AI setup on HiPerGator
   - Check if guild command is available
   - Test basic `guild_run()` call
5. **TODO:** Create Guild AI wrapper function based on examples
6. **TODO:** Test wrapper with existing `R/vit_model_train.R`
7. **COMMIT:** "Add Guild AI integration and wrapper function"

### Phase 1: Setup Infrastructure
**Status:** Pending

**Files to Create:**
- `_targets_common.R`
- `run_pipeline.R`
- `scripts/submit_download_annots.sh`
- `scripts/submit_train_repro.sh`
- `scripts/submit_train_leaf.sh`
- `scripts/submit_inference.sh`

**Tasks:**
1. Create `_targets_common.R` with shared configuration
   - Define common paths
   - Load shared libraries
   - Define helper functions for sourcing

2. Create wrapper script `run_pipeline.R`
   - argparse for command line interface
   - --pipeline, --submit, --workers arguments
   - Logic to call correct targets file or submit SLURM job

3. Create SLURM submission scripts
   - Similar structure to existing scripts/pheno_vit_finetune_*.sh
   - Call run_pipeline.R with --submit flag
   - Appropriate resources for each pipeline type

4. Set up function naming convention documentation

**COMMIT:**
```bash
git add _targets_common.R run_pipeline.R
git add scripts/submit_*.sh
git commit -m "Add pipeline infrastructure and wrapper scripts"
```

### Phase 2: Refactor Existing Functions by Theme
**Status:** Pending

**Strategy:**
1. Read current `_targets.R` and identify all custom functions
2. Extract each function to its own file
3. Add appropriate suffix (_common, _inference)
4. Test that sourcing works correctly

**Files to Extract From:**
- `R/ml_annotation_targets.r`
- `R/ml_annotation_postprocess_targets.R`
- `R/model_loading_targets.R`
- Any other files with targets-specific functions

**New Files to Create:**
- `R/load_image_metadata_common.R`
- `R/load_model_from_doi_common.R`
- `R/create_batch_groups_common.R`
- `R/annotate_batch_inference.R`
- `R/threshold_annotations_inference.R`
- `R/aggregate_by_obs_inference.R`
- `R/convert_to_long_inference.R`
- `R/finalize_annotations_inference.R`

**COMMIT:**
```bash
git add R/*_common.R R/*_inference.R
git commit -m "Refactor existing functions into themed single-function files"
```

### Phase 3: Refactor Inference Pipeline
**Status:** Pending

**Goal:** Create new `_targets_inference.R` with independent repro/leaf branches

**Key Features:**
- Separate model DOI tracking for reproductive and leaf models
- Model change triggers re-inference for that model only
- New images trigger re-inference for both models
- Targets cache handles "already processed" automatically

**Structure:**
```r
list(
  # Model versions (changing triggers re-inference)
  tar_target(model_doi_repro, "10.57967/hf/2763"),
  tar_target(model_doi_leaves, "10.57967/hf/5785"),

  # Load models
  tar_target(model_repro, load_model_from_doi(model_doi_repro)),
  tar_target(model_leaves, load_model_from_doi(model_doi_leaves)),

  # Load images (changing triggers processing new images)
  tar_target(images, load_image_metadata()),
  tar_target(batches, create_batch_groups(images)),
  tar_target(images_batch, batches, pattern = map(batches)),

  # REPRODUCTIVE BRANCH
  tar_target(annotations_repro,
             annotate_batch(images_batch, model_repro, "repro"),
             pattern = map(images_batch)),
  tar_target(thresholds_repro, load_thresholds_from_doi(model_doi_repro)),
  tar_target(annotations_thresholded_repro,
             threshold_annotations(annotations_repro, thresholds_repro),
             pattern = map(annotations_repro)),

  # LEAF BRANCH (with taxonomy filtering)
  tar_target(taxonomy_filter_leaves, get_target_genera_leaves()),
  tar_target(images_batch_leaves,
             filter_by_taxonomy(images_batch, taxonomy_filter_leaves),
             pattern = map(images_batch)),
  tar_target(annotations_leaves,
             annotate_batch(images_batch_leaves, model_leaves, "leaves"),
             pattern = map(images_batch_leaves)),
  tar_target(thresholds_leaves, load_thresholds_from_doi(model_doi_leaves)),
  tar_target(annotations_thresholded_leaves,
             threshold_annotations(annotations_leaves, thresholds_leaves),
             pattern = map(annotations_leaves)),

  # MERGE AND FINALIZE
  tar_target(annotations_combined,
             merge_annotations(annotations_thresholded_repro,
                              annotations_thresholded_leaves)),
  tar_target(annotations_by_obs, aggregate_by_obs(annotations_combined)),
  tar_target(annotations_final, finalize_for_phenobase(annotations_by_obs)),

  # OUTPUTS
  tar_target(outputs_internal,
             write_annotations(annotations_final, "internal"),
             format = "file"),
  tar_target(outputs_ingest,
             write_annotations(annotations_final, "ingest"),
             format = "file")
)
```

**Testing:**
- Compare outputs to current `_targets.R` (leaf annotations only)
- Verify targets cache works as expected
- Test that changing `model_doi_leaves` only invalidates leaf branch

**COMMIT:**
```bash
git add _targets_inference.R
git commit -m "Refactor inference pipeline with independent model branches"
```

### Phase 4: Create Training Pipelines
**Status:** Pending

#### Phase 4a: Reproductive Training Pipeline

**File:** `_targets_train_repro.R`

**Guild AI Wrapper:** `R/run_training_guild_training.R`

```r
run_training_guild <- function(script_path,
                               flags = list(),
                               label = NULL,
                               tag = NULL,
                               comment = "",
                               guild_home = ".guild") {

  # Load guildai package
  library(guildai)

  # WORKAROUND: Fix Guild AI discovery bug
  assignInNamespace("find_guild", guildai:::find_r_guildai_guild, ns = "guildai")

  # Set Guild home directory
  Sys.setenv(GUILD_HOME = guild_home)

  # Generate label if not provided
  if (is.null(label)) {
    label <- paste0("run_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  }

  # Run the script with Guild AI
  guild_run(script_path,
            label = label,
            tag = tag,
            as_job = FALSE,
            flags = flags,
            comment = comment)

  # Get run information
  run_info <- runs_info(label = label)

  # Extract useful paths
  result <- list(
    run_info = run_info,
    run_dir = run_info$dir,
    label = label,
    tag = tag,
    flags = flags
  )

  return(result)
}
```

**Pipeline Structure:**
```r
list(
  # Configuration
  tar_target(config_repro, list(
    pretrained_model = "plantclef",
    epochs = 100,
    batch_size = 384,
    lr = 0.0005,
    resume_from = NULL  # Set to checkpoint path to resume
  )),

  # Training run via Guild AI
  tar_target(training_run_repro,
             run_training_guild(
               script_path = "R/vit_model_train.R",
               flags = config_repro,
               label = paste0("repro_", Sys.Date()),
               tag = "reproductive_training",
               comment = "Reproductive structures model training"
             )),

  # Testing and threshold tuning
  tar_target(test_results_repro,
             run_testing_script(training_run_repro)),

  # Family-level accuracy statistics
  tar_target(family_stats_repro,
             compute_family_stats(test_results_repro)),

  # Convert and upload to HuggingFace
  tar_target(hf_model_repro,
             upload_to_huggingface(
               checkpoint_path = training_run_repro$run_info$checkpoint,
               repo = "phenobase/phenovision",
               model_type = "reproductive"
             )),

  # Reminder to get DOI
  tar_target(doi_reminder_repro, {
    message("Remember to get DOI from DataCite for commit: ",
            hf_model_repro$commit_hash)
    message("Update model_doi_repro in _targets_inference.R")
    hf_model_repro$commit_hash
  })
)
```

**COMMIT:**
```bash
git add _targets_train_repro.R R/run_training_guild_training.R R/*_training.R
git commit -m "Add reproductive training pipeline with Guild AI"
```

#### Phase 4b: Leaf Training Pipeline

**File:** `_targets_train_leaf.R`

**Similar to reproductive, but with:**
- Two-round training support
- 3 output classes
- Different script: `R/leaves_train_model.R`
- Different HF repo: `phenobase/phenovisionL`

**COMMIT:**
```bash
git add _targets_train_leaf.R
git commit -m "Add leaf training pipeline with Guild AI"
```

### Phase 5: Create Download Pipeline
**Status:** Pending

**File:** `_targets_download_annots.R`

**Wrapper Functions:**
- `R/download_dwc_archive_download.R`
- `R/extract_annotations_download.R`
- `R/split_data_download.R`

**Pipeline wraps existing scripts:**
- `phenobase/download_inat_data.py`
- `phenobase/split_data.py`

**COMMIT:**
```bash
git add _targets_download_annots.R R/*_download.R
git commit -m "Add annotation download pipeline"
```

### Phase 6: Testing & Documentation
**Status:** Pending

**Tasks:**
1. Test each pipeline independently
2. Test cross-pipeline dependencies (train → inference)
3. Verify outputs match main branch behavior (for inference)
4. Test on SLURM with submission scripts
5. Test that targets cache works as expected
6. Test changing model DOIs
7. Update CLAUDE.md with new architecture
8. Create README_pipelines.md
9. Fix `breaking_nuds` typo if we modify those files

**COMMIT:**
```bash
git add CLAUDE.md README_pipelines.md
# Only if fixing typos:
git add R/ml_annotation_targets.r R/ml_annotation_postprocess_targets.R
git commit -m "Add documentation for refactored pipelines"
```

### Phase 7: Merge to Main
**Status:** Pending

1. **Final verification**: All tests pass
2. **Push branch**: `git push origin targets-refactor`
3. **Review changes**: Check diff against main
4. **Merge**: `git checkout main && git merge targets-refactor`
5. **Push**: `git push origin main`
6. **Tag**: `git tag -a v2.0-multi-pipeline -m "Multi-pipeline targets refactor"`
7. **Push tags**: `git push --tags`

---

## Git Workflow

### Always Use Explicit File Paths

**CRITICAL:** This project has many large files that should not be committed.

```bash
# GOOD - explicit files
git add run_pipeline.R _targets_inference.R R/new_function_common.R

# BAD - will add unwanted files
git add .
git add R/
```

### Before Each Commit

```bash
git status          # See what's changed
git diff            # See changes in detail
git diff --cached   # See what's staged
```

### Files That Should NOT Be Committed
- Model checkpoints: `output/*/checkpoints/*.pt`
- Targets cache: `_targets/`
- Large data files: `data/`
- Guild runs: `.guild/`
- Images: `*.jpg`, `*.png` (except small documentation images)
- CSVs with results: `output/annotations/*.csv`

### New Files That SHOULD Be Committed
- All `_targets_*.R` files
- All `R/*_*.R` function files
- `run_pipeline.R`
- SLURM scripts: `scripts/submit_*.sh`
- Documentation: `*.md` files
- Configuration: `.guildignore` updates

---

## Testing Strategy

### Unit Testing
- Test each new function individually
- Verify function outputs match expected format
- Test edge cases (empty data, corrupt images, etc.)

### Integration Testing
- Run each pipeline end-to-end
- Verify outputs match expected locations
- Check file formats and contents

### Regression Testing
- Compare inference pipeline outputs to current `_targets.R`
- Ensure no breaking changes to data formats
- Verify backward compatibility where needed

### Performance Testing
- Compare runtime to current pipeline
- Check memory usage on large batches
- Verify parallel processing works correctly

### SLURM Testing
- Test each submission script
- Verify job completes successfully
- Check log files for errors

---

## Notes for Future Agents

### Where We Left Off
See `TARGETS_REFACTOR_PROGRESS.md` for current status of each task.

### Key Decision Points
1. **Guild AI vs Manual**: We chose Guild AI for automatic tracking
2. **Function Organization**: By theme (not pipeline) for reusability
3. **Targets Cache**: Using built-in caching instead of manual tracking
4. **Git Strategy**: Feature branch with explicit file adds

### Potential Issues to Watch
1. **Guild AI Bug**: May need `assignInNamespace` workaround
2. **Typo**: `breaking_nuds` exists in multiple files
3. **Large Files**: Be careful with git add
4. **Reticulate**: Python environment must be activated

### Helpful Resources
- Guild AI R docs: https://guildai.github.io/guildai-r/
- Example project: https://github.com/rdinnager/bioclim_intrinsic_dimension
- Current CLAUDE.md for project context
- .claude_notes_and_utils/ for analysis notes

---

## Success Criteria

✅ All 5 pipelines implemented and tested
✅ Guild AI integration working
✅ Training scripts unchanged
✅ Inference matches current behavior
✅ Documentation complete
✅ Merged to main without issues
✅ Team can use new pipelines easily

---

**Last Updated:** 2025-01-06
**Last Updated By:** Claude Code (initial plan creation)
