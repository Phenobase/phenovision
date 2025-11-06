# Multi-Pipeline Targets Refactor - Progress Checklist

**Last Updated:** 2025-01-06
**Current Phase:** Phase 0b - Guild AI Setup
**Branch:** targets-refactor (not yet created)

---

## Quick Status Overview

- [ ] Phase 0a: Git Branch Setup
- [x] Phase 0b: Guild AI Setup (partial - documentation complete, testing pending)
- [ ] Phase 1: Infrastructure
- [ ] Phase 2: Refactor Functions
- [ ] Phase 3: Inference Pipeline
- [ ] Phase 4: Training Pipelines
- [ ] Phase 5: Download Pipeline
- [ ] Phase 6: Testing & Documentation
- [ ] Phase 7: Merge to Main

**Completion:** 2/50 tasks (4%)

---

## Phase 0a: Git Branch Setup

- [ ] Check git status to verify clean state
- [ ] Create feature branch: `git checkout -b targets-refactor`

**Status:** Not Started
**Blockers:** None
**Notes:** Ready to proceed

---

## Phase 0b: Setup Guild AI & Test

- [x] Retrieved Guild AI examples from bioclim_intrinsic_dimension project
- [x] Documented Guild AI integration patterns in TARGETS_REFACTOR_PLAN.md
- [ ] Install guildai R package
  - Command: `install.packages("guildai")` or `remotes::install_github("t-kalinowski/guildai-r")`
  - Test in R console on HiPerGator
- [ ] Test Guild AI basic functionality
  - Check if `guild` command is available
  - Test `library(guildai)`
  - Test basic `guild_run()` call with dummy script
- [ ] Create `R/run_training_guild_training.R` wrapper function
  - Based on bioclim_intrinsic_dimension example
  - Include `assignInNamespace` workaround
  - Test with simple R script first
- [ ] Test wrapper with existing `R/vit_model_train.R`
  - Create minimal test with 1-2 epochs
  - Verify flags are passed correctly
  - Check that run_info returns expected structure
  - Verify checkpoint paths are accessible
- [ ] Git commit: "Add Guild AI integration and wrapper function"

**Status:** In Progress (2/6 completed)
**Blockers:** Need to test on HiPerGator environment
**Notes:**
- Guild AI workaround identified: `assignInNamespace("find_guild", guildai:::find_r_guildai_guild, ns = "guildai")`
- No script modifications needed - flags override defaults automatically

---

## Phase 1: Setup Infrastructure

- [ ] Create `_targets_common.R`
  - Define common paths (data, output, models)
  - Load shared libraries
  - Define function sourcing helpers
  - Add configuration variables (batch size, etc.)
- [ ] Create `run_pipeline.R` wrapper script
  - Use argparse for CLI
  - Implement --pipeline argument (download_annots, train_repro, train_leaf, inference)
  - Implement --submit flag for SLURM submission
  - Implement --workers argument for parallelization
  - Add help text and examples
- [ ] Create `scripts/submit_download_annots.sh`
- [ ] Create `scripts/submit_train_repro.sh`
- [ ] Create `scripts/submit_train_leaf.sh`
- [ ] Create `scripts/submit_inference.sh`
- [ ] Test run_pipeline.R locally (without targets files yet)
- [ ] Git commit: "Add pipeline infrastructure and wrapper scripts"

**Status:** Not Started
**Blockers:** Phase 0b must complete first
**Notes:** Template available from existing scripts/pheno_vit_finetune_*.sh

---

## Phase 2: Refactor Existing Functions by Theme

### Identify Functions to Extract
- [ ] Read `R/ml_annotation_targets.r` and list all functions
- [ ] Read `R/ml_annotation_postprocess_targets.R` and list all functions
- [ ] Read `R/model_loading_targets.R` and list all functions
- [ ] Categorize each function as: common, inference, training, or download

### Create New Function Files (_common suffix)
- [ ] `R/load_image_metadata_common.R`
- [ ] `R/load_model_from_doi_common.R`
- [ ] `R/create_batch_groups_common.R`
- [ ] `R/load_thresholds_common.R`
- [ ] Add others as identified

### Create New Function Files (_inference suffix)
- [ ] `R/annotate_batch_inference.R`
- [ ] `R/threshold_annotations_inference.R`
- [ ] `R/convert_to_long_inference.R`
- [ ] `R/aggregate_by_obs_inference.R`
- [ ] `R/finalize_annotations_inference.R`
- [ ] Add others as identified

### Testing
- [ ] Test sourcing pattern: `list.files("R", pattern = "_common\\.R$")`
- [ ] Verify all functions load correctly
- [ ] Check for naming conflicts

### Git Commit
- [ ] Git add all new R/*_common.R files explicitly
- [ ] Git add all new R/*_inference.R files explicitly
- [ ] Git commit: "Refactor existing functions into themed single-function files"

**Status:** Not Started
**Blockers:** Phase 1 should complete first
**Notes:** Maintain existing function signatures for compatibility

---

## Phase 3: Refactor Inference Pipeline

- [ ] Create `_targets_inference.R` file
- [ ] Add sourcing for _common and _inference functions
- [ ] Implement model DOI targets (repro and leaves)
- [ ] Implement model loading targets
- [ ] Implement image metadata loading
- [ ] Implement batch creation targets
- [ ] Implement reproductive annotation branch
- [ ] Implement leaf annotation branch (with taxonomy filtering)
- [ ] Implement threshold application (both branches)
- [ ] Implement annotation merging
- [ ] Implement observation-level aggregation
- [ ] Implement output writing targets
- [ ] Test pipeline runs without errors
- [ ] Compare outputs to current `_targets.R` (for leaf annotations)
- [ ] Test that changing `model_doi_leaves` only invalidates leaf branch
- [ ] Test that changing `model_doi_repro` only invalidates repro branch
- [ ] Test targets caching with `tar_make()` (second run should skip completed)
- [ ] Git commit: "Refactor inference pipeline with independent model branches"

**Status:** Not Started
**Blockers:** Phase 2 must complete first
**Notes:** This is the core pipeline - test thoroughly

---

## Phase 4a: Create Reproductive Training Pipeline

- [ ] Create `R/run_training_guild_training.R` if not done in Phase 0b
- [ ] Create `R/compute_thresholds_training.R`
- [ ] Create `R/compute_family_stats_training.R`
- [ ] Create `R/upload_to_huggingface_training.R`
- [ ] Create `_targets_train_repro.R` file
- [ ] Implement config target with training parameters
- [ ] Implement training_run target using Guild AI wrapper
- [ ] Implement test_results target
- [ ] Implement family_stats target
- [ ] Implement hf_model upload target
- [ ] Implement DOI reminder target
- [ ] Test with small training run (2-3 epochs)
- [ ] Verify Guild AI tracks the run
- [ ] Verify checkpoint is saved correctly
- [ ] Verify thresholds are computed
- [ ] Verify HuggingFace upload works
- [ ] Test resume functionality (set resume_from parameter)
- [ ] Git commit: "Add reproductive training pipeline with Guild AI"

**Status:** Not Started
**Blockers:** Phase 0b must complete first (Guild AI setup)
**Notes:** Start with small test run to validate workflow

---

## Phase 4b: Create Leaf Training Pipeline

- [ ] Create `_targets_train_leaf.R` file
- [ ] Implement config target (3 output classes)
- [ ] Implement first-round training target
- [ ] Implement second-round data filtering target
- [ ] Implement second-round training target
- [ ] Implement test_results target
- [ ] Implement family_stats target
- [ ] Implement hf_model upload target (phenobase/phenovisionL repo)
- [ ] Implement DOI reminder target
- [ ] Test with small training run
- [ ] Test two-round training workflow
- [ ] Git commit: "Add leaf training pipeline with Guild AI"

**Status:** Not Started
**Blockers:** Phase 4a should complete first (learn from repro pipeline)
**Notes:** Two-round training is more complex - test carefully

---

## Phase 5: Create Download Pipeline

- [ ] Create `R/download_dwc_archive_download.R` wrapper
- [ ] Create `R/extract_annotations_download.R` wrapper
- [ ] Create `R/split_data_download.R` wrapper
- [ ] Create `_targets_download_annots.R` file
- [ ] Implement dwc_archive download target
- [ ] Implement annotation extraction targets (repro and leaf)
- [ ] Implement data splitting target
- [ ] Implement output writing targets
- [ ] Test download from iNaturalist (may need credentials)
- [ ] Test data splitting produces correct proportions
- [ ] Verify output files in correct locations
- [ ] Git commit: "Add annotation download pipeline"

**Status:** Not Started
**Blockers:** Phase 3 recommended (understand data flow first)
**Notes:** May need user input for iNaturalist credentials or API access

---

## Phase 6: Testing & Documentation

### Comprehensive Testing
- [ ] Run `_targets_download_annots.R` end-to-end
- [ ] Run `_targets_train_repro.R` end-to-end (full epochs)
- [ ] Run `_targets_train_leaf.R` end-to-end (full epochs)
- [ ] Run `_targets_inference.R` end-to-end
- [ ] Test cross-pipeline: train → update DOI → inference
- [ ] Test run_pipeline.R with all pipeline options
- [ ] Test SLURM submission scripts
- [ ] Test targets cache behavior (invalidation, skipping)
- [ ] Compare inference outputs to main branch (regression test)
- [ ] Test error handling (corrupt images, missing files, etc.)
- [ ] Check memory usage and performance
- [ ] Verify parallel processing works correctly

### Documentation Updates
- [ ] Update CLAUDE.md with new architecture
  - Add section on multiple pipelines
  - Document function naming conventions
  - Add usage examples
  - Update file organization section
- [ ] Create README_pipelines.md
  - Overview of each pipeline
  - How to run each pipeline
  - Configuration options
  - Troubleshooting guide
- [ ] Document Guild AI integration in CLAUDE.md
- [ ] Add inline comments to complex functions
- [ ] Update .guildignore if needed

### Code Quality
- [ ] Fix `breaking_nuds` typo if modifying those files
  - `R/ml_annotation_targets.r:119`
  - `R/ml_annotation_postprocess_targets.R:212,232,334`
- [ ] Check for consistent code style
- [ ] Remove any debugging print statements
- [ ] Verify error messages are helpful

### Git Commit
- [ ] Git add CLAUDE.md README_pipelines.md
- [ ] Git add any modified R files (typo fixes)
- [ ] Git commit: "Add documentation for refactored pipelines"

**Status:** Not Started
**Blockers:** All previous phases must complete
**Notes:** This is the quality gate before merging

---

## Phase 7: Merge to Main

- [ ] Final verification: all tests pass
- [ ] Review all changes: `git diff main..targets-refactor`
- [ ] Check for any uncommitted changes: `git status`
- [ ] Push feature branch: `git push origin targets-refactor`
- [ ] Create pull request (if using PR workflow)
- [ ] Review PR / do final self-review
- [ ] Merge to main: `git checkout main && git merge targets-refactor`
- [ ] Test on main branch (smoke test)
- [ ] Push to remote: `git push origin main`
- [ ] Create tag: `git tag -a v2.0-multi-pipeline -m "Multi-pipeline targets refactor"`
- [ ] Push tags: `git push --tags`
- [ ] Announce to team / update documentation

**Status:** Not Started
**Blockers:** All previous phases must complete and pass testing
**Notes:** This is the final step - be thorough

---

## Issue Tracker

### Known Issues
1. **Guild AI Discovery Bug**
   - Workaround: `assignInNamespace("find_guild", guildai:::find_r_guildai_guild, ns = "guildai")`
   - May be fixed in newer versions - test current version first

2. **Typo: breaking_nuds**
   - Should be: breaking_buds
   - Appears in multiple files
   - Only fix if we're modifying those files anyway

3. **Large Files in Git**
   - Be careful with `git add`
   - Always use explicit file paths
   - Check `.gitignore` is working

### Blockers
None currently

### Questions for User
None currently

---

## Notes & Decisions Log

### 2025-01-06
- **Decision:** Use Guild AI for training experiment tracking
  - Rationale: Automatic parameter tracking, no script changes needed
  - Example from bioclim_intrinsic_dimension project
  - Known workaround for discovery bug

- **Decision:** Function organization by theme, not pipeline
  - Rationale: Better reusability, clearer dependencies
  - Naming: `{function_name}_{theme}.R` (e.g., `load_image_metadata_common.R`)

- **Decision:** Targets cache for tracking processed images
  - Rationale: Simpler than manual tracking, built into targets
  - Model DOI changes trigger re-inference automatically

- **Decision:** Feature branch workflow
  - Branch: `targets-refactor`
  - Explicit file adds only
  - Regular commits after each phase

---

## Quick Reference

### Git Commands
```bash
# Create branch
git checkout -b targets-refactor

# Add specific files
git add file1.R file2.R

# Check status
git status
git diff --cached

# Commit
git commit -m "Description"

# Push branch
git push origin targets-refactor
```

### Test Commands
```bash
# Run specific pipeline
Rscript run_pipeline.R --pipeline=inference --workers=4

# Submit to SLURM
Rscript run_pipeline.R --pipeline=train_repro --submit

# Check targets
Rscript -e "targets::tar_make(script = '_targets_inference.R')"
Rscript -e "targets::tar_visnetwork(script = '_targets_inference.R')"
```

### Useful Checks
```bash
# Check Guild AI installed
Rscript -e "library(guildai); packageVersion('guildai')"

# List Guild runs
guild runs

# Check file sizes before adding
ls -lh R/*.R
```

---

## Agent Handoff Checklist

When handing off to another agent, ensure:
- [ ] This progress file is up to date
- [ ] Any uncommitted changes are noted
- [ ] Current blockers are documented
- [ ] Test results are recorded
- [ ] Questions for user are listed
- [ ] Recent decisions are logged

**Current Status for Handoff:**
- Phase 0b in progress: Guild AI documentation complete, testing pending
- No blockers
- Ready to proceed with Guild AI package installation and testing

---

**For Future Agents:**
- Read `TARGETS_REFACTOR_PLAN.md` first for full context
- Check this file for current status
- Update this file as you complete tasks
- Log important decisions in Notes & Decisions
- Ask user before making major changes to the plan
