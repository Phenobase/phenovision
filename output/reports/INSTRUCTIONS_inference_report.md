# Instructions: Generating an Inference Pipeline Report

Use the following prompt after an inference pipeline run completes to generate a status report.

## Prompt for Claude Agent

```
The inference pipeline just finished. Please generate a report for this run. Check the following and write a markdown report to `output/reports/`:

1. **Pipeline log**: Find the most recent `logs/inference_run_*.log` and check the tail for completion status, duration, and error/warning counts.

2. **SLURM logs**: Check `logs/pheno_inference-*.out` and `logs/pheno_inference-*.err` for OOM kills or other system-level errors.

3. **Progress file**: Read `_targets_stores/inference/meta/progress` and count:
   - Total completed, errored, skipped
   - GPU branches completed per model (annotations_repro, annotations_leaves)
   - List all errored branch targets (excluding pattern-level and cascade errors)

4. **Output files**: Check combined CSVs exist and report sizes/row counts:
   - `output/reproductive/{version}/inference/annotations_internal_all.csv`
   - `output/reproductive/{version}/inference/annotations_ingest_all.csv`
   - `output/leaves/{version}/inference/annotations_internal_all.csv`
   - `output/leaves/{version}/inference/annotations_ingest_all.csv`
   - Count per-batch files in `final_internal/` and `final_ingest/` subdirectories

5. **Data quality**: Verify:
   - Combined CSVs have exactly 1 header row (grep for header pattern)
   - Column counts match expected schema (22 for ingest, 29 for internal)
   - Row counts in combined CSV match sum of per-batch files

6. **Error analysis**: For each errored target:
   - Find the error message in the log
   - Check if a workspace was saved (`_targets_stores/inference/workspaces/`)
   - Categorize as: transient (will retry), data issue (needs fix), or code bug

7. **Write report** to `output/reports/inference_run_{DATE}_{models}.md` following the template in this directory.

Check model versions from `_targets_inference.R` (look for `model_version_repro` and `model_version_leaves` targets).
```

## Key Commands

```bash
# Check progress counts
grep -c 'completed' _targets_stores/inference/meta/progress
grep -c 'errored' _targets_stores/inference/meta/progress

# List errored branches (excluding cascades)
grep "errored" _targets_stores/inference/meta/progress | grep -v '|pattern|'

# GPU branch counts
grep 'completed' _targets_stores/inference/meta/progress | grep 'annotations_repro_' | grep -v 'thresholded\|long\|by_obs\|internal\|ingest' | wc -l
grep 'completed' _targets_stores/inference/meta/progress | grep 'annotations_leaves_' | grep -v 'thresholded\|long\|by_obs\|internal\|ingest' | wc -l

# Combined CSV row counts
wc -l output/reproductive/*/inference/annotations_*_all.csv
wc -l output/leaves/*/inference/annotations_*_all.csv

# Check for duplicate headers
grep -c "^datasource," output/reproductive/*/inference/annotations_internal_all.csv

# Find error messages in log
grep "Error:" logs/inference_run_*.log | grep -v "could not load dependency"

# Check SLURM OOM
grep "oom_kill" logs/pheno_inference-*.err
```

## Report Naming Convention

`inference_run_{YYYY-MM-DD}_repro-v{X.Y.Z}_leaves-v{X.Y.Z}.md`

Example: `inference_run_2026-03-12_repro-v1.1.0_leaves-v1.0.0.md`
