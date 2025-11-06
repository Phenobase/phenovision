# Guild AI Testing Checklist

**Purpose:** Verify Guild AI R package is installed and working before proceeding with implementation

**Run these tests in your RStudio session or R console on HiPerGator**

---

## Test 1: Check Installation

```r
# Check if guildai package is installed
if (requireNamespace('guildai', quietly = TRUE)) {
  library(guildai)
  cat('Guild AI version:', as.character(packageVersion('guildai')), '\n')
} else {
  cat('Guild AI not installed. Installing...\n')
  install.packages("guildai")
}
```

**Expected:** Version number printed (e.g., "Guild AI version: 0.x.x")

---

## Test 2: Basic Guild AI Functionality

```r
library(guildai)

# Test if find_guild workaround is needed
tryCatch({
  guild_help()
  cat("Guild AI working without workaround\n")
}, error = function(e) {
  cat("Guild AI needs workaround. Error:\n", e$message, "\n")
})
```

**Expected:** Either help text appears, or error message indicating workaround is needed

---

## Test 3: Apply Workaround (if needed)

```r
library(guildai)

# Apply the workaround from bioclim_intrinsic_dimension project
assignInNamespace("find_guild", guildai:::find_r_guildai_guild, ns = "guildai")

# Set Guild home
Sys.setenv(GUILD_HOME = "/blue/guralnick/share/r.dinnage/Projects/phenovision/.guild")

# Try again
guild_help()
```

**Expected:** Help text should appear

---

## Test 4: Create Simple Test Script

Create a file `test_guild_simple.R` with this content:

```r
# Simple test script for Guild AI
# Any top-level scalar assignment becomes a flag

epochs <- 5
learning_rate <- 0.001
message <- "test"

# Simulate some work
cat("Starting test with:\n")
cat("  epochs:", epochs, "\n")
cat("  learning_rate:", learning_rate, "\n")
cat("  message:", message, "\n")

# Simulate training loop
for (i in 1:epochs) {
  cat("Epoch", i, "- loss:", runif(1), "\n")
  Sys.sleep(0.5)
}

# Save a simple output
cat("Test complete\n")
writeLines(paste("Completed with", epochs, "epochs"), "test_output.txt")
```

---

## Test 5: Run Script with Guild AI

```r
library(guildai)

# Apply workaround if needed
assignInNamespace("find_guild", guildai:::find_r_guildai_guild, ns = "guildai")
Sys.setenv(GUILD_HOME = "/blue/guralnick/share/r.dinnage/Projects/phenovision/.guild")

# Run the test script
guild_run(
  "test_guild_simple.R",
  label = "test_run_1",
  tag = "testing",
  as_job = FALSE,
  flags = list(epochs = 3, learning_rate = 0.0005)
)
```

**Expected:**
- Script runs with overridden parameters (3 epochs, not 5)
- No errors
- Output files created

---

## Test 6: Retrieve Run Information

```r
# Get information about the run
run_info <- runs_info(label = "test_run_1")

print(run_info)

# Check the run directory
cat("Run directory:", run_info$dir, "\n")

# List files in run directory
list.files(run_info$dir, recursive = TRUE)
```

**Expected:**
- run_info should contain metadata about the run
- Run directory should contain script outputs
- Should see test_output.txt in the files

---

## Test 7: Multiple Runs with Different Flags

```r
# Run with different parameters
guild_run(
  "test_guild_simple.R",
  label = "test_run_2",
  tag = "testing",
  as_job = FALSE,
  flags = list(epochs = 10, learning_rate = 0.01, message = "second_test")
)

# Compare runs
all_runs <- runs_info()
print(all_runs[c("label", "started")])
```

**Expected:** Both runs appear in the runs list

---

## Test 8: Guild Command Line (Optional)

In a terminal:

```bash
# Check if guild command is available
guild --version

# List runs
guild runs

# View a specific run
guild runs info test_run_1
```

**Expected:** Guild CLI works (optional - R interface is sufficient)

---

## Results Summary

After running these tests, document:

1. **Guild AI Version:** _______________
2. **Workaround Needed?** Yes / No
3. **Guild Home Path:** _______________
4. **Test Script Ran Successfully?** Yes / No
5. **Run Info Retrieved Successfully?** Yes / No
6. **Issues Encountered:** _______________

---

## If Guild AI Not Working

### Option 1: Try GitHub version
```r
remotes::install_github("t-kalinowski/guildai-r")
```

### Option 2: Install Python guild first
```bash
pip install guildai
```
Then reinstall R package.

### Option 3: Check for conflicts
```r
# Unload and reload
detach("package:guildai", unload = TRUE)
.rs.restartR()
library(guildai)
```

---

## When Tests Pass

✅ Update `TARGETS_REFACTOR_PROGRESS.md`
✅ Mark "Install and test Guild AI R package setup" as complete
✅ Proceed to creating the Guild AI wrapper function

---

**Note:** These tests should be run in an interactive R session (RStudio or R console), not via Rscript, as Guild AI may need interactive features.
