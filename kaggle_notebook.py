#!/usr/bin/env python3
"""
===================================================================
KAGGLE NOTEBOOK: Business Entity Resolution — Full Test Inference
===================================================================

SETUP INSTRUCTIONS (do these ONCE before running this notebook):
================================================================

1. CREATE A KAGGLE DATASET with your test files:
   - Go to https://kaggle.com/datasets → "New Dataset"
   - Name it: "amazon-ml-2026-test"
   - Upload these 3 files from your local machine:
       student_resource/dataset/test/test_source1.tsv  (167 MB)
       student_resource/dataset/test/test_source2.tsv  (486 MB)
       student_resource/dataset/test/test_source3.tsv  (482 MB)
   - Set visibility to "Private"
   - Click "Create"

2. Also upload these 2 local output files as ANOTHER dataset:
   - Go to https://kaggle.com/datasets → "New Dataset"
   - Name it: "amazon-ml-2026-partials"
   - Upload:
       student_resource/output/temp_partitions/match_France.tsv  (10.5 MB)
       student_resource/output/temp_partitions/cand_France.tsv   (77 MB)
       student_resource/output/temp_partitions/match_US.tsv      (26 MB)
       student_resource/output/temp_partitions/cand_US.tsv       (195 MB)
   - Set visibility to "Private"
   - Click "Create"

3. CREATE A NEW KAGGLE NOTEBOOK:
   - Go to https://kaggle.com/code → "New Notebook"
   - Set Accelerator to "None" (CPU only)
   - Set Persistence to "Files"
   - In the right sidebar, click "Add Data" → search for your datasets:
       - "amazon-ml-2026-test"
       - "amazon-ml-2026-partials"
   - Paste this ENTIRE script into a single code cell
   - Click "Run All"

The notebook will:
  1. Clone the GitHub repo
  2. Install dependencies
  3. Symlink datasets into the expected directory structure
  4. Run inference for ALL 3 countries (reusing France & US partitions)
  5. Assemble final submission files
  6. Run the official validator
  7. Save outputs to /kaggle/working/ for download
"""

import os
import subprocess
import sys
import shutil

# =====================================================================
# CONFIGURATION — Update these paths if your dataset names differ
# =====================================================================
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

# Dataset names on Kaggle (the directory names under /kaggle/input/)
# Kaggle converts dataset names to lowercase with hyphens
TEST_DATASET = "amazon-ml-2026-test"       # Contains test_source1/2/3.tsv
PARTIALS_DATASET = "amazon-ml-2026-partials"  # Contains match_France.tsv, etc.

GITHUB_REPO = "https://github.com/raviaddagalla/amazon_ml.git"
REPO_DIR = os.path.join(KAGGLE_WORKING, "amazon_ml")

# =====================================================================
# STEP 1: Obtain the repository code
# =====================================================================
print("=" * 70)
print("STEP 1: Obtaining repository code")
print("=" * 70)

code_copied = False

# Option A: Check if code was added as a Kaggle dataset (Offline mode)
if os.path.exists(KAGGLE_INPUT):
    for entry in os.listdir(KAGGLE_INPUT):
        entry_path = os.path.join(KAGGLE_INPUT, entry)
        if os.path.isdir(entry_path):
            # Check if this input folder contains student_resource directly or inside a subfolder
            target_sr = None
            if os.path.exists(os.path.join(entry_path, "student_resource")):
                target_sr = entry_path
            else:
                for sub in os.listdir(entry_path):
                    sub_p = os.path.join(entry_path, sub)
                    if os.path.isdir(sub_p) and os.path.exists(os.path.join(sub_p, "student_resource")):
                        target_sr = sub_p
                        break
            if target_sr:
                print(f"  Found code dataset in Kaggle input: {target_sr}")
                if os.path.exists(REPO_DIR):
                    shutil.rmtree(REPO_DIR)
                shutil.copytree(target_sr, REPO_DIR)
                print(f"  Copied code to {REPO_DIR}")
                code_copied = True
                break

# Option B: Clone from GitHub (Online mode)
if not code_copied:
    os.makedirs(KAGGLE_WORKING, exist_ok=True)
    os.chdir(KAGGLE_WORKING)

    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    
    print(f"Cloning from GitHub: {GITHUB_REPO} ...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", GITHUB_REPO, REPO_DIR], cwd=KAGGLE_WORKING, check=True)
        print(f"Successfully cloned to {REPO_DIR}")
        code_copied = True
    except subprocess.CalledProcessError as e:
        print("\n" + "!" * 70)
        print("ERROR: Git clone failed! Could not reach github.com.")
        print("!" * 70)
        print("\nWHY THIS HAPPENED:")
        print("  By default, Kaggle notebooks have INTERNET TURNED OFF.")
        print("\nHOW TO FIX THIS IN 10 SECONDS:")
        print("  1. Look at the right sidebar of your Kaggle notebook window.")
        print("  2. In the 'Notebook options' / 'Settings' panel, find 'Internet'.")
        print("  3. Toggle 'Internet' to ON (requires one-time SMS verification if not verified).")
        print("  4. Re-run this cell!")
        print("\nALTERNATIVE (OFFLINE MODE):")
        print("  If you cannot turn on Internet, create a Kaggle dataset containing")
        print("  the 'amazon_ml' repository files, add it to this notebook, and re-run.")
        print("!" * 70 + "\n")
        raise e

# =====================================================================
# STEP 2: Install dependencies
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Installing / verifying dependencies")
print("=" * 70)

req_file = os.path.join(REPO_DIR, "student_resource", "code", "business_entity_resolution", "requirements.txt")
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file], check=True)
    print("Dependencies installed successfully.")
except Exception as e:
    print(f"Warning: pip install encountered an issue: {e}")
    print("Checking if pre-installed packages in Kaggle are sufficient...")
    missing = []
    for pkg in ["lightgbm", "duckdb", "rapidfuzz", "anyascii", "jellyfish", "pandas", "numpy", "scipy"]:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} is available")
        except ImportError:
            missing.append(pkg)
            print(f"  ✗ {pkg} is MISSING")
    if missing:
        raise RuntimeError(f"Missing required packages: {missing}. Please ensure Internet is ON to install them.")

# =====================================================================
# STEP 3: Set up directory structure with symlinks
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: Setting up directory structure")
print("=" * 70)

resource_dir = os.path.join(REPO_DIR, "student_resource")

def find_file_in_kaggle(target_name):
    """Search recursively for target_name under /kaggle/input."""
    if not os.path.exists(KAGGLE_INPUT):
        return None
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        if target_name in files:
            return os.path.join(root, target_name)
    return None

# 3a. Create dataset/test/ directory and symlink test TSVs
test_dir = os.path.join(resource_dir, "dataset", "test")
os.makedirs(test_dir, exist_ok=True)

test_files = ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]
missing_test_files = []

for fname in test_files:
    found_path = find_file_in_kaggle(fname)
    dst = os.path.join(test_dir, fname)
    if found_path:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(found_path, dst)
        size_mb = os.path.getsize(found_path) / 1e6
        print(f"  ✓ Linked {fname} ({size_mb:.1f} MB) <- {found_path}")
    else:
        missing_test_files.append(fname)

if missing_test_files:
    print(f"\n❌ ERROR: Could not find required test file(s): {missing_test_files}")
    print(f"Files currently visible in {KAGGLE_INPUT}:")
    all_files = []
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        for f in files:
            all_files.append(os.path.join(root, f))
    if all_files:
        for f in all_files[:30]:
            print(f"  - {f}")
        if len(all_files) > 30:
            print(f"  ... and {len(all_files) - 30} more files")
    else:
        print("  (Directory is empty or no files attached)")
    sys.exit(1)

# 3b. Create output directory and copy partial results
output_dir = os.path.join(resource_dir, "output")
temp_dir = os.path.join(output_dir, "temp_partitions")
os.makedirs(temp_dir, exist_ok=True)

partial_files = ["match_France.tsv", "cand_France.tsv", "match_US.tsv", "cand_US.tsv"]
for fname in partial_files:
    found_path = find_file_in_kaggle(fname)
    dst = os.path.join(temp_dir, fname)
    if found_path:
        shutil.copy2(found_path, dst)
        size_mb = os.path.getsize(dst) / 1e6
        print(f"  ✓ Reusing partial {fname} ({size_mb:.1f} MB) <- {found_path}")
    else:
        country_part = fname.split('_')[1].split('.')[0]
        print(f"  ⚠ Note: {fname} not found in Kaggle input — {country_part} will be computed during inference")

# 3c. Create train dir (for config import compatibility, even if empty)
train_dir = os.path.join(resource_dir, "dataset", "train")
os.makedirs(train_dir, exist_ok=True)

# =====================================================================
# STEP 4: Verify setup
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: Verifying setup")
print("=" * 70)

project_dir = os.path.join(resource_dir, "code", "business_entity_resolution")
model_path = os.path.join(project_dir, "models", "lgbm_entity_resolver.txt")
print(f"  Project dir: {project_dir} (exists: {os.path.isdir(project_dir)})")
print(f"  Model file:  {model_path} (exists: {os.path.isfile(model_path)})")
print(f"  Test S1:     {os.path.join(test_dir, 'test_source1.tsv')} (exists: {os.path.isfile(os.path.join(test_dir, 'test_source1.tsv'))})")

for f in os.listdir(temp_dir):
    print(f"  Partial:     {f} ({os.path.getsize(os.path.join(temp_dir, f)) / 1e6:.1f} MB)")

# =====================================================================
# STEP 5: Run full inference
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: Running full inference pipeline")
print("=" * 70)

# Add project to sys.path and run
sys.path.insert(0, project_dir)
os.chdir(project_dir)

# Import and run
from src.inference import run_full_inference
from src.config import MATCHING_OUTPUT, CANDIDATE_OUTPUT

run_full_inference(
    matching_path=MATCHING_OUTPUT,
    candidate_path=CANDIDATE_OUTPUT,
    threshold=0.84,
)

# =====================================================================
# STEP 6: Validate submission
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: Running official submission validator")
print("=" * 70)

validator_script = os.path.join(resource_dir, "utils", "validate_submission.py")
if os.path.exists(validator_script):
    result = subprocess.run([
        sys.executable, validator_script,
        "--matching", MATCHING_OUTPUT,
        "--candidate", CANDIDATE_OUTPUT,
        "--test-dir", test_dir,
    ])
    if result.returncode == 0:
        print("\n✅ SUBMISSION VALIDATION PASSED! Files are safe to submit.")
    else:
        print("\n❌ Validation failed. Check errors above.")
else:
    print(f"Validator not found at {validator_script}")

# =====================================================================
# STEP 7: Copy outputs to /kaggle/working for download
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: Copying outputs for download")
print("=" * 70)

for fname, src_path in [
    ("matching_results.tsv", MATCHING_OUTPUT),
    ("candidate_pairs.tsv", CANDIDATE_OUTPUT),
]:
    dst = os.path.join(KAGGLE_WORKING, fname)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst)
        size_mb = os.path.getsize(dst) / 1e6
        with open(dst, "r") as f:
            lines = sum(1 for _ in f) - 1  # minus header
        print(f"  {fname}: {lines:,} rows, {size_mb:.1f} MB -> {dst}")

print("\n" + "=" * 70)
print("DONE! Download matching_results.tsv and candidate_pairs.tsv")
print("from the Output tab on the right sidebar.")
print("=" * 70)
