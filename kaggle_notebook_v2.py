#!/usr/bin/env python3
"""
===================================================================
KAGGLE NOTEBOOK v2: Tiered Deterministic-First Entity Resolution
===================================================================

Runs validation and full test inference for Amazon ML Challenge 2026
using the new src_v2 tiered cascade architecture.
"""

import os
import subprocess
import sys
import shutil

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

GITHUB_REPO = "https://github.com/raviaddagalla/amazon_ml.git"
REPO_DIR = os.path.join(KAGGLE_WORKING, "amazon_ml")

# =====================================================================
# STEP 1: Obtain the repository code
# =====================================================================
print("=" * 70)
print("STEP 1: Obtaining repository code")
print("=" * 70)

# Reset Python's working directory to /kaggle/working to avoid orphaned directory errors on re-runs
os.makedirs(KAGGLE_WORKING, exist_ok=True)
os.chdir(KAGGLE_WORKING)

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

print(f"Cloning latest code from GitHub: {GITHUB_REPO} ...")
subprocess.run(["git", "clone", "--depth", "1", GITHUB_REPO, REPO_DIR], cwd=KAGGLE_WORKING, check=True)
print(f"Successfully cloned to {REPO_DIR}")

# =====================================================================
# STEP 2: Install dependencies
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Installing dependencies")
print("=" * 70)

req_file = os.path.join(REPO_DIR, "student_resource", "code", "business_entity_resolution", "requirements.txt")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file], check=True)
print("Dependencies installed successfully.")

# =====================================================================
# STEP 3: Setup test dataset symlinks
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: Linking dataset files")
print("=" * 70)

resource_dir = os.path.join(REPO_DIR, "student_resource")
project_dir = os.path.join(resource_dir, "code", "business_entity_resolution")
test_dir = os.path.join(resource_dir, "dataset", "test")
os.makedirs(test_dir, exist_ok=True)

def find_file_in_kaggle(target_name):
    if not os.path.exists(KAGGLE_INPUT):
        return None
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        if target_name in files:
            return os.path.join(root, target_name)
    return None

for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
    found = find_file_in_kaggle(fname)
    dst = os.path.join(test_dir, fname)
    if found:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(found, dst)
        print(f"  ✓ Linked {fname} <- {found}")
    else:
        print(f"  ❌ Missing required test file: {fname}")
        sys.exit(1)

# Ensure train dir exists
os.makedirs(os.path.join(resource_dir, "dataset", "train"), exist_ok=True)

# Add project directory to sys.path
sys.path.insert(0, project_dir)
os.chdir(project_dir)

# =====================================================================
# STEP 4: Run full test inference with Tiered Cascade (src_v2)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: Running Tiered Cascade Inference (v2)")
print("=" * 70)

# Purge any cached in-memory modules from previous notebook runs
for mod in list(sys.modules.keys()):
    if mod.startswith("src") or mod.startswith("src_v2"):
        del sys.modules[mod]

# Execute infer in a fresh subprocess so it ALWAYS loads the newly cloned code from disk
subprocess.run([
    sys.executable, "-m", "src_v2.infer", "--overwrite"
], cwd=project_dir, check=True)

# Output paths
matching_output = os.path.join(resource_dir, "output", "matching_results.tsv")
candidate_output = os.path.join(resource_dir, "output", "candidate_pairs.tsv")

# =====================================================================
# STEP 5: Validate submission
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: Running official submission validator")
print("=" * 70)

validator_script = os.path.join(resource_dir, "utils", "validate_submission.py")
if os.path.exists(validator_script):
    result = subprocess.run([
        sys.executable, validator_script,
        "--matching", matching_output,
        "--candidate", candidate_output,
        "--test-dir", test_dir,
    ], cwd=project_dir)
    if result.returncode == 0:
        print("\n✅ SUBMISSION VALIDATION PASSED!")
    else:
        print("\n❌ Validation warning/failure. Check output above.")

# =====================================================================
# STEP 6: Copy outputs to /kaggle/working for download
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: Copying outputs for download")
print("=" * 70)

for fname, src_path in [
    ("matching_results.tsv", matching_output),
    ("candidate_pairs.tsv", candidate_output),
]:
    dst = os.path.join(KAGGLE_WORKING, fname)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst)
        size_mb = os.path.getsize(dst) / 1e6
        with open(dst, "r") as f:
            lines = sum(1 for _ in f) - 1
        print(f"  ✓ {fname}: {lines:,} rows, {size_mb:.1f} MB -> {dst}")

print("\n" + "=" * 70)
print("COMPLETE! Download matching_results.tsv from the Output tab.")
print("=" * 70)
