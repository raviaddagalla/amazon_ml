#!/usr/bin/env python3
"""
Master Execution Pipeline for Business Entity Resolution
Amazon ML Challenge 2026

Usage:
    python run_pipeline.py [--train] [--infer] [--validate]
"""

import os
import sys
import argparse
import subprocess

# Ensure src package is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.config import (
    MODEL_PATH, MATCHING_OUTPUT, CANDIDATE_OUTPUT, TEST_DIR, RESOURCE_DIR,
    PROBABILITY_THRESHOLD
)
from src.train import train_matching_model
from src.inference import run_full_inference


def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--train", action="store_true", help="Retrain LightGBM matching model")
    parser.add_argument("--use-saved-pairs", action="store_true", help="Load pre-extracted training pairs from disk")
    parser.add_argument("--infer", action="store_true", help="Run inference on test set to generate outputs")
    parser.add_argument("--validate", action="store_true", help="Run official validate_submission.py script")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline (Train -> Infer -> Validate)")
    parser.add_argument("--overwrite", action="store_true", help="Force recomputing test partitions (ignore old partition caches)")
    parser.add_argument("--threshold", type=float, default=PROBABILITY_THRESHOLD, help="Probability threshold for matching")
    parser.add_argument("--countries", nargs="+", default=None, help="Filter inference to specific countries")
    args = parser.parse_args()

    # If no flags passed, default to all
    if not (args.train or args.infer or args.validate or args.all):
        args.all = True

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    # 1. Train Model (if requested)
    if args.train or (args.all and not os.path.exists(MODEL_PATH)):
        print("\n>>> STEP 1: TRAINING MODEL", flush=True)
        train_matching_model(use_saved_pairs=args.use_saved_pairs)
    elif os.path.exists(MODEL_PATH):
        print(f"\n[INFO] Found existing pre-trained model at: {MODEL_PATH}", flush=True)

    # 2. Run Inference
    if args.infer or args.all:
        print("\n>>> STEP 2: RUNNING STREAMING INFERENCE ON TEST SET")
        run_full_inference(
            countries_to_process=args.countries,
            threshold=args.threshold,
            overwrite=args.overwrite,
        )

    # 3. Validate Submission Outputs
    if args.validate or args.all:
        print("\n>>> STEP 3: RUNNING SUBMISSION VALIDATOR")
        validator_script = os.path.join(RESOURCE_DIR, "utils", "validate_submission.py")
        if os.path.exists(validator_script):
            cmd = [
                sys.executable,
                validator_script,
                "--matching", MATCHING_OUTPUT,
                "--candidate", CANDIDATE_OUTPUT,
                "--test-dir", TEST_DIR
            ]
            print(f"Running: {' '.join(cmd)}")
            res = subprocess.run(cmd)
            if res.returncode == 0:
                print("\n[SUCCESS] Submission files PASSED all validation checks! Safe to submit.")
            else:
                print("\n[ERROR] Submission validation failed. Please check the errors above.")
                sys.exit(res.returncode)
        else:
            print(f"[WARNING] Validator script not found at {validator_script}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
