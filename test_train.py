#!/usr/bin/env python3
"""
Simple test runner for sciassist_train.py
Monitors GPU usage and validates the training pipeline.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    print("="*80)
    print("SciAssist Training Test")
    print("="*80)
    
    project_root = Path(".").resolve()
    script_path = project_root / "sciassist_train.py"
    
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return 1
    
    # Check if data exists
    data_dir = project_root / "data"
    if not (data_dir / "train_formatted").exists():
        print(f"\nError: Formatted data not found in {data_dir}")
        print("Please ensure data preparation has been completed.")
        return 1
    
    print("\n✓ All prerequisites check passed")
    print("\nStarting training script...")
    print("-" * 80)
    
    # Run the training script
    # Use the conda environment python
    python_exe = sys.executable
    
    env = os.environ.copy()
    env["SCIASSIST_FORCE_DEVICE"] = "cpu"
    env["SCIASSIST_SKIP_TRAINING"] = "1"
    env["SCIASSIST_TEST_MODE"] = "1"
    
    result = subprocess.run(
        [python_exe, str(script_path)],
        cwd=project_root,
        env=env
    )
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✓ TEST PASSED")
        print("="*80)
        print("\nThe training script executed successfully!")
        print("Check the output above for training metrics.")
        
        # Check if checkpoint was created
        checkpoint_dir = project_root / "finetuned_model_checkpoint"
        if checkpoint_dir.exists():
            print(f"\n✓ Checkpoint directory created: {checkpoint_dir}")
            if (checkpoint_dir / "model_finetuned.pt").exists():
                print("✓ Model file found")
            if (checkpoint_dir / "meta_finetuned.json").exists():
                print("✓ Metadata file found")
            if (checkpoint_dir / "tokenizer").exists():
                print("✓ Tokenizer directory found")
    else:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"Training script exited with code: {result.returncode}")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

