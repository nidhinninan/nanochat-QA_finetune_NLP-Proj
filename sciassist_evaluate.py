#!/usr/bin/env python3
"""
SciAssist Model Evaluation Script

This script is designed to run after `sciassist_train.py` to evaluate the
fine-tuned model. It performs the following actions:
1.  Parses training logs to plot training and validation loss curves.
2.  Loads the base and fine-tuned models.
3.  Runs a quantitative evaluation on the validation set to compare accuracy.
4.  Provides qualitative comparisons for specific examples.
5.  Allows for interactive testing with custom questions.

Usage:
    python sciassist_evaluate.py --log_file training.log
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import pandas as pd
from datasets import load_from_disk

# Plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    print("Please install matplotlib and seaborn to generate plots: pip install matplotlib seaborn")
    plt = None

# Ensure nanochat package is importable
PROJECT_ROOT = Path(__file__).resolve().parent
NANOCHAT_REPO = PROJECT_ROOT / "nanochat"
if NANOCHAT_REPO.exists():
    sys.path.insert(0, str(NANOCHAT_REPO))

try:
    from nanochat.common import autodetect_device_type, compute_init
    from nanochat.checkpoint_manager import build_model
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.gpt import GPT, GPTConfig
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "nanochat package not found. Please ensure it is installed or cloned at ./nanochat"
    ) from exc

# -----------------------------------------------------------------------------
# Configuration (from sciassist_train.py)
# -----------------------------------------------------------------------------

# Paths
DATA_DIR = Path("data")
VAL_DATA_PATH = DATA_DIR / "val_formatted"
OUTPUT_DIR = Path("finetuned_model_checkpoint")

# Model checkpoint location
BASE_CACHE = Path.home() / ".cache" / "nanochat"
CHECKPOINT_DIR = BASE_CACHE / "chatsft_checkpoints" / "d20"
CHECKPOINT_STEP = 650

# -----------------------------------------------------------------------------
# Utility Functions (adapted from sciassist_train.py)
# -----------------------------------------------------------------------------

def print_separator(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def load_finetuned_model_from_dir(model_dir, device):
    model_path = model_dir / "model_finetuned.pt"
    meta_path = model_dir / "meta_finetuned.json"
    if not model_path.exists() or not meta_path.exists():
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    config = GPTConfig(**meta["model_config"])
    model = GPT(config)
    state_dict = torch.load(model_path, map_location=device)
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, meta

def extract_choice_letter(text):
    patterns = [
        r"\b([A-F])\.",
        r"answer is ([A-F])\b",
        r"correct answer is ([A-F])\b",
        r"\b([A-F])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None

def generate_response(model, tokenizer, conversation, autocast_ctx, max_tokens=256, temperature=0.7, top_k=50):
    conversation_copy = json.loads(json.dumps(conversation))
    conversation_copy["messages"][-1]["content"] = ""
    prompt_tokens = tokenizer.render_for_completion(conversation_copy)
    tokens = list(prompt_tokens)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    generated = []
    with torch.no_grad():
        with autocast_ctx:
            for token in model.generate(tokens, max_tokens=max_tokens, temperature=temperature, top_k=top_k):
                tokens.append(token)
                generated.append(token)
                if token == assistant_end:
                    break
    decoded = tokenizer.decode([tok for tok in generated if tok != assistant_end])
    return decoded.strip()

def compute_accuracy(results, response_key):
    correct = 0
    total = 0
    for entry in results:
        expected_letter = extract_choice_letter(entry["expected"])
        predicted_letter = extract_choice_letter(entry.get(response_key, ""))
        if expected_letter and predicted_letter:
            total += 1
            if expected_letter == predicted_letter:
                correct += 1
    return (correct, total, correct / total if total else 0.0)

# -----------------------------------------------------------------------------
# New Evaluation and Plotting Functions
# -----------------------------------------------------------------------------

def parse_training_log(log_file):
    """Parses a log file to extract training and validation losses."""
    if not Path(log_file).exists():
        print(f"Warning: Log file '{log_file}' not found. Skipping loss plot.")
        return None

    train_loss_pattern = re.compile(r"Step (\d+)/\d+ \| Train loss: ([\d.]+)")
    val_loss_pattern = re.compile(r"Step (\d+) \| Val loss: ([\d.]+)")

    data = []
    with open(log_file, 'r') as f:
        for line in f:
            train_match = train_loss_pattern.search(line)
            if train_match:
                step, loss = train_match.groups()
                data.append({'step': int(step), 'loss': float(loss), 'type': 'Train'})

            val_match = val_loss_pattern.search(line)
            if val_match:
                step, loss = val_match.groups()
                data.append({'step': int(step), 'loss': float(loss), 'type': 'Validation'})

    if not data:
        print(f"Warning: No loss data found in '{log_file}'.")
        return None

    return pd.DataFrame(data)

def plot_loss_curves(df, output_dir):
    """Generates and saves a plot of training and validation loss."""
    if plt is None or df is None or df.empty:
        return

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='step', y='loss', hue='type', marker='o', markersize=5)
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend(title='Loss Type')
    plt.tight_layout()

    plot_path = output_dir / "training_validation_loss.png"
    plt.savefig(plot_path)
    print(f"✓ Loss plot saved to {plot_path}")
    plt.close()

def display_comparison(example):
    """Display side-by-side comparison of base and fine-tuned responses"""
    print("="*100)
    print(f"EXAMPLE {example['index']}")
    print("="*100)

    print("\n📝 QUESTION:")
    print("-" * 100)
    print(example['question'])

    print("\n✅ EXPECTED ANSWER:")
    print("-" * 100)
    print(example['expected'])

    print("\n🤖 BASE MODEL RESPONSE:")
    print("-" * 100)
    print(example['base_response'])

    print("\n🎯 FINE-TUNED MODEL RESPONSE:")
    print("-" * 100)
    print(example['finetuned_response'])
    print("\n" + "="*100 + "\n")

def test_single_question(question_text, base_model, ft_model, tokenizer, autocast_ctx, is_interactive=False):
    """Test a single question with both base and fine-tuned models."""
    test_conversation = {
        "messages": [
            {"role": "system", "content": "You are a helpful science tutor..."},
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": ""}
        ]
    }

    if is_interactive:
        print_separator(f"Testing Question: {question_text[:50]}...")

    # Base Model
    base_response = generate_response(base_model, tokenizer, test_conversation, autocast_ctx, temperature=0.0, top_k=1)
    if is_interactive:
        print("\n🤖 BASE MODEL RESPONSE:")
        print("-" * 50)
        print(base_response)

    # Fine-tuned Model
    ft_response = generate_response(ft_model, tokenizer, test_conversation, autocast_ctx, temperature=0.0, top_k=1)
    if is_interactive:
        print("\n🎯 FINE-TUNED MODEL RESPONSE:")
        print("-" * 50)
        print(ft_response)

# -----------------------------------------------------------------------------
# Main Evaluation Function
# -----------------------------------------------------------------------------

def main(args):
    print_separator("SciAssist Model Evaluation")

    # --- 1. Plot Loss Curves ---
    if args.log_file:
        loss_df = parse_training_log(args.log_file)
        if loss_df is not None:
            plot_loss_curves(loss_df, OUTPUT_DIR)

    # --- 2. Setup Environment & Load Data ---
    device_type = autodetect_device_type()
    forced_device = os.environ.get("SCIASSIST_FORCE_DEVICE", "").strip().lower()
    if forced_device in {"cuda", "cpu", "mps"}:
        print(f"Overriding device type to '{forced_device}' via SCIASSIST_FORCE_DEVICE")
        device_type = forced_device
    _, _, _, _, device = compute_init(device_type=device_type)
    print(f"Using device: {device}")

    if not VAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Validation data not found at {VAL_DATA_PATH}")
    val_dataset = load_from_disk(str(VAL_DATA_PATH))
    print(f"Loaded {len(val_dataset)} validation samples.")

    # --- 3. Load Models and Tokenizer ---
    print_separator("Loading Models")
    finetuned_model, _ = load_finetuned_model_from_dir(OUTPUT_DIR, device)
    if finetuned_model is None:
        raise FileNotFoundError(f"Fine-tuned model not found in {OUTPUT_DIR}. Please run training first.")

    base_model, tokenizer, _ = build_model(
        checkpoint_dir=str(CHECKPOINT_DIR), step=CHECKPOINT_STEP, device=device, phase="eval"
    )
    print("✓ Base and fine-tuned models loaded.")

    # --- 4. Quantitative Evaluation ---
    print_separator("Quantitative Evaluation")
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    evaluation_results = []
    num_eval_samples = min(args.eval_samples, len(val_dataset))

    for idx in range(num_eval_samples):
        example = val_dataset[idx]
        conversation = example["messages"]
        base_response = generate_response(base_model, tokenizer, {"messages": conversation}, autocast_ctx, temperature=0.0, top_k=1)
        finetuned_response = generate_response(finetuned_model, tokenizer, {"messages": conversation}, autocast_ctx, temperature=0.0, top_k=1)
        evaluation_results.append({
            "index": idx,
            "question": conversation[1]["content"],
            "expected": conversation[-1]["content"],
            "base_response": base_response,
            "finetuned_response": finetuned_response,
        })
        print(f"  Evaluated sample {idx + 1}/{num_eval_samples}", end='\r')

    base_correct, base_total, base_acc = compute_accuracy(evaluation_results, "base_response")
    ft_correct, ft_total, ft_acc = compute_accuracy(evaluation_results, "finetuned_response")

    print(f"\n\nBase model accuracy: {base_correct}/{base_total} ({base_acc:.2%})")
    print(f"Fine-tuned model accuracy: {ft_correct}/{ft_total} ({ft_acc:.2%})")

    # --- 5. Qualitative Comparison ---
    print_separator("Qualitative Comparison")
    for i in range(min(3, len(evaluation_results))):
        display_comparison(evaluation_results[i])

    # --- 6. Interactive Testing ---
    print_separator("Interactive Testing")
    sample_questions = [
        "What is photosynthesis?\nA. The process plants use to make food from sunlight\nB. The process of cell division\nC. The process of breathing\nD. The process of digestion",
        "What is the main cause of ocean tides?\nA. Wind blowing across the ocean surface\nB. The gravitational pull of the Moon\nC. Temperature differences in the water\nD. Earth's rotation",
        "Which state of matter has a definite volume but no definite shape?\nA. Solid\nB. Liquid\nC. Gas\nD. Plasma"
    ]
    for question in sample_questions:
        test_single_question(question, base_model, finetuned_model, tokenizer, autocast_ctx, is_interactive=True)

    print_separator("Evaluation Complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned SciAssist model.")
    parser.add_argument(
        "--log_file",
        type=str,
        default="log_SciAssist-FineTuning.txt",
        help="Path to the training log file to parse for loss curves."
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=100,
        help="Number of validation samples to use for accuracy evaluation."
    )
    args = parser.parse_args()
    main(args)