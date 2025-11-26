#!/usr/bin/env python3
"""
SciAssist Fine-Tuning Script
Refactored from sciassist-fine-tune-model-v5-single_kaggle.ipynb
Uses nanochat's native functions for model loading, training, and checkpointing.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import gc
import math
from datasets import load_from_disk

# Ensure nanochat package is importable (either installed or local clone)
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
        "nanochat package not found. Please ensure it is installed in the current "
        "environment or cloned at ./nanochat"
    ) from exc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Test mode: Run only a few steps then create mock checkpoint
TEST_MODE = os.environ.get("SCIASSIST_TEST_MODE", "1") not in {"0", "false", "False"}
SKIP_TRAINING = os.environ.get("SCIASSIST_SKIP_TRAINING", "0") in {"1", "true", "True"}

# Paths
DATA_DIR = Path("data")
TRAIN_DATA_PATH = DATA_DIR / "train_formatted"
VAL_DATA_PATH = DATA_DIR / "val_formatted"
OUTPUT_DIR = Path("finetuned_model_checkpoint")

# Model checkpoint location (using nanochat's cache)
BASE_CACHE = Path.home() / ".cache" / "nanochat"
CHECKPOINT_DIR = BASE_CACHE / "chatsft_checkpoints" / "d20"
CHECKPOINT_STEP = 650

# Training hyperparameters (from notebook Cell 33)
DEVICE_BATCH_SIZE = 1
TARGET_EXAMPLES_PER_STEP = 64
NUM_EPOCHS = 2
UNEMBEDDING_LR = 2e-3
EMBEDDING_LR = 1e-1
MATRIX_LR = 1e-2
WEIGHT_DECAY = 0.0
INIT_LR_FRAC = 0.02
EVAL_EVERY = 25
EVAL_STEPS = 25

# Test mode settings
TEST_NUM_STEPS = 3  # Only train for this many steps in test mode

# -----------------------------------------------------------------------------
# Data Generator (from notebook Cell 27)
# -----------------------------------------------------------------------------

def sft_data_generator(dataset, tokenizer, batch_size, device):
    """Data generator adapted from nanochat/scripts/chat_sft.py"""
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets
    
    batch = []
    while True:
        for i in range(len(dataset)):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------

def get_lr_multiplier(step, num_iterations):
    """Linear decay from notebook Cell 35"""
    return 1.0 - step / num_iterations


def copy_pretrained_checkpoint_to_output():
    """Copy base checkpoint files into OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    src_model = CHECKPOINT_DIR / f"model_{CHECKPOINT_STEP:06d}.pt"
    src_meta = CHECKPOINT_DIR / f"meta_{CHECKPOINT_STEP:06d}.json"
    if not src_model.exists() or not src_meta.exists():
        raise FileNotFoundError("Base checkpoint files not found. Please download them first.")
    shutil.copyfile(src_model, OUTPUT_DIR / "model_finetuned.pt")
    shutil.copyfile(src_meta, OUTPUT_DIR / "meta_finetuned.json")


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
    import re
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
    conversation_copy = json.loads(json.dumps({"messages": conversation}))
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


def run_evaluation_pipeline(tokenizer, val_dataset, device):
    eval_samples = 5 if device.type == "cuda" else 1
    print_separator("Evaluation")
    print(f"Preparing evaluation on {eval_samples} sample(s) using device: {device}")
    finetuned_model, _ = load_finetuned_model_from_dir(OUTPUT_DIR, device)
    if finetuned_model is None:
        print("No finetuned checkpoint found. Skipping evaluation.")
        return
    base_model_eval, _, _ = build_model(
        checkpoint_dir=str(CHECKPOINT_DIR),
        step=CHECKPOINT_STEP,
        device=device,
        phase="eval",
    )
    eval_autocast = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    results = []
    for idx in range(min(eval_samples, len(val_dataset))):
        example = val_dataset[idx]
        conversation = example["messages"]
        expected = conversation[-1]["content"]
        base_response = generate_response(base_model_eval, tokenizer, conversation, eval_autocast, max_tokens=128, temperature=0.0, top_k=10)
        finetuned_response = generate_response(finetuned_model, tokenizer, conversation, eval_autocast, max_tokens=128, temperature=0.0, top_k=10)
        results.append({
            "index": idx,
            "question": conversation[1]["content"] if len(conversation) > 1 else "",
            "expected": expected,
            "base_response": base_response,
            "finetuned_response": finetuned_response,
        })
    base_correct, base_total, base_acc = compute_accuracy(results, "base_response")
    ft_correct, ft_total, ft_acc = compute_accuracy(results, "finetuned_response")
    print(f"\nBase accuracy: {base_correct}/{base_total} ({base_acc:.2%})")
    print(f"Finetuned accuracy: {ft_correct}/{ft_total} ({ft_acc:.2%})")
    summary = {
        "samples": len(results),
        "base": {"correct": base_correct, "total": base_total, "accuracy": base_acc},
        "finetuned": {"correct": ft_correct, "total": ft_total, "accuracy": ft_acc},
    }
    with open(OUTPUT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if results:
        sample = results[0]
        print("\nSample comparison:")
        print("- Question:", sample["question"])
        print("- Expected:", sample["expected"])
        print("- Base response:", sample["base_response"])
        print("- Finetuned response:", sample["finetuned_response"])

# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------

def main():
    print("="*80)
    print("SciAssist Fine-Tuning")
    print("="*80)
    
    # Check data exists
    if not TRAIN_DATA_PATH.exists() or not VAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Formatted data not found at {DATA_DIR}. "
            "Please run data preparation first."
        )
    
    # Setup compute (from notebook Cell 8)
    device_type = autodetect_device_type()
    forced_device = os.environ.get("SCIASSIST_FORCE_DEVICE", "").strip().lower()
    if forced_device in {"cuda", "cpu", "mps"}:
        print(f"Overriding device type to '{forced_device}' via SCIASSIST_FORCE_DEVICE")
        device_type = forced_device
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type=device_type)
    master_process = (ddp_rank == 0)
    
    print(f"\nDevice: {device}")
    print(f"World size: {ddp_world_size}")
    
    if TEST_MODE:
        print(f"\n⚠️  TEST MODE ENABLED - Will run only {TEST_NUM_STEPS} training steps")
        print("   Set TEST_MODE = False in script for full training\n")
    
    # Load model and tokenizer (from notebook Cell 8)
    print("\nLoading base model...")
    model, tokenizer, meta = build_model(
        checkpoint_dir=str(CHECKPOINT_DIR),
        step=CHECKPOINT_STEP,
        device=device,
        phase="train",
    )
    base_model = model  # Keep reference for optimizer setup
    
    print(f"Model loaded: {meta['model_config']['n_layer']} layers, "
          f"~{sum(p.numel() for p in base_model.parameters()) / 1e6:.0f}M parameters")
    
    # Load datasets (from notebook Cell 32)
    print("\nLoading datasets...")
    train_dataset = load_from_disk(str(TRAIN_DATA_PATH))
    val_dataset = load_from_disk(str(VAL_DATA_PATH))
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Setup training (from notebook Cell 33)
    grad_accum_steps = TARGET_EXAMPLES_PER_STEP // DEVICE_BATCH_SIZE
    num_iterations = (len(train_dataset) // (TARGET_EXAMPLES_PER_STEP * ddp_world_size)) * NUM_EPOCHS
    
    if TEST_MODE:
        num_iterations = TEST_NUM_STEPS
        
    print(f"\nTraining configuration:")
    print(f"  Batch size (per device): {DEVICE_BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")
    print(f"  Total iterations: {num_iterations}")
    
    # Setup data loaders
    train_loader = sft_data_generator(train_dataset, tokenizer, DEVICE_BATCH_SIZE, device)
    val_loader = sft_data_generator(val_dataset, tokenizer, DEVICE_BATCH_SIZE, device)
    
    # Setup optimizers (from notebook Cell 34)
    print("\nInitializing optimizers...")
    optimizers = base_model.setup_optimizers(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
    
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * INIT_LR_FRAC
            group["initial_lr"] = group["lr"]
    
    # Setup autocast context (from notebook Cell 35)
    ptdtype = torch.bfloat16  # Notebook uses bfloat16, autocasts as needed
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) if device.type == "cuda" else nullcontext()
    
    # Training preparation
    OUTPUT_DIR.mkdir(exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = OUTPUT_DIR / "model_finetuned.pt"
    best_meta_path = OUTPUT_DIR / "meta_finetuned.json"
    
    # Copy tokenizer (from notebook Cell 35)
    if master_process:
        tokenizer_dest = OUTPUT_DIR / "tokenizer"
        if tokenizer_dest.exists():
            shutil.rmtree(tokenizer_dest)
        shutil.copytree(BASE_CACHE / "tokenizer", tokenizer_dest)
        print(f"✓ Tokenizer copied to {tokenizer_dest}")
    
    # Clear cache before training
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    training_performed = False
    
    if SKIP_TRAINING:
        if master_process:
            print("\nSkipping training cycle (SCIASSIST_SKIP_TRAINING=1).")
            copy_pretrained_checkpoint_to_output()
    else:
        print("\nStarting training...")
        print("-" * 80)
        
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        training_performed = True
        
        # Training loop (from notebook Cell 35)
        for step in range(num_iterations):
            t0 = time.time()
            last_step = (step == num_iterations - 1)
            
            # Validation
            if last_step or step % EVAL_EVERY == 0:
                model.eval()
                losses = []
                for _ in range(min(EVAL_STEPS, 5 if TEST_MODE else EVAL_STEPS)):
                    val_inputs, val_targets = next(val_iter)
                    with torch.no_grad(), autocast_ctx:
                        loss = model(val_inputs, val_targets)
                    losses.append(loss)
                    del val_inputs, val_targets
                
                val_loss = torch.stack(losses).mean().item()
                
                if ddp:
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                
                if master_process:
                    print(f"Step {step:05d} | Val loss: {val_loss:.6f}")
                    
                    # Save best checkpoint
                    if val_loss < best_val_loss:
                        print(f"  → New best! (was {best_val_loss:.6f})")
                        best_val_loss = val_loss
                        
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), best_model_path)
                        
                        best_meta = meta.copy()
                        best_meta['best_step'] = step
                        best_meta['best_val_loss'] = val_loss
                        best_meta['training_info'] = {
                            'num_train_samples': len(train_dataset),
                            'num_val_samples': len(val_dataset),
                            'num_epochs': NUM_EPOCHS,
                            'effective_batch_size': TARGET_EXAMPLES_PER_STEP * ddp_world_size,
                        }
                        
                        with open(best_meta_path, "w") as f:
                            json.dump(best_meta, f, indent=2)
                
                model.train()
            
            # Training step
            num_tokens = 0
            for micro_step in range(grad_accum_steps):
                train_inputs, train_targets = next(train_iter)
                with autocast_ctx:
                    loss = model(train_inputs, train_targets)
                
                train_loss = loss.detach()
                loss = loss / grad_accum_steps
                loss.backward()
                num_tokens += (train_targets >= 0).sum()
                
                del train_inputs, train_targets, loss
            
            # Update learning rate
            lrm = get_lr_multiplier(step, num_iterations)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lrm
            
            # Optimizer step
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            
            # Logging
            if master_process:
                t1 = time.time()
                dt = t1 - t0
                tokens_per_sec = num_tokens.item() / dt
                print(f"Step {step:05d}/{num_iterations:05d} | "
                      f"Train loss: {train_loss.item():.6f} | "
                      f"LR: {lrm:.4f} | "
                      f"Tok/s: {tokens_per_sec:.0f} | "
                      f"Time: {dt:.2f}s")
            
            del train_loss
        
        # Post-training: Create mock checkpoint if in test mode
        if TEST_MODE and master_process:
            print("\n" + "="*80)
            print("TEST MODE: Creating mock fine-tuned checkpoint from base model")
            print("="*80)
            
            mock_dir = OUTPUT_DIR.parent / "mock_finetuned_checkpoint"
            if mock_dir.exists():
                shutil.rmtree(mock_dir)
            
            # Copy the base model checkpoint
            shutil.copytree(CHECKPOINT_DIR, mock_dir)
            print(f"✓ Mock checkpoint created at: {mock_dir}")
            print("  (This is just the base model for testing evaluation pipeline)")
            
            # Also use it as our "fine-tuned" model for rest of script
            shutil.copy(mock_dir / f"model_{CHECKPOINT_STEP:06d}.pt", OUTPUT_DIR / "model_finetuned.pt")
            shutil.copy(mock_dir / f"meta_{CHECKPOINT_STEP:06d}.json", OUTPUT_DIR / "meta_finetuned.json")
            print(f"✓ Copied to {OUTPUT_DIR} for evaluation")
    
    if SKIP_TRAINING and master_process:
        mock_dir = OUTPUT_DIR.parent / "mock_finetuned_checkpoint"
        if mock_dir.exists():
            shutil.rmtree(mock_dir)
        shutil.copytree(CHECKPOINT_DIR, mock_dir)
        print(f"✓ Mock checkpoint ready at: {mock_dir}")
    
    if master_process:
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        val_display = f"{best_val_loss:.6f}" if training_performed and not math.isinf(best_val_loss) else "N/A"
        print(f"Best model saved to: {OUTPUT_DIR / 'model_finetuned.pt'}")
        print(f"Best validation loss: {val_display}")
    
    run_evaluation_pipeline(tokenizer, val_dataset, device)

if __name__ == "__main__":
    main()

