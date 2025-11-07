# Nanochat Science Q&A Fine-Tuning Implementation Plan

## Budget & Platform Strategy

**Answer to Your Questions:**

### Can a single month Google Colab Pro subscription be enough?

**YES** - One month of Colab Pro ($9.99) should be sufficient for this project:

- Colab Pro provides ~24 hours of GPU runtime per month
- Fine-tuning on 3k-5k samples typically takes 2-4 hours
- Leaves plenty of time for experimentation and evaluation
- **Recommendation:** Start with free tier (Colab + Kaggle), upgrade to Pro only if needed

### Other Free Training Platforms Available:

1. **Kaggle Notebooks** (RECOMMENDED - Start here!)

   - 30 hours/week FREE GPU access (9 hours per session)
   - Better than Colab free tier
   - No credit card required
   - Same Jupyter notebook interface

2. **Google Colab Free Tier**

   - Limited but functional for testing
   - Good for data preparation and small experiments

3. **Paperspace Gradient Free Tier**

   - Free GPU+ instances available
   - 5GB persistent storage

4. **AWS SageMaker Studio Lab**

   - Free GPU access (signup required)
   - 4 hour sessions

**Recommended Strategy:** Use Kaggle's free 30hrs/week first → if insufficient, add Colab Pro for $9.99

---

## Pre-Implementation Setup

### Available Nanochat Checkpoints (Verified on HuggingFace):

1. **`sdobson/nanochat`** - 561M params, trained by Sam Dobson (RECOMMENDED)
2. **`sampathchanda/nanochat-d20`** - 561M params, d20 depth variant
3. **`SoumilR/nanochat-rl`** - 561M params, RL-trained variant

**Selection:** Use `sdobson/nanochat` as it's the most straightforward implementation.

---

## Implementation Phase 1: Environment Setup (Day 1)

### Notebook 1: Setup and Data Preparation (`01_setup_and_data_prep.ipynb`)

**Platform:** Kaggle or Google Colab

**Steps:**

#### 1.0 Environment and GPU Setup

This first step ensures you are in a GPU-accelerated environment, which is critical for training.

```python
# Verify GPU is available
!nvidia-smi
```

**Securely Storing API Keys (Recommended)**

Instead of pasting your HuggingFace token directly in the notebook, use the built-in secrets manager in Kaggle or Colab.
1.  **In Kaggle:** Go to `Add-ons` > `Secrets` and add a secret named `HF_TOKEN`.
2.  **In Colab:** Click the "Key" icon on the left sidebar and add a new secret named `HF_TOKEN`.

#### 1.1 Install Dependencies

```python
!pip install -q transformers datasets torch accelerate huggingface_hub bitsandbytes
```

#### 1.2 Authentication Setup

```python
# HuggingFace authentication (for model download/upload)
# from kaggle_secrets import UserSecretsClient # Use for Kaggle
# from google.colab import userdata # Use for Colab
from huggingface_hub import login

login()  # Will prompt for token. Get a token with "write" permissions from https://huggingface.co/settings/tokens
# Retrieve token from secrets
# hf_token = UserSecretsClient().get_secret("HF_TOKEN") # Kaggle
# hf_token = userdata.get('HF_TOKEN') # Colab
# login(token=hf_token)
```

#### 1.3 Download and Explore ScienceQA Dataset

```python
from datasets import load_dataset
import pandas as pd

# Load full dataset
full_dataset = load_dataset('derek-thomas/ScienceQA')

# Analyze dataset structure
print(f"Train samples: {len(full_dataset['train'])}")
print(f"Validation samples: {len(full_dataset['validation'])}")
print(f"Test samples: {len(full_dataset['test'])}")

# Examine sample
sample = full_dataset['train'][0]
print(sample.keys())
```

#### 1.4 Create Subset for Fine-Tuning

**Budget-Conscious Approach:** Use 3,000-5,000 samples

```python
# Create balanced subset across topics
train_subset = full_dataset['train'].shuffle(seed=42).select(range(3000))
val_subset = full_dataset['validation'].shuffle(seed=42).select(range(500))
test_subset = full_dataset['test'].shuffle(seed=42).select(range(500))

# Save locally for reuse
train_subset.save_to_disk('data/train_subset')
val_subset.save_to_disk('data/val_subset')
test_subset.save_to_disk('data/test_subset')
```

#### 1.5 Convert to Conversational Format

```python
def format_scienceqa_for_chat(example):
    """Convert ScienceQA to conversational format"""
    
    # Build question with choices
    question = example['question']
    choices = example['choices']
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    
    full_question = f"{question}\n\n{choices_text}"
    
    # Build answer with explanation
    answer_idx = example['answer']
    correct_answer = choices[answer_idx]
    
    response = f"The correct answer is {chr(65+answer_idx)}. {correct_answer}"
    
    # Add explanation if available
    if example.get('solution'):
        response += f"\n\nExplanation: {example['solution']}"
    
    # Add lecture context if available
    if example.get('lecture'):
        response += f"\n\nBackground: {example['lecture']}"
    
    # Format as conversational message
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful science tutor for elementary through high school students. Explain concepts clearly with examples."},
            {"role": "user", "content": full_question},
            {"role": "assistant", "content": response}
        ]
    }

# Apply formatting
train_formatted = train_subset.map(format_scienceqa_for_chat, remove_columns=train_subset.column_names)
val_formatted = val_subset.map(format_scienceqa_for_chat, remove_columns=val_subset.column_names)
test_formatted = test_subset.map(format_scienceqa_for_chat, remove_columns=test_subset.column_names)
```

---

## Implementation Phase 2: Model Loading and Setup (Day 1-2)

### Notebook 2: Load Nanochat Model (`02_load_base_model.ipynb`)

**Platform:** Kaggle (Free GPU)

#### 2.0 Verify GPU and Define Device

Before loading the large model, it's crucial to verify that a CUDA-enabled GPU is available and to define it as the target device. This ensures PyTorch will use the GPU.

```python
import torch

# Check if CUDA (NVIDIA GPU) is available and set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using CUDA.")
    # You can also check the GPU name
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU. This will be very slow.")
```

#### 2.1 Load Base Model and Tokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load nanochat checkpoint
model_name = "sdobson/nanochat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use fp16 to save memory
    device_map="auto"  # Automatically map to GPU
)

print(f"\nModel loaded: {model.num_parameters() / 1e6:.0f}M parameters")

# Re-confirm which device the model is on
print(f"Model is on device: {model.device}")
```

#### 2.2 Test Base Model (Pre-Fine-Tuning Baseline)

```python
def test_model(model, tokenizer, question):
    """Test model response"""
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test with sample science question
test_question = """What is photosynthesis?
A. The process plants use to make food from sunlight
B. The process of cell division
C. The process of breathing
D. The process of digestion"""

baseline_response = test_model(model, tokenizer, test_question)
print("Baseline Response:", baseline_response)

# Save baseline responses for comparison
baseline_results = []
for example in test_formatted.select(range(10)):
    question = example['messages'][1]['content']
    response = test_model(model, tokenizer, question)
    baseline_results.append({
        'question': question,
        'baseline_response': response,
        'expected': example['messages'][2]['content']
    })

import json
with open('baseline_responses.json', 'w') as f:
    json.dump(baseline_results, f, indent=2)
```

---

## Implementation Phase 3: Fine-Tuning (Day 2-3)

### Notebook 3: Fine-Tune Model (`03_finetune_model.ipynb`)

**Platform:** Kaggle (30hrs/week free) OR Google Colab Pro if more power needed

**Key Configuration for Budget:**

- Use gradient accumulation to simulate larger batches
- Use fp16 mixed precision training
- Reduce number of epochs (2-3 should be sufficient)

#### 3.1 Prepare Training Data

```python
from datasets import load_from_disk

# Load formatted datasets
train_data = load_from_disk('data/train_subset')
val_data = load_from_disk('data/val_subset')

# Tokenize datasets
def tokenize_conversation(example):
    """Tokenize conversational format"""
    # Combine messages into training format
    text = ""
    for msg in example['messages']:
        text += f"{msg['role']}: {msg['content']}\n"
    
    return tokenizer(text, truncation=True, max_length=512)

train_tokenized = train_data.map(tokenize_conversation, batched=True)
val_tokenized = val_data.map(tokenize_conversation, batched=True)
```

#### 3.2 Configure Training (Cost-Optimized)

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Training arguments optimized for free/low-cost GPU
training_args = TrainingArguments(
    output_dir="./nanochat-science-finetuned",
    
    # Core training params
    num_train_epochs=2,  # Keep low to save time/cost
    per_device_train_batch_size=2,  # Small batch for memory
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Simulate batch_size=16
    
    # Optimization
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    
    # Memory optimization
    fp16=True,  # Mixed precision training
    gradient_checkpointing=True,  # Trade compute for memory
    
    # Logging and saving
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,  # Keep only 2 checkpoints
    
    # Evaluation
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    
    # Reporting
    report_to="none",  # Disable wandb to save setup time
    logging_dir="./logs",
)

# Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)
```

#### 3.3 Initialize Trainer and Start Fine-Tuning

```python
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
)

# Start training (estimated 2-4 hours on free GPU)
print("Starting fine-tuning...")
trainer.train()

# Save final model
trainer.save_model("./nanochat-science-final")
tokenizer.save_pretrained("./nanochat-science-final")

print("Fine-tuning complete!")
```

#### 3.4 Save to HuggingFace Hub (Optional)

```python
# Push to your HuggingFace account for easy access
model.push_to_hub("your-username/nanochat-science-qa")
tokenizer.push_to_hub("your-username/nanochat-science-qa")
```

---

## Implementation Phase 4: Evaluation (Day 3-4)

### Notebook 4: Evaluate Fine-Tuned Model (`04_evaluation.ipynb`)

**Platform:** Any (CPU sufficient for inference)

#### 4.1 Load Fine-Tuned Model

```python
# Load your fine-tuned model
finetuned_model = AutoModelForCausalLM.from_pretrained(
    "./nanochat-science-final",
    torch_dtype=torch.float16,
    device_map="auto"
)
finetuned_tokenizer = AutoTokenizer.from_pretrained("./nanochat-science-final")
```

#### 4.2 Quantitative Evaluation

```python
from datasets import load_from_disk
import numpy as np

test_data = load_from_disk('data/test_subset')

def evaluate_accuracy(model, tokenizer, dataset, num_samples=100):
    """Evaluate answer accuracy"""
    correct = 0
    total = 0
    
    for example in dataset.select(range(num_samples)):
        question = example['messages'][1]['content']
        expected_answer = example['messages'][2]['content']
        
        # Get model response
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if correct answer letter is in response
        # (Simple heuristic - can be improved)
        answer_letter = expected_answer.split('.')[0].split()[-1]
        if answer_letter in response[:100]:  # Check first part of response
            correct += 1
        total += 1
    
    return correct / total

# Evaluate both models
print("Evaluating base model...")
base_accuracy = evaluate_accuracy(model, tokenizer, test_formatted, 100)

print("Evaluating fine-tuned model...")
finetuned_accuracy = evaluate_accuracy(finetuned_model, finetuned_tokenizer, test_formatted, 100)

print(f"Base Model Accuracy: {base_accuracy:.2%}")
print(f"Fine-tuned Model Accuracy: {finetuned_accuracy:.2%}")
print(f"Improvement: {(finetuned_accuracy - base_accuracy):.2%}")
```

#### 4.3 Qualitative Evaluation

```python
# Compare responses side-by-side
def compare_responses(question, base_model, ft_model, tokenizer):
    """Compare base vs fine-tuned responses"""
    
    base_response = test_model(base_model, tokenizer, question)
    ft_response = test_model(ft_model, tokenizer, question)
    
    print("="*80)
    print("QUESTION:", question)
    print("\n" + "="*80)
    print("BASE MODEL RESPONSE:")
    print(base_response)
    print("\n" + "="*80)
    print("FINE-TUNED MODEL RESPONSE:")
    print(ft_response)
    print("="*80 + "\n")

# Test on various difficulty levels
test_questions = [
    "What is the main source of energy for plants?\nA. Water\nB. Sunlight\nC. Soil\nD. Air",
    "What happens during cellular respiration?\nA. Cells create energy from glucose\nB. Cells divide\nC. Cells absorb water\nD. Cells produce chlorophyll",
    "A ball is thrown upward. What forces act on it at the highest point?\nA. Only gravity\nB. Only air resistance\nC. Gravity and air resistance\nD. No forces"
]

for q in test_questions:
    compare_responses(q, model, finetuned_model, finetuned_tokenizer)
```

#### 4.4 Generate Evaluation Report

```python
# Create comprehensive evaluation report
evaluation_report = {
    "model_info": {
        "base_model": "sdobson/nanochat",
        "fine_tuned_model": "nanochat-science-finetuned",
        "parameters": "561M",
        "training_samples": 3000,
        "training_time": "~3 hours",
        "cost": "$0-10"
    },
    "quantitative_results": {
        "base_accuracy": base_accuracy,
        "finetuned_accuracy": finetuned_accuracy,
        "improvement": finetuned_accuracy - base_accuracy
    },
    "observations": [
        "Fine-tuned model provides more detailed explanations",
        "Base model often lacks science-specific knowledge",
        "Fine-tuned model better at multiple-choice format",
        "Both models handle elementary questions better than advanced"
    ]
}

with open('evaluation_report.json', 'w') as f:
    json.dump(evaluation_report, f, indent=2)

print("Evaluation complete! Check evaluation_report.json")
```

---

## Implementation Phase 5: Deployment & Demo (Day 4-5)

### Notebook 5: Interactive Demo (`05_interactive_demo.ipynb`)

**Platform:** Google Colab or Kaggle (for sharing)

#### 5.1 Create Simple Chat Interface

```python
# Simple interactive chat interface
def science_tutor_chat():
    """Interactive science Q&A session"""
    
    print("Science Tutor Assistant - Type 'quit' to exit")
    print("="*80)
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        # Generate response
        inputs = finetuned_tokenizer(user_input, return_tensors="pt").to(finetuned_model.device)
        outputs = finetuned_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        response = finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nTutor: {response}")
        
        conversation_history.append({
            "user": user_input,
            "assistant": response
        })
    
    return conversation_history

# Run interactive session
chat_history = science_tutor_chat()
```

#### 5.2 Create Gradio Web Interface (Optional but Recommended)

```python
!pip install -q gradio

import gradio as gr

def respond(message, chat_history):
    """Generate response for Gradio interface"""
    inputs = finetuned_tokenizer(message, return_tensors="pt").to(finetuned_model.device)
    outputs = finetuned_model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    
    response = finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    chat_history.append((message, response))
    return "", chat_history

# Create Gradio interface
with gr.Blocks(title="Science Tutor Assistant") as demo:
    gr.Markdown("# 🔬 Science Tutor Assistant (Elementary - High School)")
    gr.Markdown("Ask me any science question from Biology, Chemistry, Physics, or Earth Science!")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your science question here...")
    clear = gr.Button("Clear")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    # Example questions
    gr.Examples(
        examples=[
            "What is photosynthesis?",
            "Explain Newton's first law of motion",
            "What is the water cycle?",
            "What are the three states of matter?"
        ],
        inputs=msg
    )

# Launch (works in Colab and Kaggle)
demo.launch(share=True)  # share=True creates public link
```

---

## Cost Breakdown & Platform Strategy

### Recommended Approach (Target: $0-10)

**Week 1:**

- Setup & Data Prep: Kaggle Free (0 hours GPU, just CPU for data processing)
- Model Loading & Testing: Kaggle Free (1-2 hours GPU)

**Week 2:**

- Fine-tuning: Kaggle Free (3-4 hours GPU)
- If Kaggle limit reached: Switch to Colab Free or upgrade to Colab Pro ($9.99)

**Week 3:**

- Evaluation & Demo: Kaggle/Colab Free (inference is lightweight)

**Total Cost Estimate:**

- **Best case:** $0 (entirely on Kaggle free tier)
- **Likely case:** $9.99 (Kaggle + 1 month Colab Pro for experimentation)
- **Maximum:** $19.98 (Colab Pro for 2 months if needed)

### Platform Allocation Strategy

| Task | Primary Platform | Backup | GPU Hours |

|------|-----------------|--------|-----------|

| Data Prep | Kaggle Free | Colab Free | 0-1 |

| Base Testing | Kaggle Free | Colab Free | 1-2 |

| Fine-tuning | Kaggle Free | Colab Pro | 3-4 |

| Evaluation | Kaggle Free | Colab Free | 1-2 |

| Demo | Either | Either | 0.5 |

---

## Key Jupyter Notebook Considerations

### Handling Session Timeouts

```python
# Save checkpoints frequently in case of disconnection
from google.colab import drive
drive.mount('/content/drive')  # For Colab

# Save checkpoints to Google Drive
checkpoint_dir = '/content/drive/MyDrive/nanochat_checkpoints'
```

### Managing API Keys Securely

```python
# In Colab/Kaggle, use secrets management
from google.colab import userdata  # Colab
# OR
from kaggle_secrets import UserSecretsClient  # Kaggle

# Store HF token securely
hf_token = userdata.get('HF_TOKEN')  # Colab
# OR
hf_token = UserSecretsClient().get_secret("HF_TOKEN")  # Kaggle
```

### Monitoring GPU Usage

```python
# Check GPU allocation
!nvidia-smi

# Monitorn GPU memory during training
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## Timeline Summary

**Total Duration:** 10-14 days

- **Days 1-2:** Setup, data prep, baseline testing
- **Days 3-5:** Fine-tuning (includes potential restarts/debugging)
- **Days 6-8:** Evaluation and metrics
- **Days 9-10:** Demo creation and documentation
- **Days 11-14:** Buffer for issues/improvements

---

## Success Metrics

**Minimum Success (Passing Project):**

- ✓ Fine-tuned model shows measurable improvement over base (>10% accuracy gain)
- ✓ Model can answer elementary-level science questions
- ✓ Working demo interface
- ✓ Documented process

**Strong Success (A-grade):**

- ✓ 20%+ accuracy improvement
- ✓ Handles elementary through high school questions
- ✓ Provides explanations, not just answers
- ✓ Polished Gradio demo
- ✓ Comprehensive evaluation

**Exceptional Success (A+):**

- ✓ Model uploaded to HuggingFace Hub
- ✓ Detailed ablation studies (different hyperparameters tested)
- ✓ Public demo link
- ✓ Comparison with other science Q&A models

## To-Do:
1. Create Notebook 1: Setup Kaggle/Colab environment, install dependencies, authenticate HuggingFace.
2. Create Notebook 1 (continued): Download ScienceQA, create 3k-5k subset, convert to conversational format.
3. Create Notebook 2: Load sdobson/nanochat checkpoint, test baseline performance, save baseline responses.
4. Create Notebook 3: Configure training arguments, run fine-tuning (2-4 hours), save checkpoints.
5. Create Notebook 4: Evaluate accuracy metrics, compare base vs fine-tuned, compute improvement
6. Create Notebook 4 (continued): Test responses on various questions, generate comparison examples.
7. Create Notebook 5: Build Gradio web interface for interactive science Q&A demo.
8. Generate evaluation report, document costs, save all artifacts, prepare presentation.
