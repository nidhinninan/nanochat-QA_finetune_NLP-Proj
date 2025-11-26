# Fine-Tuning Nanochat for Science Question Answering

This project fine-tunes Andrej Karpathy's `nanochat` model to create a specialized Science Teaching Assistant. The goal is to adapt a general-purpose conversational model to accurately answer multiple-choice science questions and provide helpful explanations for students from elementary to high school levels.

The entire fine-tuning process was conducted within a Google Colab environment, leveraging a single NVIDIA A100 GPU.

## Key Features

- **Base Model**: Utilizes a 561M parameter GPT-style model from the `nanochat` project.
- **Dataset**: Fine-tuned on a curated subset of the **ScienceQA** dataset from Hugging Face.
- **Task**: Supervised Fine-Tuning (SFT) for a question-answering and explanation task.
- **Performance**: The fine-tuned model shows improved accuracy and better adherence to the desired conversational format compared to the base model.

---

## Model Architecture

The model is based on the `nanochat` GPT architecture, which incorporates several modern features for efficiency and performance:

- **Rotary Position Embeddings (RoPE)**: For effective relative position encoding, allowing better generalization to different sequence lengths.
- **RMSNorm**: A simplified and efficient normalization layer used pre-normalization.
- **ReLU² Activation**: The MLP uses a squared ReLU activation function.
- **QK Norm**: Normalization is applied to query and key vectors before the attention mechanism to enhance training stability.
- **Dual Optimizer Strategy**:
  - **AdamW**: Used for the token embedding and final language model head layers.
  - **Muon**: A custom optimizer used for the main transformer block matrices.

The specific configuration for this model includes:
- **Layers**: 20
- **Embedding Dimension**: 1280
- **Attention Heads**: 10
- **Vocabulary Size**: 65,536

---

## Data Pipeline

The data pipeline transforms the raw ScienceQA dataset into a conversational format suitable for fine-tuning.

1.  **Dataset Source**: The `derek-thomas/ScienceQA` dataset is loaded from the Hugging Face Hub.

2.  **Subsetting**: To manage computational resources, the dataset is downsampled to:
    - **Training Set**: 12,000 examples
    - **Validation Set**: 4,000 examples
    - **Test Set**: 4,000 examples

3.  **Formatting**: Each sample is converted into a structured conversational format with `system`, `user`, and `assistant` roles. The `format_scienceqa_for_chat` function handles this by:
    - Setting a system prompt to define the "Science Tutor" persona.
    - Structuring the user's message to include the question and multiple-choice options.
    - Crafting the assistant's response to include the correct answer, a detailed explanation, and relevant background context from the dataset.

4.  **Tokenization**:
    - The `RustBPETokenizer` from `nanochat` is used.
    - A masking strategy is employed during tokenization to ensure that the model's loss is only calculated on the assistant's responses. User prompts and system messages are masked out.

---

## Training Process

The model was fine-tuned using the `sciassist_train.py` script, orchestrated from the main Colab notebook.

- **Hardware**: Single NVIDIA A100-SXM4-40GB GPU on Google Colab.
- **Framework**: PyTorch and the `nanochat` library.
- **Training Time**: Approximately 15 minutes for 748 steps.

### Key Hyperparameters

- **Effective Batch Size**: 64 (achieved with a device batch size of 8 and 8 gradient accumulation steps).
- **Epochs**: 4
- **Learning Rate Scheduler**: Linear decay.
- **Optimizers**:
  - **AdamW** for embedding layers (LR: `1e-1` for embeddings, `2e-3` for unembedding).
  - **Muon** for transformer blocks (LR: `1e-2`).

---

## Evaluation

The fine-tuned model was evaluated against the base model on both quantitative and qualitative measures.

### Quantitative Results

- **Accuracy**: The fine-tuned model achieved an accuracy of **42.0%** on a subset of the validation set, compared to the base model's **38.2%**.
- **Validation Loss**: The validation loss was significantly reduced from an initial **1.967** to a best of **0.669**, indicating successful learning.

### Training & Validation Loss Curve

The plot below shows the training and validation loss over the course of the fine-tuning process. The best validation loss was achieved at step 625.

! [Training and Validation Loss Curve] (./training_validation_loss_sample_v2.png)

### Qualitative Analysis

- **Format Adherence**: The fine-tuned model consistently follows the desired response format (e.g., "The correct answer is...").
- **Explanation Capability**: While the model attempts to provide explanations, the factual accuracy of these explanations can vary and sometimes includes hallucinations.
- **Answer Selection**: The model shows improved capability in selecting the correct multiple-choice answer compared to the base model.

---

## How to Reproduce

1.  **Environment Setup**:
    - Clone this repository and the `nanochat` submodule:
      ```bash
      git clone --recurse-submodules https://github.com/nidhinninan/nanochat-QA_finetune_NLP-Proj.git
      cd nanochat-QA_finetune_NLP-Proj
      ```
    - Install the necessary dependencies. It is recommended to use a virtual environment.
      ```bash
      pip install -e ./nanochat
      ```

2.  **Run the Notebook**:
    - Open and run the `sciassist-fine-tune-model-NEW_v7_single_colab.ipynb` notebook in a GPU-enabled environment (like Google Colab or Kaggle).
    - The notebook will handle data downloading, preprocessing, training, and evaluation.

3.  **Run Scripts Directly**:
    - First, run the data preparation cells in the notebook to create the formatted datasets.
    - Then, execute the training script:
      ```bash
      python sciassist_train.py
      ```
    - After training, run the evaluation script:
      ```bash
      python sciassist_evaluate.py --log_file <path_to_log_file>
      ```

## File Structure

- `sciassist-fine-tune-model-NEW_v7_single_colab.ipynb`: The main notebook for orchestrating the entire pipeline.
- `sciassist_train.py`: The core script for fine-tuning the model.
- `sciassist_evaluate.py`: Script for evaluating the model and generating performance plots.
- `nanochat/`: Submodule containing the `nanochat` library source code.
- `data/`: Directory where formatted datasets are stored.
- `finetuned_model_checkpoint/`: Directory where the final model, tokenizer, and evaluation results are saved.

---

## Acknowledgements

This project is built upon the excellent `nanochat` repository by Andrej Karpathy.
