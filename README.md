# Fine-tuning LLMs for unit test generation

This repository contains the scripts for fine-tuning a Phi-3-mini model for unit test generation.

## Hyperparameter Tuning

Hyperparameter tuning is performed in `hyperparameter_search.ipynb`. The process involves:

- Testing various combinations of LoRA hyperparameters:
  - LoRA alpha values: 16, 28, 32
  - LoRA rank values: 16, 28, 32
- Evaluating different learning rates: 1e-4, 2e-4, 5e-4
- Running for 300 steps (equivalent to 4,800 samples) for each configuration
- Determining the optimal combination based on the mean loss of the last few steps

## Model Training

The actual model training is implemented in `phi3_tinetuning.ipynb`. Key aspects include:

- Using the Unsloth library for efficient QLoRA fine-tuning
- Training on 56,813 samples from the synthetic pytest dataset
- Applying the optimal hyperparameters found during tuning:
  - LoRA rank: 28
  - LoRA alpha: 28
  - Learning rate: 2e-4
- Using AdamW optimizer with 8-bit precision
- Implementing a cosine decay learning rate scheduler
- Training for 2 epochs (7,102 steps)

The fine-tuned model achieved a 58% relative improvement in test generation quality compared to the base model.