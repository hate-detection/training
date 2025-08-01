# Model Training and Evaluation

<div align="center">
    <a href="#overview">Overview</a> •
    <a href="#file-structure">File Structure</a> •
    <a href="#ml-algorithms-overview">ML Algorithms Overview</a> •
    <a href="#muril-fine-tuning">MuRIL Fine Tuning</a> •
    <a href="#muril-with-adapters">MuRIL with Adapters</a> •
    <a href="#results">Results</a>
</div>

## Overview

> [!IMPORTANT]
> 
> Train and Test data is not included with the repo due to Data Usage restrictions by the Dataset Authors.

This repository contains code for Model Training and Evaluation for the BOLI project. I train and evaluate 3 types of models, namely:
1. Classic ML Algorithms (SVM, Random Forest, XGBoost)
2. BERT-based model [MuRIL](https://huggingface.co/google/muril-base-cased) fine-tuning
3. MuRIL with Adapters fine-tuning

## File Structure:
```
training
|_
|_
|_ muril-fine-tuning.ipynb
|_ training-ml-algorithms.ipynb
|_ training-muril-w-adapters.ipynb
```
- [`muril-fine-tuning.ipynb`](https://github.com/hate-detection/training/blob/master/muril-fine-tuning.ipynb) contains code for MuRIL Fine-tuning.
- [`training-ml-algorithms.ipynb`](https://github.com/hate-detection/training/blob/master/training-ml-algorithms.ipynb) contains code for classic ML algorithms train-eval.
- [`training-muril-w-adapters.ipynb`](https://github.com/hate-detection/training/blob/master/training-muril-w-adapters.ipynb) fine-tunes MuRIL by injecting `LoRA` Adapters. 

## ML Algorithms Overview:
I train three algorithms: SVM, Random Forest, XGBoost. The Precision, Recall and F1 Score for all three are as follows:

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| SVM   | 70%       | 69%    | 66%      |
| Random Forest| 68%| 67%    | 63%      |
| XGBoost | 70%     | 69%    | 65%      |

SVM performs the best while Random Forest performs the worst. However, the Accuracy metrics for all three algorithms are disappointing.

## MuRIL Fine Tuning
The hyperparameters for full-model MuRIL Fine-tuning are:
```python
batch_size = 32
learning_rate = 2e-5
epochs = 4
```
The process:
- Use `google/muril-base-cased` from HuggingFace Transformers library along with `AutoModelForSequenceClassification`
- Divide the dataset into Train and Validation sets
- Perform a 90:10 split
- Tokenize data and pass it to HuggingFace `Trainer`
- Train the model with Task 1 Dataset
- Evaluate the trained model with Holdout Testing Set (Task 1)
- Train again with Train set (Task 2)
- Evaluate the trained model with Holdout Testing Set (Task 1 and Task 2)
- Note the performance

## MuRIL with Adapters
The hyperparameters for MuRIL with LoRA Adapters fine-tuning are:
```python
batch_size = 32
learning_rate = 2e-5
epochs = 20            # with early-stopping callback
```
The process:
- Use `google/muril-base-cased` from HuggingFace Transformers library along with `AutoModelForSequenceClassification`
- Inject `PEFT LoRA` adapters into the base model
- Divide the dataset (Task 1 and Task 2) into Train and Validation sets
- Perform a 90:10 split
- Tokenize data and pass it to HuggingFace `Trainer`
- Train the model with Adapter 1
- Evaluate the trained model with Holdout Testing Set (Task 1)
- Activate Adapter 2 and train again with Train set (Task 2)
- Evaluate the trained model with Holdout Testing Set (Task 1 and Task 2)
- Note the performance

## Results

The F1 Score for both full-model MuRIL fine-tuning and MuRIL with Adapters is as follows:

| Model | Task 1 (before continual learning) | Task 2 | Task 1 (after continual learning) |
|-------|-----------|--------|----------|
| MuRIL | 77.5%     | 82%    | 73%      |
| MuRIL (with adapters)| 74.5%    | 82% | 74% |

**Conclusion:** Although full-model Fine-tuning performs well for Task 1, its performance drops after continual learning. On the other hand, MuRIL (with adapters) performs well for Task 1 both before and after continual learning.
