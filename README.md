# MuRIL Fine Tuning

Fine tuning BERT-based model [MuRIL](https://huggingface.co/google/muril-base-cased) for Sentiment Analysis in code-mixed Hinglish (Hindi-English).

> [!IMPORTANT]
> Train and Test data is not included with the repo due to Data Usage restrictions by the Dataset Authors.
>
> The notebook [`muril-fine-tuning.ipynb`](https://github.com/hate-detection/training/blob/master/muril-fine-tuning.ipynb) only includes code for full MuRIL model training. Jupyter Notebooks for LoRA Adapter training is currently under-development and will be added at a later date.

### Hyperparameters:
```python
batch_size = 32
learning_rate = 2e-5
epochs = 4
```

### High-level overview:
- Use `google/muril-base-cased` from HuggingFace Transformers library along with `AutoModelForSequenceClassification`
- Divide the dataset into Train and Validation sets
- Perform a 90:10 split
- Tokenize data and pass it to HuggingFace `Trainer`
- Evaluate the trained model with Holdout Testing Set
