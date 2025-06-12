# Transformers — Week 2 Summary

Over the past week, I focused on getting a deep understanding of Transformer architectures. I studied the original paper, *Attention Is All You Need*, and broke down concepts like self-attention, multi-head attention, positional encoding, and the full encoder-decoder pipeline.

A huge shoutout to Umar Jamil — his YouTube series was incredibly helpful. I followed along and implemented key components like:

- Encoder block  
- Decoder block  
- Feed-forward layers  
- Multi-head attention block  

That hands-on coding experience was very valuable to me.

## Project: IMDb Sentiment Classification with Transformers

To apply what I learned, I built a sentiment classification pipeline using my own IMDb dataset (uploaded as CSV). The steps included:

- Cleaning and exploring the dataset  
- Tokenizing with Hugging Face Transformers  
- Training and evaluating three models:

  - BERT (`bert-base-uncased`)  
  - DistilBERT (`distilbert-base-uncased`)  
  - RoBERTa (`roberta-base`)  

Each model was trained on a smaller subset (2000 train samples, 500 test samples) for benchmarking purposes.

### Final Results (IMDb)

| Model      | Accuracy | F1 Score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| BERT       | 0.904    | 0.9051   | 0.8842    | 0.9271 |
| DistilBERT | 0.884    | 0.8831   | 0.8795    | 0.8866 |
| RoBERTa    | 0.910    | 0.9072   | 0.9244    | 0.8907 |

---

## Project: Image Classification with Vision Transformers (ViT)

In the second part of the week, I moved on to **Vision Transformers**, applying the same concepts to image classification using the **CIFAR-10** dataset.

### What I Tried:
- Used `google/vit-base-patch16-224-in21k` from Hugging Face
- Applied advanced `torchvision` transforms like resizing, cropping, flipping, and normalization
- Used `ViTImageProcessor` to preprocess images for the ViT model
- Implemented a full training pipeline using PyTorch and Hugging Face Transformers

### Challenges Faced:
- Faced GPU under-utilization on Colab (solved by switching to T4 and tuning batch sizes)
- Needed to clamp image pixel values and disable redundant rescaling to avoid preprocessing errors
- Training was **very slow**, so I limited the dataset size and number of epochs
- Experimented with learning rate schedulers to boost convergence

### Final Results (CIFAR-10 on ViT)

| Metric     | Score   |
|------------|---------|
| Accuracy   | 0.9329  |
| Precision  | 0.9328  |
| Recall     | 0.9329  |
| F1 Score   | 0.9327  |

> Trained for 3 epochs on ~50,000 CIFAR-10 images with data augmentation and AdamW optimizer.  
> Performance was excellent after resolving preprocessing bottlenecks.

---

## TL;DR

- BERT-based models handled text classification efficiently with high accuracy across the board.
- RoBERTa was the top performer in sentiment classification.
- Vision Transformers were slower to train, but delivered strong performance on image data.
- Learned to debug real-world model training issues, from GPU usage to Hugging Face quirks.

