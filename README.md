# Transformers — Week 2 Summary

Over the past week, I focused on getting a deep understanding of Transformer architectures. I studied the original paper, *Attention Is All You Need*, and broke down concepts like self-attention, multi-head attention, positional encoding, and the full encoder-decoder pipeline.

A huge shoutout to Umar Jamil — his YouTube series was incredibly helpful. I followed along and implemented key components like:

- Encoder block  
- Decoder block  
- Feed-forward layers  
- Multi-head attention block  

That hands-on coding experience was very valuable to me

## Project: IMDb Sentiment Classification with Transformers

To apply what I learned, I built a sentiment classification pipeline using my own IMDb dataset (uploaded as CSV). The steps included:

- Cleaning and exploring the dataset  
- Tokenizing with Hugging Face Transformers  
- Training and evaluating three models:

  - BERT (`bert-base-uncased`)  
  - DistilBERT (`distilbert-base-uncased`)  
  - RoBERTa (`roberta-base`)  

Each model was trained on a smaller subset (2000 train samples, 500 test samples) for benchmarking purposes.

## Final Results

| Model      | Accuracy | F1 Score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| BERT       | 0.904    | 0.9051   | 0.8842    | 0.9271 |
| DistilBERT | 0.884    | 0.8831   | 0.8795    | 0.8866 |
| RoBERTa    | 0.910    | 0.9072   | 0.9244    | 0.8907 |

## TL;DR

- BERT performed well, DistilBERT was faster with slightly lower accuracy, and RoBERTa achieved the best overall scores.
- Going from scratch implementation to real-world fine-tuning helped solidify my understanding.
- This wraps up the text classification part of Week 2. Next up: Vision Transformers.
