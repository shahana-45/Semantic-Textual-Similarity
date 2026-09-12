# Semantic Textual Similarity

Three PyTorch models for Semantic Textual Similarity (STS), the task of predicting how semantically related two sentences are, built from scratch for the *Neural Networks: Theory and Implementation* (NNTI) course at Saarland University. Each task builds on the last, culminating in an open-ended challenge to beat the earlier architectures.

Trained and evaluated on the [SICK](http://marcobaroni.org/composes/sick.html) (Sentences Involving Compositional Knowledge) dataset, which pairs sentences with a human-annotated relatedness score from 1 to 5.

## Tasks

**Task 1: Siamese BiLSTM with self-attention**
A Siamese network where each sentence is encoded independently by a shared BiLSTM, then compressed into a fixed-size embedding via the structured self-attention mechanism from Lin et al. (2017). The two sentence embeddings are compared with the exponential-distance similarity function from Mueller & Thyagarajan (2016). This is the baseline for the other two tasks.

**Task 2: + Transformer encoder**
Adds a Transformer encoder (Vaswani et al., 2017), implemented from scratch with no `nn.TransformerEncoder`/`nn.MultiheadAttention`, in front of the BiLSTM, so word embeddings are first contextualized by multi-head self-attention before being passed through the same BiLSTM + self-attention pipeline as Task 1. Number of encoder layers is configurable.

**Task 3: Challenge task, Deep Averaging Network**
An open task to try to beat Tasks 1 and 2 by any reasonable means. Instead of adding complexity, this went the other way: a [Deep Averaging Network](https://aclanthology.org/P15-1162/) (mean-pool word embeddings, then a stack of feed-forward layers) replaces the BiLSTM/attention stack entirely. It trains faster and generalized far better than either recurrent model.

All three models share the same Siamese architecture (two towers with tied weights) and the same similarity head: `exp(-||a - b||₁)`, clamped to `(0, 1)`.

## Results

Reported metric is `1 - MSE` between predicted and normalized (0-1) gold relatedness scores on the SICK test set, as computed in each task's `test.py`:

| Task | Architecture | Test score (1 - MSE) |
|------|-------------|----------------------|
| 1 | Siamese BiLSTM + self-attention | 0.457 |
| 2 | + from-scratch Transformer encoder | 0.671 |
| 3 | Siamese Deep Averaging Network | 0.972 |

Takeaways:
- Adding a from-scratch Transformer encoder ahead of the BiLSTM (Task 2) clearly helped over the plain BiLSTM baseline (Task 1).
- The simpler Deep Averaging Network (Task 3) outperformed both recurrent/attention-based models by a wide margin. Averaging embeddings threw away far less signal than the BiLSTM was able to preserve, and being far shallower it was much easier to optimize on a small dataset like SICK.
- `1 - MSE` is a simple regression proxy, not a calibrated correlation metric (like Pearson's r, which is more standard for STS), so treat the numbers as relative comparisons across these three models, not absolute benchmarks.

## Pipeline

1. **Data**: SICK train/validation/test splits loaded via Hugging Face `datasets`.
2. **Preprocessing**: lowercasing, punctuation stripping, stopword removal (`preprocess.py`).
3. **Tokenization & vocab**: `torchtext` `Field` with basic English tokenization; vocabulary built from the full corpus.
4. **Embeddings**: pretrained 300-d FastText vectors (`fasttext.simple.300d`), frozen and loaded into an `nn.Embedding` layer.
5. **Model**: task-specific Siamese encoder (see above) producing one embedding per sentence.
6. **Similarity & loss**: exponential negative L1-distance similarity score, trained against the normalized gold relatedness score (MSE loss).
7. **Evaluation**: dev-set accuracy tracked during training; final test-set score computed after reloading the best saved checkpoint.

## Repository structure

```
Task 1/   Task 1 code + notebook (siamese BiLSTM + self-attention) + saved weights
Task 2/   Task 2 code + notebook (+ from-scratch Transformer encoder)
Task 3/   Task 3 code + notebook (Siamese DAN) + saved weights
```

Each task folder is self-contained:
- `*.ipynb`: the runtime notebook (data loading, training, evaluation)
- `sts_data.py` / `dataset.py` / `preprocess.py`: data loading, vocab/tokenization, and text cleaning
- `<model>.py`: the model definition for that task (`siamese_lstm_attention.py`, `siamese_lstm_attention_with_transformer.py` + `Trans_Encoder.py`, `siamese_dan.py`)
- `train.py` / `test.py`: training loop and test-set evaluation
- `utils.py`: shared similarity-score function
- `*.pth`: saved trained weights (Tasks 1 and 3)

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Each task can then be run via its notebook, or directly from the corresponding `main.py`/`train.py` in that task's folder.

## References

- Mueller, J., & Thyagarajan, A. (2016). [Siamese Recurrent Architectures for Learning Sentence Similarity](https://ojs.aaai.org/index.php/AAAI/article/view/10350). AAAI.
- Lin, Z., et al. (2017). [A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130). ICLR.
- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.

## Background

Originally completed in 2022 as the final project for *Neural Networks: Theory and Implementation* (NNTI), Saarland University.
