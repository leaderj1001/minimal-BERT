# Bidirectional Encoder Representations from Transformers (BERT)

## Experiments
| Model | Layer | Hidden | Heads | Acc (%) |
|:-:|:-:|:-:|:-:|:-:|
| BERT, BASE (Paper) | 12 | 768 | 12 | 66.4 |
| BERT, BASE (Our) | 12 | 768 | 12 | 67.9 |
| BERT, LARGE (Paper) | 24 | 1024 | 16 | 70.1 |
| BERT, LARGE (Our) | 24 | 1024 | 16 | wip |

## Quick Start
  - install
  ```
  pip install pytorch_transformers
  ```
  - Prepare Data [RTE dataset link](https://gluebenchmark.com/tasks)
  ```
  data
   ├── train.tsv
   ├── dev.tsv
   ├── test.tsv
  ```
  - BERT eval
  ```
  python BERT_eval/bert_eval.py
  ```

## Pre-training BERT
  - wip

## Reference
  - [Paper Link](https://arxiv.org/abs/1810.04805)
  - [HuggingFace Github](https://github.com/huggingface/transformers)
  - [HuggingFace Documentation](https://huggingface.co/transformers/)
  - [GLUE Benchmark](https://gluebenchmark.com/)
