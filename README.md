# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Repository

GitHub Repository:
https://github.com/tanushreehajare/nlptransformer

## Weights & Biases Report

W&B Report:
https://wandb.ai/id25s004-indian/da6401-a3/reports/DA6401-Assignment-3-ID25S004--VmlldzoxNjg1NDcyNw?accessToken=j46qgjul9s0m5ges87y7tlzplldjtmuwbroxu3xi9f1k09tpqgu68a40a0f1aofa

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
