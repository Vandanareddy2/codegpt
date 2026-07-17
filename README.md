# CodeGPT

Training a GPT-style Decoder Transformer from Scratch for Python Code Generation

## Project Overview

This project builds a small GPT model from scratch using PyTorch to understand transformer architecture and train it on Python code.

## Project Structure

codegpt/
├── src/
│   ├── data/
│   ├── tokenizer/
│   ├── model/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
├── checkpoints/
├── tests/
├── notebooks/
├── requirements.txt
└── README.md

## Getting Started

1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Next Steps

- Build tokenizer
- Create dataset pipeline
- Implement transformer architecture
- Train the model