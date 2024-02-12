# Dedale LLM

## General Description

Dedale is a large language model (LLM) using the Transformer architecture and Mixture of Experts.

The main idea is to have a **large number** of different Transformer blocks and a router. The router will choose the **next blocks** to pass through a certain number of times before passing by an FFN to get the **next token**.

**Warning:** It is currently not trained. I will first train it on very small datasets, and if I can get the resources, I will try to train it on a large dataset.

## Description of this Code

The model source code is split into the `lib.py` and `model.py` files. It uses the PyTorch library.