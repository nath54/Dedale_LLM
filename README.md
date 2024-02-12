# Dedale LLM

## General Description

Dedale is a Large Language Model using the Transformer architecture and Mixture of Expert.

The main idea is to have a lot of differents Transformer blocks and a router.
The router will choose the next blocks to pass through a certain number of time before passing by a FFN to get the next token.

**Warning :** It is not currently trained. I will firstly train it on very small datasets, and if I get the resources, I will try to train it on a large dataset.
