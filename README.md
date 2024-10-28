# Code_LLM_Compression

In this project we combine pruning and SVD compression techniques to compress code LLMs.
The pruning method from Shortened LLaMa: https://arxiv.org/pdf/2402.02834
The SVD compression methods from LASER: https://arxiv.org/abs/2312.13558 and SVD-LLM: https://arxiv.org/abs/2403.07378
After compression we heal models with LoRA and evluate models on HumanEval benchmark.

Our hybrid approach combining pruning and SVD technics outperforms compress technics based only on pruning or SVD compression.
