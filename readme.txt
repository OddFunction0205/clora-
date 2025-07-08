这些代码是对《Controlled Low-Rank Adaptation with Subspace Regularization for
Continued Training on Large Language Models》这篇论文的复现。

复现了这些：

1.复现了CLoRA的代码框架

2.基于MVP（Minimum Viable Product（最小可行性产品））准则，用CLoRA，在commonsense_170k数据集上训练了TinyLlama-1.1B-Chat-v1.0

3.基于lm-eval测评框架，完成了对boolq，piqa，openbookqa，hellaswag，winogrande，arc_easy，arc_challenge等in domain数据集的测评

复现结果：

![pic/lora.png](pic/lora.png)

![pic/clora.png](pic/clora.png)

![pic/compare.png](pic/compare.png)

结果显示，clora比lora平均提高的2.75%，和论文结果2.9%差不多