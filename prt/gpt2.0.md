---
description: >-
  MT-DNN paper：“Multi-Task Deep Neural Networks for Natural Language
  Understanding”，GPT2.0 Paper："Language Models are Unsupervised Multitask
  Learners".
---

# MT-DNN和GPT2.0

MT-DNN和GPT2.0都是对Bert更进一步的改进：

* **MT-DNN：**更进一步的多任务改造，结构上底层就是标准的Bert Transformer，第一阶段采用Bert的预训练模型不动，在Finetuning阶段，在上层针对不同任务构造不同优化目标，所有不同上层任务共享底层Transformer参数，这样就强迫Transformer通过预训练做很多NLP任务，来学会新的知识，并编码到Transformer的参数中。
* **GPT2.0：**用更大的模型、更高质量、更广泛、更大的数据来做预训练。

具体看下面这篇博客即可：

{% embed url="https://zhuanlan.zhihu.com/p/56865533" %}





