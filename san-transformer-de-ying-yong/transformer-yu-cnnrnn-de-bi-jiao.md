---
description: Transformer与CNN、RNN的比较。
---

# Transformer与CNN、RNN的比较

## CNN与RNN的改进方向

**RNN并行改进的两种思路：**

* **隐藏神经元之间的并行计算：**SRU
* **打断部分隐层之间的连接，并加深层深**：Sliced RNN

**CNN的解决长距离特征的改进思路：**

* **卷积核跳跃间隔覆盖序列：**dilated CNN\(扩张的CNN\)
* **加深卷积层：**Deep CNN

**寄居改进思路：**

* CNN版的Transformer：将原生Transformer里的multi-head attention替换为CNN；
* RNN版的Transformer：将原生Transformer里的multi-head attention替换为RNN。

## **Transformer与CNN、RNN的比较**

**从以下四个方面比较Transformer、CNN和RNN：**

* **语义特征提取能力**：Transformer&gt;&gt;原生CNN==原生RNN；
* **长距离特征捕获能力**：原生CNN特征抽取器在这方面极为显著地弱于RNN和Transformer，Transformer微弱优于RNN模型\(**尤其在主语谓语距离小于13时**\)，能力由强到弱排序为Transformer&gt;RNN&gt;&gt;CNN; 但在比较远的距离上（**主语谓语距离大于13**），RNN微弱优于Transformer，所以综合看，可以认为Transformer和RNN在这方面能力差不太多，而CNN则显著弱于前两者。
* **任务综合特征抽取能力**：Transformer&gt;原生CNN==原生RNN；
* **并行计算能力及运行效率**：Transformer Base最快，CNN次之，再次Transformer Big，最慢的是RNN。RNN比前两者慢了3倍到几十倍之间。

**单从任务综合效果方面来说，Transformer明显优于CNN，CNN略微优于RNN。速度方面Transformer和CNN明显占优，RNN在这方面劣势非常明显。这两者再综合起来，如果我给的排序结果是Transformer&gt;CNN&gt;RNN。**

## **总结**

* **进退维谷的RNN**
* **一希尚存的CNN**
* **稳操胜券的transformer**

{% embed url="https://zhuanlan.zhihu.com/p/54743941" %}

