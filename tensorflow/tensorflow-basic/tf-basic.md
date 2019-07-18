# TF basic

“TensorFlow™ is an open source software library for numerical computation using data flow graphs.” 

### 基本流程

TensorFlow separates definition of computations from their execution.

* **Phase 1: assemble a graph**
* **Phase 2: use a session to execute operations in the graph.**

### Tf.Session\(\)的作用：

* **A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.**
* **Session will also allocate memory to store the current values of variables.**

### Tf.Graph\(\) and Why Graph：

**It’s better to have disconnected subgraphs within one graph for distributed computing.**

1. **Save computation. Only run subgraphs that lead to the values you want to fetch.**
2. **Break computation into small, differential pieces to facilitate auto-differentiation.**
3. **Facilitate distributed computation, spread the work across multiple CPUs, GPUs, TPUs, or other devices.**
4. **Many common machine learning models are taught and visualized as directed graphs**

\*\*\*\*

  


