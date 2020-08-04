---
description: 高效训练
---

# TF.data + TF.estimator

* [Part1: Introduction to TensorFlow Datasets and Estimators](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)
* [Part2: Introducing TensorFlow Feature Columns](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)
* [Part3: Creating Custom Estimators in TensorFlow](https://developers.googleblog.com/2017/12/creating-custom-estimators-in-tensorflow.html)

### **自定义Estimator通用步骤**：

1. 实现input\_fn：可以使用`tf.data.Datasets` API；
2. 创建feature columns：`tf.feature_column` API；
3. 实现model\_fn：

   ```text
   def my_model_fn(
      features, # This is batch_features from input_fn
      labels,   # This is batch_labels from input_fn
      mode,     # Instance of tf.estimator.ModeKeys: training, predicting, or evaluating.
      params):  # 模型结构的参数
   ```

   * **定义模型结构**

     如果你的自定义Estimator生成的是一个深层的神经网络，你必须定义以下三层（可以使用`tf.layers`或`tf.keras.layers`来定义hidden layers和output layers）：

     * an input layer

       ```text
       # Create the layer of input
       input_layer = tf.feature_column.input_layer(features, feature_columns)
       ```

       如果为了给一个线性模型创建输入层，则需要使用tf.feature\_column.linear\_model\(\)。

     * one or more hidden layers

       ```text
       # Definition of hidden layer: h1
       # (Dense returns a Callable so we can provide input_layer as argument to it)
       h1 = tf.layers.Dense(10, activation=tf.nn.relu)(input_layer)
       ​
       # Definition of hidden layer: h2
       # (Dense returns a Callable so we can provide h1 as argument to it)
       h2 = tf.layers.Dense(10, activation=tf.nn.relu)(h1)
       ```

     * an output layer

       ```text
       # Output 'logits' layer is three numbers = probability distribution
       # (Dense returns a Callable so we can provide h2 as argument to it)
       logits = tf.layers.Dense(3)(h2)
       ```

     * **指定模型在三种模式下的行为**

       即需要分支实现train、evaluate和predict。

       调用方式如下：

       ```text
       classifier.train(
         input_fn=lambda: my_input_fn(FILE_TRAIN, repeat_count=500, shuffle_count=256))
       ```

       * **PREDICT**

         When `model_fn` is called with `mode == ModeKeys.PREDICT`, the model function must return a `tf.estimator.EstimatorSpec` containing the following information:

         * the mode, which is `tf.estimator.ModeKeys.PREDICT`
         * the prediction

         ```text
         # class_ids will be the model prediction for the class (Iris flower type)
         # The output node with the highest value is our prediction
         predictions = { 'class_ids': tf.argmax(input=logits, axis=1) }
         ​
         # Return our prediction
         if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
         ```

         * **EVAL**

           When `model_fn` is called with `mode == ModeKeys.EVAL`, the model function must evaluate the model, returning loss and possibly one or more metrics.

           * losses：使用 `tf.losses` API

             ```text
             # To calculate the loss, we need to convert our labels
             # Our input labels have shape: [batch_size, 1]
             labels = tf.squeeze(labels, 1)          # Convert to shape [batch_size]
             loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
             ```

           * metrics：使用`tf.metrics` API

             ```text
             # Calculate the accuracy between the true labels, and our predictions
             accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])
             ```

             ```text
             # Return our loss (which is used to evaluate our model)
             # Set the TensorBoard scalar my_accurace to the accuracy
             # Obs: This function only sets value during mode == ModeKeys.EVAL
             # To set values during training, see tf.summary.scalar
             if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    eval_metric_ops={'my_accuracy': accuracy})
             ```

           * **TRAIN**

             When `model_fn` is called with `mode == ModeKeys.TRAIN`, the model function must train the model.

             * 指定optimzer：可以使用`tf.train` API；
             * 设置train\_op；
             * 设置tensorboard需要监控的变量：使用`tf.summary.scalar`。

             ```text
             optimizer = tf.train.AdagradOptimizer(0.05)
             train_op = optimizer.minimize(
                loss,
                global_step=tf.train.get_global_step())
             ​
             # Set the TensorBoard scalar my_accuracy to the accuracy
             tf.summary.scalar('my_accuracy', accuracy[1])
             ```

             ```text
             # Return training operations: loss and train_op
             return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op)
             ```

             至此我们的`model_fn`就完整了！

             * the mode, which is `tf.estimator.ModeKeys.TRAIN`
             * the loss
             * the result of the training op

             When the model is called with `mode == ModeKeys.TRAIN`, the model function must return a `tf.estimator.EstimatorSpec` containing the following information:

             * the `mode`, which is `tf.estimator.ModeKeys.EVAL`
             * the model's loss
             * typically, one or more metrics encased in a dictionary.

             When the model is called with `mode == ModeKeys.EVAL`, the model function returns a `tf.estimator.EstimatorSpec` containing the following information:

为了实现一个典型的model\_fn，你必须实现以下两个步骤:

**使用方式如下：**

1. 实例化The custom Estimator

   ```text
   classifier = tf.estimator.Estimator(
      model_fn=my_model_fn,
      model_dir=PATH)  # Path to where checkpoints etc are stored
   ```

2. Train、eval 和predict

   ```text
   classifier.train(
     input_fn=lambda: my_input_fn(FILE_TRAIN, repeat_count=500, shuffle_count=256))
   ```

   ```text
   # Evaluate our model using the examples contained in FILE_TEST
   # Return value will contain evaluation_metrics such as: loss & average_loss
   evaluate_result = estimator.evaluate(
      input_fn=lambda: my_input_fn(FILE_TEST, False, 4)
   print("Evaluation results")
   for key in evaluate_result:
      print("   {}, was: {}".format(key, evaluate_result[key]))
   ```

   ```text
   # Predict the type of some Iris flowers.
   # Let's predict the examples in FILE_TEST, repeat only once.
   predict_results = classifier.predict(
       input_fn=lambda: my_input_fn(FILE_TEST, False, 1))
   print("Predictions on test file")
   for prediction in predict_results:
      # Will print the predicted class, i.e: 0, 1, or 2 if the prediction
      # is Iris Sentosa, Vericolor, Virginica, respectively.
      print prediction["class_ids"][0] 
   ```

{% embed url="https://zhuanlan.zhihu.com/p/53345706" %}



