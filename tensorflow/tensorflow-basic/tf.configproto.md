# tf.ConfigProto\(\)

**tensorflow中使用tf.ConfigProto\(\)配置Session运行参数&&GPU设备指定。**

```python
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
sess = tf.Session(config=config)
```

* **记录设备指派情况 :  tf.ConfigProto\(log\_device\_placement=True\)**

        设置tf.ConfigProto\(\)中参数log\_device\_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备\(几号CPU或几号GPU\)上运行,会在终端打印出各项操作是在哪个设备上运行的。

* **自动选择运行设备 ： tf.ConfigProto\(allow\_soft\_placement=True\)**

        在tf中，通过命令 "with tf.device\('/cpu:0'\):",允许手动设置操作运行的设备。如果手动设置的设备不存在或者不可用，就会导致tf程序等待或异常，为了防止这种情况，可以设置tf.ConfigProto\(\)中参数allow\_soft\_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。

* **限制GPU资源使用**

        为了加快运行效率，TensorFlow在初始化时会尝试分配所有可用的GPU显存资源给自己，这在多人使用的服务器上工作就会导致GPU占用，别人无法使用GPU工作的情况。

tf提供了两种控制GPU资源使用的方法：

**方法一：动态申请显存**

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
```

**方法二：限制GPU的使用率**

```python
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
session = tf.Session(config=config)

或者如下

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)
```

* **设置使用哪块GPU**

**方法一：在Python程序中设置**

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
```

**方法二：在执行Python程序时设置**

```bash
CUDA_VISIBLE_DEVICES=0,1 python yourcode.py
```

**推荐使用第二种，因为第二种更加灵活，不用修改源码。**







