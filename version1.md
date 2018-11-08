# Bofei's Knowledge Outputs (Oct.)



## python - numpy foundation
### 数组对象ndarray
创建一个数组对象(一维数组)，并改变维度(二维数组，两行三列)
	np.array([0,1,2,3,4,5]).reshape(2,3)
	np.arange(6).reshape(2,3)
创建一个5个元素的数组，并初始化为0
	np.zeros(5)
#### ndarray 属性
	itemsize 给出数组中的元素在内存中所占的字节数
	2.size 给出数组元素的总个数
	nbytes 整个数组所占的存储空间（itemsize * size）
	T 数组转置 等于 transpose操作 (对于一维数组，其T属性就是原数组)
	flat 返回一个numpy.flatiter对象
	imag 给出复数数组的虚部，如果数组中包含复数元素，则其数据类型自动变为复数型
	real 给出复数数组的实部。如果数组中只包含实数元素，则其real属性将输出原数组 
	ndim 给出数组的维数，或数组轴的个数
	shape 获取数组对象的维度
	dtype 返回数组对象的元素数据类型
#### ndarray常用操作
1. 索引和切片

2. 改变数组维度
  reshape() 改变数组维度 
  ravel() 展平数组
  flatten() 类似于ravel, 会请求分配内存来保存结果，ravel 只是返回数组的一个视图
3. 分割
  水平 hsplit
  垂直 vsplit
  深度 dsplit
4. 组合
  水平 hstack
  垂直 vstack
  深度 dstack
### numpy 常用操作
1. 读写文件 (读取data.csv文件的第1至5列，赋值到a,b,c,d,e。delimiter分隔符，unpack true表示 )：
  a,b,c,d,e=np.loadtxt('data.csv', delimiter=',', usecols=(1,2,3,4,5), unpack=True)
### numpy常用函数
	max()
	min()
	median()
	msort(c)
	ptp()
	std()
	diff()
	where()
	sqrt()
	take()
	argmin()
	argmax()
	split()
	exp()
	sun()
	linspace()
	convolve()


## Deep Learning Neural Network foundation
### CNN 卷积神经网络 (Lenet)

### RNN-LSTM 循环长短期记忆神经网络

### 目标检测模型

#### Faster - RCNN

#### SSD 


## python - tensorflow operation

```python
# Tensorflow 模型持久化，保存为PB格式
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   graph_def = tf.get_default_graph().as_graph_def() 
   # 将变量转换为常量，output1,output2… 为输出节点名
   output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['output1','output2'])
# 在当前目录下生成model.pb模型文件
   with tf.gfile.GFile('./model.pb','wb') as f:
      f.write(output_graph_def.SerializeToString())
```

```python
#读取PB格式模型文件
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = './Model/CNN_ckpt.pb'
    with open(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input = sess.graph.get_tensor_by_name("input:0") # 根据tensor名获取tenor
        output = sess.graph.get_tensor_by_name("output:0") # 根据tensor名获取tenor
        y_conv_2 = sess.run(output, feed_dict={input:mnist.test.images}) # 运行pb模型
        y_2 = mnist.test.labels
        correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y_2, 1))
        accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
        print ("check accuracy %g" % sess.run(accuracy_2))

```

## python - tensorflow visualization - tensorboard





## python - NN model action in tensorflow
### 基于MNIST数据集训练CNN神经网络

### 基于MNIST数据集训练RNN-LSTM神经网络


## 集成HiAI DDK
**流程** 
	1. 算子兼用性评估 使用OperatorsCheck工具
	2. 模型格式装换 使用pb_to_offine工具
	3. 模型集成 - 模型预处理、加载离线模型，数据预处理，运行模型、数据后处理，卸载模型
### 算子兼用性评估
	运行OperatorsCheck 工具进行算子评估
	在OperatorsCheck.jar所在文件夹中打开命令行，输入 java -jar ./OperatorsCheck.jar -t tensorflow -p ./*.pb -on "output_node_names" 
	*.pb为模型文件，output_node_names为输出节点名
	如显示未通过可根据生成的report，结合DDK文件夹中提供的算子支持文档对相关算子进行修改

### 模型格式装换
Tensorflow pb模型转换为Cambricon 离线模型可使用DDK提供的pb_to_offline 文件在linux系统上执行。转换需提供待转换的pb模型文件和txt相关参数文档。txt参数文档的格式为：
```text
model_name: InceptionV3.cambricon //生成的cambricon离线模型 文件名
session_run{ 
input_nodes(1): //指定输入节点个数(1个) 
"input",1,299,299,3 //指定输入节点的名称和shape （N,H,W,C）
output_nodes(1): //指定输出节点个数(1个) 
"InceptionV3/Predictions/Softmax" //指定输出节点的名称 
}
```
在linux上运行模型转换程序： 
	sh run.sh
后根据提示输入要转换的pb模型文件名和txt相关参数文件。如terminal打印 [Info]:modelDesc init success! model segment : 1 cngen::ipuLibExit [Info]:ipuMaxMemory used: 83032320 则转换成功，且会在当前目录下生成cambricon模型文件


###在手机上测试 Cambricon 模型 & ADB 常用操作
在手机上可使用ai_test 文件单独测试cambricon模型是否可用，步骤为
1. 将cambricon模型文件和ai_test文件使用adb工具上传至手机data/local/tmp目录下
2. 在adb shell 中执行
```text
./ai_test model model.cambricon inputfile 1
```
控制台输出目标模型的输入和输出节点信息则表示cambricon模型可在手机平台上使用，否则会报load model result:-1000 错误，HIAI_ModelMnager_getModelTensorInfo failed!

** ADB 操作 Tips:**
	1. adb devices 显示已连接的设备
	2. adb push D:\aa\bb /data/local/tmp 将本地D盘aa文件夹中的bb 上传至手机中的data/local/tmp目录
	3. adb pull /data/log D:\aa\bb 将Android设备上的日志拷贝至本地目录
	3. adb remount 如push显示传输失败也是用adb remount命令获取相关权限
	4. adb shell 连接android 终端
	5. chmod 777 ai_test 将ai_test赋予777权限，运行ai_test程序出现permission denied 时可尝试此操作


## Hiai 模型集成流程


## Android NNAPI


## 模型量化






