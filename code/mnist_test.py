# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:21:40 2019

@author: admin
"""

from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

im = Image.open('C:\\Users\\admin\\Pictures\\3.png')
data = list(im.getdata())
result = [(255-x)*1.0/255.0 for x in data] 
#print(result)

# 为输入图像和目标输出类别创建节点
tf.reset_default_graph()
#x = tf.placeholder("float", shape=[None, 784]) # 训练所需数据  占位符

# *************** 开始识别 *************** #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with gfile.FastGFile('e:\\model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    x = sess.graph.get_tensor_by_name("x:0")
    y = sess.graph.get_tensor_by_name("y:0")
    keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
   # print(test) 
    #saver.restore(sess, "./save/model.ckpt")#这里使用了之前保存的模型参数

    prediction = tf.argmax(y,1)
    predint = prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)

    print("recognize result: %d" %predint[0])