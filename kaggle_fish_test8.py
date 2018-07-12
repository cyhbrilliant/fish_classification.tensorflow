import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vgg19_trainable as vgg19
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

file = h5py.File('H5fish_test.h5','r')
DATA_ALL = file['DATA_ALL'][:]



#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


Xp = tf.placeholder(tf.float32, [None, 224, 224, 3])
Yp = tf.placeholder(tf.float32, [None, 8])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('vgg19.npy')
vgg.build(Xp, train_mode)


Wdense18=weight_variable([4096,1024])
Bdense18=bias_variable([1024])
dense18=tf.nn.relu(tf.matmul(vgg.fc6,Wdense18)+Bdense18)

Wdense19=weight_variable([1024,256])
Bdense19=bias_variable([256])
dense19=tf.nn.relu(tf.matmul(dense18,Wdense19)+Bdense19)

Wdense20=weight_variable([256,8])
Bdense20=bias_variable([8])
OUT=tf.nn.softmax(tf.matmul(dense19,Wdense20)+Bdense20)

loss=tf.reduce_mean(-tf.reduce_sum(Yp*tf.log(tf.clip_by_value(OUT,1e-10,1.0)),reduction_indices=[1]))
TrainStep=tf.train.AdamOptimizer(0.00001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(Yp,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,'session_class8/session8.ckpt')


Test_out=[]
for i in range(10):
    print(i)
    Test_out.append(sess.run(OUT,feed_dict={Xp:DATA_ALL[i*100:(i+1)*100,:,:,:],Yp:np.zeros([100,8]),train_mode:False}))

Test_out=np.array(Test_out)
Test_out=np.reshape(Test_out,[1000,8])

print(Test_out.shape)
# for i in range(1000):
#     print(i)

for i in range(1000):
    for j in range(8):
        if Test_out[i,j]>0.79:
            Test_out[i,j]=0.79
        if Test_out[i,j]<0.03:
            Test_out[i,j]=0.03
np.savetxt('testout.csv', Test_out, fmt='%s', delimiter=',')


