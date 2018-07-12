import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vgg19_trainable_640360 as vgg19
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

file = h5py.File('H5fish_test3.h5','r')
DATA_ALL = file['DATA_ALL'][:]

# file = h5py.File('H5fish_vgg3_noflip.h5','r')
# DATA_ALL = file['DATA_ALL'][:]

#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


Xp = tf.placeholder(tf.float32, [None, 360, 640, 3])
Yp = tf.placeholder(tf.float32, [None, 8])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('vgg19.npy')
vgg.build(Xp, train_mode)

Wconv1=weight_variable([1,1,512,512])
Bconv1=bias_variable([512])
conv1=tf.nn.relu(tf.nn.conv2d(vgg.pool5,Wconv1,strides=[1,1,1,1],padding='SAME')+Bconv1)

Wconv2=weight_variable([3,3,512,512])
Bconv2=bias_variable([512])
conv2=tf.nn.relu(tf.nn.conv2d(conv1,Wconv2,strides=[1,2,2,1],padding='SAME')+Bconv2)

Wconv3=weight_variable([3,3,512,512])
Bconv3=bias_variable([512])
conv3=tf.nn.relu(tf.nn.conv2d(conv2,Wconv3,strides=[1,1,1,1],padding='SAME')+Bconv3)
conv3_line=tf.reshape(conv3,shape=[-1,30720])

Wdense17=weight_variable([30720,4096])
Bdense17=bias_variable([4096])
dense17=tf.nn.relu(tf.matmul(conv3_line,Wdense17)+Bdense17)

Wdense18=weight_variable([4096,1024])
Bdense18=bias_variable([1024])
dense18=tf.nn.relu(tf.matmul(dense17,Wdense18)+Bdense18)

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
saver.restore(sess,'session_class8/session8_3.ckpt')


Test_out=[]
for i in range(40):
    print(i)
    Test_out.append(sess.run(dense19,feed_dict={Xp:DATA_ALL[i*25:(i+1)*25,:,:,:],Yp:np.zeros([25,8]),train_mode:False}))

Test_out=np.array(Test_out)
Test_out=np.reshape(Test_out,[1000,256])

print(Test_out.shape)
# for i in range(1000):
#     print(i)

# for i in range(1000):
#     for j in range(8):
#         if Test_out[i,j]>0.79:
#             Test_out[i,j]=0.79
#         if Test_out[i,j]<0.03:
#             Test_out[i,j]=0.03
# np.savetxt('testout.csv', Test_out, fmt='%s', delimiter=',')

#np.savetxt('Data_vector.csv', Test_out, fmt='%s', delimiter=',')
np.savetxt('Test_vector.csv', Test_out, fmt='%s', delimiter=',')


