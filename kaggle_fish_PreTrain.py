import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vgg19_trainable as vgg19
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

#3777_in_all
ALB_num=1719
BET_num=200
DOL_num=117
LAG_num=67
NoF_num=465
OTHER_num=299
SHARK_num=176
YFT_num=734

file = h5py.File('H5fish_vgg.h5','r')
DATA_ALL = file['DATA_ALL'][:]
LABEL2 = file['LABEL2'][:]



def getBatch(Batch_num):
    DATA=[]
    LABEL=[]
    for i in range(int(Batch_num/2)):
        index=np.random.randint(DATA_ALL.shape[0])
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL2[index,:])
    for i in range(int(Batch_num/2)):
        index = np.random.randint(2103,2568)
        DATA.append(DATA_ALL[index, :, :, :])
        LABEL.append(LABEL2[index, :])

    DATA=np.array(DATA)
    LABEL=np.array(LABEL)

    indexlist = np.arange(Batch_num)
    np.random.shuffle(indexlist)
    # print(indexlist)
    DATA_shuffle = []
    LABEL_shuffle=[]
    for i in range(Batch_num):
        DATA_shuffle.append(DATA[indexlist[i], :,:,:])
        LABEL_shuffle.append(LABEL[indexlist[i],:])

    return np.array(DATA_shuffle),np.array(LABEL_shuffle)


#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


Xp = tf.placeholder(tf.float32, [None, 224, 224, 3])
Yp = tf.placeholder(tf.float32, [None, 2])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('vgg19.npy')
vgg.build(Xp, train_mode)


Wdense18=weight_variable([4096,4096])
Bdense18=bias_variable([4096])
dense18=tf.nn.relu(tf.matmul(vgg.fc6,Wdense18)+Bdense18)

Wdense19=weight_variable([4096,1024])
Bdense19=bias_variable([1024])
dense19=tf.nn.relu(tf.matmul(dense18,Wdense19)+Bdense19)

Wdense20=weight_variable([1024,2])
Bdense20=bias_variable([2])
OUT=tf.nn.softmax(tf.matmul(dense19,Wdense20)+Bdense20)

loss=tf.reduce_mean(-tf.reduce_sum(Yp*tf.log(tf.clip_by_value(OUT,1e-10,1.0)),reduction_indices=[1]))
TrainStep=tf.train.AdamOptimizer(0.00001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(Yp,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()

for iter in range(100000):
    print('\n',iter)
    Input_batch,Label_batch=getBatch(32)
    acc,error,result=sess.run([accuracy,loss,TrainStep],feed_dict={Xp:Input_batch,Yp:Label_batch,train_mode:False})
    print(error,acc)

    if (iter+1)%500==0:
        path = saver.save(sess,'session_class2/session2.ckpt')
        print(path)