import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vgg19_trainable as vgg19
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

#3777_in_all
ALB_num = 1722*2
BET_num = 1922*2
DOL_num = 2039*2
LAG_num = 2106*2
NoF_num = 2566*2
OTHER_num = 2865*2
SHARK_num = 3041*2
YFT_num = 3776*2

file = h5py.File('H5fish_vgg2.h5','r')
DATA_ALL = file['DATA_ALL'][:]
LABEL8 = file['LABEL8'][:]
print(DATA_ALL.shape)
def getBatch_aver(Batch_num):
    DATA=[]
    LABEL=[]
    sub_num=int(Batch_num/8)
    for i in range(sub_num):
        index=np.random.randint(0,ALB_num)
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])
    for i in range(sub_num):
        index=np.random.randint(ALB_num,BET_num)
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])
    for i in range(sub_num):
        index=np.random.randint(BET_num,DOL_num)
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])
    for i in range(sub_num):
        index=np.random.randint(DOL_num,LAG_num)
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])
    for i in range(sub_num):
        index=np.random.randint(LAG_num,NoF_num)
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])
    for i in range(sub_num):
        index=np.random.randint(NoF_num,OTHER_num)
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])
    for i in range(sub_num):
        index=np.random.randint(OTHER_num,SHARK_num)
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])
    for i in range(sub_num):
        index=np.random.randint(SHARK_num,YFT_num)
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])
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

def getBatch(Batch_num):
    DATA=[]
    LABEL=[]
    for i in range(Batch_num):
        index=np.random.randint(DATA_ALL.shape[0])
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])

    DATA=np.array(DATA)
    LABEL=np.array(LABEL)

    return np.array(DATA),np.array(LABEL)


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
tf.nn.dropout(dense19,0.5)

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
for iter in range(100000):
    print('\n',iter)
    Input_batch,Label_batch=getBatch_aver(64)
    acc,error,result=sess.run([accuracy,loss,TrainStep],feed_dict={Xp:Input_batch,Yp:Label_batch,train_mode:True})
    #print(error,acc)
    if (iter + 1) % 10 == 0:
        print(sess.run(loss,feed_dict={Xp:DATA_ALL[3400:3500,:,:,:],Yp:LABEL8[3400:3500,:],train_mode:False}))
    if (iter+1)%200==0:
        path = saver.save(sess,'session_class8/session8_2.ckpt')
        print(path)