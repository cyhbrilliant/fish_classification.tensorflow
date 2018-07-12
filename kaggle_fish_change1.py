import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

#3777_in_all
ALB_num=1719
BET_num=200
DOL_num=117
LAG_num=67
NoF_num=465
OTHER_num=299
SHARK_num=176
YFT_num=734

file = h5py.File('H5fish_sam.h5','r')
DATA_ALL = file['DATA_ALL'][:]
LABEL2 = file['LABEL2'][:]



def getBatch(Batch_num):
    DATA=[]
    LABEL=[]
    for i in range(int(Batch_num/2)+1):
        index=np.random.randint(DATA_ALL.shape[0])
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL2[index,:])
    for i in range(int(Batch_num/2)-1):
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

Xp=tf.placeholder(tf.float32,shape=[None,360,640,3])
Yp=tf.placeholder(tf.float32,shape=[None,2])


Wconv1=weight_variable([7,7,3,64])
Bconv1=bias_variable([64])
conv1=tf.nn.relu(tf.nn.conv2d(Xp,Wconv1,strides=[1,1,2,1],padding='SAME')+Bconv1)

pool2=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

Wconv3=weight_variable([7,7,64,96])
Bconv3=bias_variable([96])
conv3=tf.nn.relu(tf.nn.conv2d(pool2,Wconv3,strides=[1,1,1,1],padding='SAME')+Bconv3)

pool4=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

Wconv5=weight_variable([5,5,96,128])
Bconv5=bias_variable([128])
conv5=tf.nn.relu(tf.nn.conv2d(pool4,Wconv5,strides=[1,1,1,1],padding='SAME')+Bconv5)

Wconv6=weight_variable([5,5,128,128])
Bconv6=bias_variable([128])
conv6=tf.nn.relu(tf.nn.conv2d(conv5,Wconv6,strides=[1,1,1,1],padding='SAME')+Bconv6)

pool7=tf.nn.max_pool(conv6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

Wconv8=weight_variable([5,5,128,256])
Bconv8=bias_variable([256])
conv8=tf.nn.relu(tf.nn.conv2d(pool7,Wconv8,strides=[1,1,1,1],padding='SAME')+Bconv8)

Wconv9=weight_variable([5,5,256,256])
Bconv9=bias_variable([256])
conv9=tf.nn.relu(tf.nn.conv2d(conv8,Wconv9,strides=[1,1,1,1],padding='SAME')+Bconv9)

pool10=tf.nn.max_pool(conv9,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')


Wconv11=weight_variable([3,3,256,384])
Bconv11=bias_variable([384])
conv11=tf.nn.relu(tf.nn.conv2d(pool10,Wconv11,strides=[1,1,1,1],padding='SAME')+Bconv11)

Wconv12=weight_variable([3,3,384,384])
Bconv12=bias_variable([384])
conv12=tf.nn.relu(tf.nn.conv2d(conv11,Wconv12,strides=[1,1,1,1],padding='SAME')+Bconv12)

pool13=tf.nn.max_pool(conv12,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



Wconv14=weight_variable([3,3,384,512])
Bconv14=bias_variable([512])
conv14=tf.nn.relu(tf.nn.conv2d(pool13,Wconv14,strides=[1,1,1,1],padding='SAME')+Bconv14)

Wconv15=weight_variable([3,3,512,512])
Bconv15=bias_variable([512])
conv15=tf.nn.relu(tf.nn.conv2d(conv14,Wconv15,strides=[1,1,1,1],padding='SAME')+Bconv15)

pool16=tf.nn.max_pool(conv15,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


Wconv17=weight_variable([3,3,512,512])
Bconv17=bias_variable([512])
conv17=tf.nn.relu(tf.nn.conv2d(pool16,Wconv17,strides=[1,1,1,1],padding='VALID')+Bconv17)
conv17_line=tf.reshape(conv17,shape=[-1,6144])


Wdense18=weight_variable([6144,6144])
Bdense18=bias_variable([6144])
dense18=tf.nn.relu(tf.matmul(conv17_line,Wdense18)+Bdense18)

Wdense19=weight_variable([6144,1024])
Bdense19=bias_variable([1024])
dense19=tf.nn.relu(tf.matmul(dense18,Wdense19)+Bdense19)

Wdense20=weight_variable([1024,2])
Bdense20=bias_variable([2])
OUT=tf.nn.softmax(tf.matmul(dense19,Wdense20)+Bdense20)

loss=tf.reduce_mean(-tf.reduce_sum(Yp*tf.log(tf.clip_by_value(OUT,1e-10,1.0)),reduction_indices=[1]))
TrainStep=tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(Yp,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()

for iter in range(100000):
    print('\n',iter)
    Input_batch,Label_batch=getBatch(32)
    acc,error,result=sess.run([accuracy,loss,TrainStep],feed_dict={Xp:Input_batch,Yp:Label_batch})
    print(error,acc)

    if (iter+1)%500==0:
        path = saver.save(sess,'brain_session_kaggle_fish/brain_session.ckpt')
        print(path)
