import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

file = h5py.File('H5fish_sam.h5','r')
DATA_ALL = file['DATA_ALL'][:]
LABEL8 = file['LABEL8'][:]



def getBatch(Batch_num):
    DATA=[]
    LABEL=[]
    for i in range(Batch_num):
        index=np.random.randint(DATA_ALL.shape[0])
        DATA.append(DATA_ALL[index,:,:,:])
        LABEL.append(LABEL8[index,:])

    return np.array(DATA),np.array(LABEL)


#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

Xp=tf.placeholder(tf.float32,shape=[None,360,640,3])
Yp=tf.placeholder(tf.float32,shape=[None,8])


Wconv1=weight_variable([65,65,3,64])
Bconv1=bias_variable([64])
conv1=tf.nn.relu(tf.nn.conv2d(Xp,Wconv1,strides=[1,1,2,1],padding='VALID')+Bconv1)

Wconv2=weight_variable([65,65,64,64])
Bconv2=bias_variable([64])
conv2=tf.nn.relu(tf.nn.conv2d(conv1,Wconv2,strides=[1,1,1,1],padding='SAME')+Bconv2)

pool3=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')




Wconv4=weight_variable([33,33,64,128])
Bconv4=bias_variable([128])
conv4=tf.nn.relu(tf.nn.conv2d(pool3,Wconv4,strides=[1,1,1,1],padding='VALID')+Bconv4)

Wconv5=weight_variable([33,33,128,128])
Bconv5=bias_variable([128])
conv5=tf.nn.relu(tf.nn.conv2d(conv4,Wconv5,strides=[1,1,1,1],padding='SAME')+Bconv5)

pool6=tf.nn.max_pool(conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')




Wconv7=weight_variable([17,17,128,256])
Bconv7=bias_variable([256])
conv7=tf.nn.relu(tf.nn.conv2d(pool6,Wconv7,strides=[1,1,1,1],padding='VALID')+Bconv7)

Wconv8=weight_variable([17,17,256,256])
Bconv8=bias_variable([256])
conv8=tf.nn.relu(tf.nn.conv2d(conv7,Wconv8,strides=[1,1,1,1],padding='SAME')+Bconv8)

Wconv9=weight_variable([17,17,256,256])
Bconv9=bias_variable([256])
conv9=tf.nn.relu(tf.nn.conv2d(conv8,Wconv9,strides=[1,1,1,1],padding='SAME')+Bconv9)

pool10=tf.nn.max_pool(conv9,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')



Wconv11=weight_variable([7,7,256,512])
Bconv11=bias_variable([512])
conv11=tf.nn.relu(tf.nn.conv2d(pool10,Wconv11,strides=[1,1,1,1],padding='VALID')+Bconv11)

Wconv12=weight_variable([7,7,512,512])
Bconv12=bias_variable([512])
conv12=tf.nn.relu(tf.nn.conv2d(conv11,Wconv12,strides=[1,1,1,1],padding='SAME')+Bconv12)

Wconv13=weight_variable([7,7,512,512])
Bconv13=bias_variable([512])
conv13=tf.nn.relu(tf.nn.conv2d(conv12,Wconv13,strides=[1,1,1,1],padding='SAME')+Bconv13)

pool14=tf.nn.max_pool(conv13,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')



Wconv15=weight_variable([3,3,512,512])
Bconv15=bias_variable([512])
conv15=tf.nn.relu(tf.nn.conv2d(pool14,Wconv15,strides=[1,1,1,1],padding='VALID')+Bconv15)


Wconv16=weight_variable([3,3,512,512])
Bconv16=bias_variable([512])
conv16=tf.nn.relu(tf.nn.conv2d(conv15,Wconv16,strides=[1,1,1,1],padding='VALID')+Bconv16)
conv16_line=tf.reshape(conv16,shape=[-1,4608])

Wdense17=weight_variable([4608,4608])
Bdense17=bias_variable([4608])
dense17=tf.nn.relu(tf.matmul(conv16_line,Wdense17)+Bdense17)

Wdense18=weight_variable([4608,1000])
Bdense18=bias_variable([1000])
dense18=tf.nn.relu(tf.matmul(dense17,Wdense18)+Bdense18)

Wdense19=weight_variable([1000,8])
Bdense19=bias_variable([8])
OUT=tf.nn.softmax(tf.matmul(dense18,Wdense19)+Bdense19)

loss=tf.reduce_mean(-tf.reduce_sum(Yp*tf.log(tf.clip_by_value(OUT,1e-10,1.0)),reduction_indices=[1]))
TrainStep=tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(Yp,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()

for iter in range(10000):
    print('\n',iter)
    Input_batch,Label_batch=getBatch(16)
    error,result=sess.run([loss,TrainStep],feed_dict={Xp:Input_batch,Yp:Label_batch})
    print(error)

    if (iter+1)%500==0:
        path = saver.save(sess,'brain_session_kaggle_fish/brain_session.ckpt')
        print(path)