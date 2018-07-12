import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

label_bf= np.loadtxt(open("class2_label.csv","rb"),delimiter=",",skiprows=0)  

label=np.zeros([1000,2])
for i in range(label_bf.shape[0]):
    label[int(label_bf[i]),0]=1

for i in range(1000):
    if label[i,0]!=1:
        label[i,1]=1

preict= np.loadtxt(open("class2_testout.csv","rb"),delimiter=",",skiprows=0)  
cluser=np.loadtxt(open("cluserout.csv","rb"),delimiter=",",skiprows=0)
preict_clu=np.zeros([1000,2])

for i in range(1000):
    if cluser[i,4]>0.5:
        preict_clu[i,0]=1
    else:
        preict_clu[i,1]=1

correct_prediction = tf.cast(tf.equal(tf.argmax(preict,1), tf.argmax(label,1)), "float")
accuracy = tf.reduce_mean(correct_prediction)
acc,Corr=tf.Session().run([accuracy,correct_prediction])

print(acc)

Ecast=[]
for i in range(Corr.shape[0]):
    if Corr[i]==0:
        Ecast.append(i)

Ecast=np.array(Ecast)
print(Ecast)

#print(tf.Session().run(correct_prediction))





