import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc


Wsample=640
Hsample=360

#3777_in_all
ALB_num=1719
BET_num=200
DOL_num=117
LAG_num=67
NoF_num=465
OTHER_num=299
SHARK_num=176
YFT_num=734


#DATA_ALB=[]
#DATA_BET=[]
#DATA_DOL=[]
#DATA_LAG=[]
#DATA_SHARK=[]
#DATA_YFT=[]
#DATA_OTHER=[]
#DATA_NoF=[]
DATA_ALL=[]
LABEL8=[]#8_classif
LABEL2=[]#2_classif

for i in range(ALB_num):
    tempimg=mpimg.imread('train/ALB/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    else:
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    #print(tempimg.shape)
    DATA_ALL.append(tempimg)
    LABEL8.append([1,0,0,0,0,0,0,0])
    LABEL2.append([0,1])
print('OK')


for i in range(BET_num):
    tempimg=mpimg.imread('train/BET/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    else:
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    DATA_ALL.append(tempimg)
    LABEL8.append([0,1,0,0,0,0,0,0])
    LABEL2.append([0,1])
    #print(tempimg.shape)
print('OK')

for i in range(DOL_num):
    tempimg=mpimg.imread('train/DOL/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    else:
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,1,0,0,0,0,0])
    LABEL2.append([0,1])
    #print(tempimg.shape)
print('OK')

for i in range(LAG_num):
    tempimg=mpimg.imread('train/LAG/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    else:
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,1,0,0,0,0])
    LABEL2.append([0,1])
    #print(tempimg.shape)
print('OK')



for i in range(NoF_num):
    tempimg=mpimg.imread('train/NoF/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    else:
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,0,1,0,0,0])
    LABEL2.append([1,0])
    #print(tempimg.shape)
print('OK')

for i in range(OTHER_num):
    tempimg=mpimg.imread('train/OTHER/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    else:
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,0,0,1,0,0])
    LABEL2.append([0,1])
    #print(tempimg.shape)
print('OK')

for i in range(SHARK_num):
    tempimg=mpimg.imread('train/SHARK/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    else:
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,0,0,0,1,0])
    LABEL2.append([0,1])
    #print(tempimg.shape)
print('OK')

for i in range(YFT_num):
    tempimg=mpimg.imread('train/YFT/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    else:
        tempimg=misc.imresize(tempimg,(Hsample,Wsample))
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,0,0,0,0,1])
    LABEL2.append([0,1])
    #print(tempimg.shape)
print('OK')

#DATA_ALB=np.array(DATA_ALB)
#DATA_BET=np.array(DATA_BET)
#DATA_DOL=np.array(DATA_DOL)
#DATA_LAG=np.array(DATA_LAG)
#DATA_SHARK=np.array(DATA_SHARK)
#DATA_YFT=np.array(DATA_YFT)
#DATA_OTHER=np.array(DATA_OTHER)
#DATA_NoF=np.array(DATA_NoF)

#print(DATA_ALB.shape)

DATA_ALL=np.array(DATA_ALL)
LABEL8=np.array(LABEL8)
LABEL2=np.array(LABEL2)

print(DATA_ALL.shape,LABEL8.shape,LABEL2.shape)

file = h5py.File('H5fish_sam.h5','w')
file.create_dataset('DATA_ALL', data = DATA_ALL)
file.create_dataset('LABEL8', data = LABEL8)
file.create_dataset('LABEL2', data = LABEL2)
#file.create_dataset('ALB', data = DATA_ALB)
#file.create_dataset('BET', data = DATA_BET)
#file.create_dataset('DOL', data = DATA_DOL)
#file.create_dataset('LAG', data = DATA_LAG)
#file.create_dataset('SHARK', data = DATA_SHARK)
#file.create_dataset('YFT', data = DATA_YFT)
#file.create_dataset('OTHER', data = DATA_OTHER)
#file.create_dataset('NoF', data = DATA_NoF)


