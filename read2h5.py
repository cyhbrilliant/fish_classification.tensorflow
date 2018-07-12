import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc


#3777_in_all
ALB_num=1719
BET_num=200
DOL_num=117
LAG_num=67
NoF_num=465
OTHER_num=299
SHARK_num=176
YFT_num=734


DATA_ALB=[]
DATA_BET=[]
DATA_DOL=[]
DATA_LAG=[]
DATA_SHARK=[]
DATA_YFT=[]
DATA_OTHER=[]
DATA_NoF=[]

for i in range(ALB_num):
    print(i)
    tempimg=mpimg.imread('train/ALB/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    else:
        pass
    DATA_ALB.append(tempimg)
    #print(tempimg.shape)


for i in range(BET_num):
    print(i)
    tempimg=mpimg.imread('train/BET/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    else:
        pass
    DATA_BET.append(tempimg)
    #print(tempimg.shape)


for i in range(DOL_num):
    print(i)
    tempimg=mpimg.imread('train/DOL/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    else:
        pass
    DATA_DOL.append(tempimg)
    #print(tempimg.shape)

for i in range(LAG_num):
    print(i)
    tempimg=mpimg.imread('train/LAG/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    else:
        pass
    DATA_LAG.append(tempimg)
    #print(tempimg.shape)


for i in range(SHARK_num):
    print(i)
    tempimg=mpimg.imread('train/SHARK/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    else:
        pass
    DATA_SHARK.append(tempimg)
    #print(tempimg.shape)

for i in range(YFT_num):
    print(i)
    tempimg=mpimg.imread('train/YFT/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    else:
        pass
    DATA_YFT.append(tempimg)
    #print(tempimg.shape)


for i in range(OTHER_num):
    print(i)
    tempimg=mpimg.imread('train/OTHER/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    else:
        pass
    DATA_OTHER.append(tempimg)
    #print(tempimg.shape)

for i in range(NoF_num):
    print(i)
    tempimg=mpimg.imread('train/NoF/'+str(i)+'.jpg')
    Tshape=tempimg.shape
    if Tshape[0]/Tshape[1]>0.5625:
        Hnew=Tshape[1]*0.5625
        div=int((Tshape[0]-Hnew)/2)
        tempimg=tempimg[div:Tshape[0]-div,:,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    elif Tshape[0]/Tshape[1]<0.5625:
        Wnew=Tshape[0]/0.5625
        div=int((Tshape[1]-Wnew)/2)
        tempimg=tempimg[:,div:Tshape[1]-div,:]
        tempimg=misc.imresize(tempimg,(720,1280))
    else:
        pass
    DATA_NoF.append(tempimg)
    #print(tempimg.shape)

DATA_ALB=np.array(DATA_ALB)
DATA_BET=np.array(DATA_BET)
DATA_DOL=np.array(DATA_DOL)
DATA_LAG=np.array(DATA_LAG)
DATA_SHARK=np.array(DATA_SHARK)
DATA_YFT=np.array(DATA_YFT)
DATA_OTHER=np.array(DATA_OTHER)
DATA_NoF=np.array(DATA_NoF)

print(DATA_ALB.shape)
file = h5py.File('H5fish.h5','w')
file.create_dataset('ALB', data = DATA_ALB)
file.create_dataset('BET', data = DATA_BET)
file.create_dataset('DOL', data = DATA_DOL)
file.create_dataset('LAG', data = DATA_LAG)
file.create_dataset('SHARK', data = DATA_SHARK)
file.create_dataset('YFT', data = DATA_YFT)
file.create_dataset('OTHER', data = DATA_OTHER)
file.create_dataset('NoF', data = DATA_NoF)

