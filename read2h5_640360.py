import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import utils


#3777_in_all
ALB_num=1722
BET_num=200
DOL_num=117
LAG_num=67
NoF_num=460
OTHER_num=299
SHARK_num=176
YFT_num=735

OK='OK'
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
H=360
W=640
for i in range(ALB_num):
    print(i)
    tempimg=utils.load_image2('train/ALB/'+str(i)+'.jpg',H,W)
    tempimg = tempimg.astype(np.float32)
    tempimg_flip=np.zeros(tempimg.shape)
    for m in range(H):
        for n in range(W):
            tempimg_flip[m,n,:]=tempimg[m,W-n-1,:]
    #plt.imshow(tempimg)
    #.show()
    DATA_ALL.append(tempimg_flip)
    DATA_ALL.append(tempimg)
    LABEL8.append([1,0,0,0,0,0,0,0])
    LABEL2.append([0,1])
    LABEL8.append([1, 0, 0, 0, 0, 0, 0, 0])
    LABEL2.append([0, 1])
print(OK)


for i in range(BET_num):
    print(i)
    tempimg=utils.load_image2('train/BET/'+str(i)+'.jpg',H,W)
    tempimg = tempimg.astype(np.float32)
    for m in range(H):
        for n in range(W):
            tempimg_flip[m,n,:]=tempimg[m,W-n-1,:]
    DATA_ALL.append(tempimg_flip)
    DATA_ALL.append(tempimg)
    LABEL8.append([0,1,0,0,0,0,0,0])
    LABEL2.append([0,1])
    LABEL8.append([0, 1, 0, 0, 0, 0, 0, 0])
    LABEL2.append([0, 1])
    #print(tempimg.shape)
print(OK)

for i in range(DOL_num):
    print(i)
    tempimg=utils.load_image2('train/DOL/'+str(i)+'.jpg',H,W)
    tempimg = tempimg.astype(np.float32)
    for m in range(H):
        for n in range(W):
            tempimg_flip[m,n,:]=tempimg[m,W-n-1,:]
    DATA_ALL.append(tempimg_flip)
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,1,0,0,0,0,0])
    LABEL2.append([0,1])
    LABEL8.append([0, 0, 1, 0, 0, 0, 0, 0])
    LABEL2.append([0, 1])
    #print(tempimg.shape)
print(OK)

for i in range(LAG_num):
    print(i)
    tempimg=utils.load_image2('train/LAG/'+str(i)+'.jpg',H,W)
    tempimg = tempimg.astype(np.float32)
    for m in range(H):
        for n in range(W):
            tempimg_flip[m,n,:]=tempimg[m,W-n-1,:]
    DATA_ALL.append(tempimg_flip)
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,1,0,0,0,0])
    LABEL2.append([0,1])
    LABEL8.append([0, 0, 0, 1, 0, 0, 0, 0])
    LABEL2.append([0, 1])
    #print(tempimg.shape)
print(OK)



for i in range(NoF_num):
    print(i)
    tempimg=utils.load_image2('train/NoF/'+str(i)+'.jpg',H,W)
    tempimg = tempimg.astype(np.float32)
    for m in range(H):
        for n in range(W):
            tempimg_flip[m,n,:]=tempimg[m,W-n-1,:]
    DATA_ALL.append(tempimg_flip)
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,0,1,0,0,0])
    LABEL2.append([1,0])
    LABEL8.append([0, 0, 0, 0, 1, 0, 0, 0])
    LABEL2.append([1, 0])
    #print(tempimg.shape)
print(OK)

for i in range(OTHER_num):
    print(i)
    tempimg=utils.load_image2('train/OTHER/'+str(i)+'.jpg',H,W)
    tempimg = tempimg.astype(np.float32)
    for m in range(H):
        for n in range(W):
            tempimg_flip[m,n,:]=tempimg[m,W-n-1,:]
    DATA_ALL.append(tempimg_flip)
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,0,0,1,0,0])
    LABEL2.append([0,1])
    LABEL8.append([0, 0, 0, 0, 0, 1, 0, 0])
    LABEL2.append([0, 1])
    #print(tempimg.shape)
print(OK)

for i in range(SHARK_num):
    print(i)
    tempimg=utils.load_image2('train/SHARK/'+str(i)+'.jpg',H,W)
    tempimg = tempimg.astype(np.float32)
    for m in range(H):
        for n in range(W):
            tempimg_flip[m,n,:]=tempimg[m,W-n-1,:]
    DATA_ALL.append(tempimg_flip)
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,0,0,0,1,0])
    LABEL2.append([0,1])
    LABEL8.append([0,0,0,0,0,0,1,0])
    LABEL2.append([0,1])

    #print(tempimg.shape)
print(OK)

for i in range(YFT_num):
    print(i)
    tempimg=utils.load_image2('train/YFT/'+str(i)+'.jpg',H,W)
    tempimg = tempimg.astype(np.float32)
    for m in range(H):
        for n in range(W):
            tempimg_flip[m,n,:]=tempimg[m,W-n-1,:]
    DATA_ALL.append(tempimg_flip)
    DATA_ALL.append(tempimg)
    LABEL8.append([0,0,0,0,0,0,0,1])
    LABEL2.append([0,1])
    LABEL8.append([0, 0, 0, 0, 0, 0, 0, 1])
    LABEL2.append([0, 1])
    #print(tempimg.shape)
print(OK)

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

file = h5py.File('H5fish_vgg3.h5','w')
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


