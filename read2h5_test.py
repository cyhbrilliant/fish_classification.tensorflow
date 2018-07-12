import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import utils

OK='OK'

DATA_ALL=[]


for i in range(1000):
    print(i)
    tempimg=utils.load_image('test_stg1/'+str(i)+'.jpg') 
    tempimg=tempimg.astype(np.float32)
    DATA_ALL.append(tempimg)
print(OK)



DATA_ALL=np.array(DATA_ALL)


print(DATA_ALL.shape)

file = h5py.File('H5fish_test.h5','w')
file.create_dataset('DATA_ALL', data = DATA_ALL)

#file.create_dataset('ALB', data = DATA_ALB)
#file.create_dataset('BET', data = DATA_BET)
#file.create_dataset('DOL', data = DATA_DOL)
#file.create_dataset('LAG', data = DATA_LAG)
#file.create_dataset('SHARK', data = DATA_SHARK)
#file.create_dataset('YFT', data = DATA_YFT)
#file.create_dataset('OTHER', data = DATA_OTHER)
#file.create_dataset('NoF', data = DATA_NoF)


