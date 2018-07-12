import matplotlib.pyplot as plt
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
num=1028

file = h5py.File('H5fish_vgg3_noflip.h5','r')
DATA_ALL = file['DATA_ALL'][:]
while True:
    num = int(input('num:'))
    print(num)
    plt.imshow(DATA_ALL[num,:,:,:])
    plt.show()