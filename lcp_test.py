#coding=utf8
import time
import cv2

time1 = time.time()
for i in range(100):
    cv2.imread(r'E:\cuiyuhao\python\kaggle_fish\train\ALB\{}.jpg'.format(i))
ts = time.time()-time1
print(ts)