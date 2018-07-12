import numpy as np

Data_vector=np.loadtxt(open('Data_vector.csv','rb'),delimiter=',',skiprows=0)
Test_vector=np.loadtxt(open('Test_vector.csv','rb'),delimiter=',',skiprows=0)

allrandst=[]
for k in  range(Test_vector.shape[0]):
    print(k)
    dist = []
    vector=Test_vector[k,:]
    for i in range(Data_vector.shape[0]):
        dist.append(np.linalg.norm(vector-Data_vector[i,:]))


    dist=np.array([dist])
    dist=np.reshape(dist,[Data_vector.shape[0]])
    dist=np.argsort(dist)

    #3777_in_all
    ALB_num=1722
    BET_num=1922
    DOL_num=2039
    LAG_num=2106
    NoF_num=2566
    OTHER_num=2865
    SHARK_num=3041
    YFT_num=3776

    randist=np.zeros([Data_vector.shape[0]])
    for i in range(Data_vector.shape[0]):
        if dist[i]<ALB_num:
            randist[i]=0
        elif dist[i]<BET_num:
            randist[i]=1
        elif dist[i]<DOL_num:
            randist[i]=2
        elif dist[i]<LAG_num:
            randist[i]=3
        elif dist[i]<NoF_num:
            randist[i]=4
        elif dist[i]<OTHER_num:
            randist[i]=5
        elif dist[i]<SHARK_num:
            randist[i]=6
        elif dist[i]<YFT_num:
            randist[i]=7
        else:
            print('error')
            pass
    allrandst.append(randist)

allrandst=np.array(allrandst)

print(allrandst.shape)

np.savetxt('sort.csv', allrandst, fmt='%s', delimiter=',')
# temp=np.zeros([1000,8])
# for i in range(1000):
#     print(i)
#     for j in range(8):
#         if j==allrandst[i,0]:
#             temp[i,j]=0.79
#         else:
#             temp[i,j]=0.03



#print(allrandst.shape)
# np.savetxt('testout.csv', temp, fmt='%s', delimiter=',')


allp=np.zeros([1000,8])+15

for i in range(1000):
    for j in range(300):
        allp[i,int(allrandst[i,j])]+=1

allp/=320.
np.savetxt('cluserout.csv', allp, fmt='%s', delimiter=',')
