import numpy as np
import operator

def fileLoad(filename):
     f=open(filename)
     lines=f.readlines()
     size=len(lines)
     returnMat=np.zeros((size,3))
     classLabelVector=[]
     index=0
     for line in lines:
         line=line.strip()
         line=line.split('\t')
         returnMat[index,:]=line[0:3]
         if line[-1] == 'didntLike':
            classLabelVector.append(1)
         elif line[-1] == 'smallDoses':
            classLabelVector.append(2)
         elif line[-1] == 'largeDoses':
            classLabelVector.append(3)
     index +=1
     return returnMat, classLabelVector
         
         
def classify0(inX,dataSet,labels,k):
    dataSetsize=dataSet.shape[0]
    dataInx=np.tile(inX,[dataSetsize,1])-dataSet
    sqdataInx=dataInx**2
    sqSumdataInx=sqdataInx.sum(axis=1)
    distance=sqSumdataInx**0.5
    sortedDistance=distance.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistance[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

if __name__ == '__main__':
    #创建数据集
    group, labels = fileLoad(r"...\dataset.txt") #路径
    #输入
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    test = [ffMiles,iceCream,precentTats]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    dictdata={1:"didntLike",2:'smallDoses',3:'largeDoses'}
    print(test_class)
    print(dictdata[test_class])
