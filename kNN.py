#coding=utf-8

from numpy import *
import operator

import matplotlib
import matplotlib.pyplot as plt

from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):  #第一参数为未知向量X，第二为训练集，第三为标签集，第四为k值
    dataSetSize = dataSet.shape[0] #读取矩阵第一维的长度，即行数
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #将单向量重复填充成矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  #排序返回元素的索引
    classCount={} #字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #按索引去找标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #已经有了标签的+1，没有的使用默认值0
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True) # k在0位，v在1位，默认排序是有小到大
    #排序时字典被分解成了元组列表，排序后是List类型，key在0位，返回标签
    return sortedClassCount[0][0]

'''
group,labels = createDataSet()
print classify0([0.5,0.8],group,labels,3)
'''

def file2matrix(filename):
    with open(filename,'r') as fr:
        arrayOLines = fr.readlines() #逐行读进List
    numberOfLines = len(arrayOLines) #总行数
    returnMat = zeros((numberOfLines,3)) #二维矩阵，填充为固定值3（无用）
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()   #去除回车换行
        listFromLine = line.split('\t') #以Tab键作为分隔符，切割成为4段的List
        returnMat[index,:] = listFromLine[0:3] #前三段存至矩阵内各列
        classLabelVector.append(int(listFromLine[-1])) #最后一段是标签
        index += 1
    return returnMat,classLabelVector

datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')



'''散点图可视
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()
'''

'''归一化函数'''
def autoNorm(dataSet):
    minVals = dataSet.min(0) #参数0确保取得的是整个列的极值，而不是当前一行的
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals #真实样子都是 1X3 的数组
    m = dataSet.shape[0]
    normDataSet = zeros(shape(dataSet)) #创建相同体积的矩阵
    normDataSet = dataSet - tile(minVals,(m,1))  #当前值减去极小值
    normDataSet /= tile(ranges,(m,1)) #归一化
    return normDataSet,ranges,minVals

normMat,ranges,minVals = autoNorm(datingDataMat)

def datingClassTest():
    hoRatio = 0.05  #用作测试数据的比例,因源数据本身没有排序，不影响分布，故抽取0~m*hoRatio行作为测试
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)  #使用了全局变量datingDataMat
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs): #四个参数分别为 X,Set,Labels,k 值,前2个参数内的冒号可省略（写法习惯而已）
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the calssifier came back with: %d , the real answer is : %d" % (classifierResult,datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))

#print datingClassTest()


def img2vector(filename):
    returnVector = zeros((1,1024))
    with open(filename,'r') as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVector[0,i*32+j] = int(lineStr[j])
    return returnVector

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i] = img2vector('trainingDigits/%s'%fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with: %d, the real answer is: %d " %(classifierResult,classNumStr)
        if (classifierResult != classNumStr):errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


handwritingClassTest()
















