#earlier i was using this parser while coding without the TF estimators,
#but estimators already have one hot encoding so i reduced the code for this parser
#to be used in the irisnn.py file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import numpy as np
import urllib as ul

irisURL="http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
irisData=ul.request.urlretrieve(irisURL,filename='irisData')

irisData='./irisData'
dframe=pd.read_csv(irisData,sep=',')
dframe.columns=['sepalLength','sepalWidth','petalLength','petalWidth','species']

X=[]
Y=[]

for i,r in dframe.iterrows():
	#print (r['species'],r['sepalLength'],r['sepalWidth'],r['petalLength'],r['petalWidth'])
	X.append([r['sepalLength'],r['sepalWidth'],r['petalLength'],r['petalWidth']])
	if r['species']=='Iris-setosa':
		Y.append([1,0,0])
	if r['species']=='Iris-versicolor':
		Y.append([0,1,0])
	if(r['species'])=='Iris-virginica':
		Y.append([0,0,1])

X=np.array(X)
Y=np.array(Y) #parsed inputs and outputs

batch_s=10
batch_n=int(485/batch_s)

x=tf.placeholder(shape=[4,485],dtype=tf.float32)
y=tf.placeholder(shape=[3,485],dtype=tf.float32)

ses=tf.Session()
beg=0 #beginning and end of a single batch
end=10
def next_batch():
	global beg,end
	x_batch=[]
	y_batch=[]
	temp=[]
	for j in range(beg,end):
		for i in range(x.get_shape()[0]):
			temp.append(ses.run(x[i][j],feed_dict={x:X.T}))
		x_batch.append(temp)
		temp=[]

	for j in range(beg,end):
		for i in range(y.get_shape()[0]):
			temp.append(ses.run(y[i][j],feed_dict={y:Y.T}))
		y_batch.append(temp)
		temp=[]

	beg=end
	end=end+10

	if beg==150:
		beg=0
		end=10

	x_batch=np.array(x_batch).T
	y_batch=np.array(y_batch).T

	return x_batch,y_batch

print (next_batch())
print (next_batch())

#x_batch and y_batch are 4xbatch and 3xbatch respectively
