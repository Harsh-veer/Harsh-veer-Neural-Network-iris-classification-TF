from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import numpy as np
import urllib as ul

tf.logging.set_verbosity(tf.logging.INFO)


#irisURL="http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#irisData=ul.request.urlretrieve(irisURL,filename='irisData')
#commented these lines after once locally importing data
irisData='./irisData'
dframe=pd.read_csv(irisData,sep=',')
dframe.columns=['sepalLength','sepalWidth','petalLength','petalWidth','species']

X=[]
Y=[]

for i,r in dframe.iterrows():
	#print (r['species'],r['sepalLength'],r['sepalWidth'],r['petalLength'],r['petalWidth'])
	X.append([r['sepalLength'],r['sepalWidth'],r['petalLength'],r['petalWidth']])
	if r['species']=='Iris-setosa':
		Y.append(0)
	if r['species']=='Iris-versicolor':
		Y.append(1)
	if(r['species'])=='Iris-virginica':
		Y.append(2)

#why values of Y are 0,1,2, check the page "https://www.tensorflow.org/api_docs/python/tf/losses/sparse_softmax_cross_entropy"

X=np.array(X)
Y=np.array(Y) #parsed inputs and outputs


def iris_model_fn(features,labels,mode):
	dense1=tf.layers.dense(inputs=features,units=6,activation=tf.nn.relu)
	dense2=tf.layers.dense(inputs=dense1,units=6,activation=tf.nn.relu)
	logits=tf.layers.dense(inputs=dense2,units=3)

	predictions={
		"classes":tf.argmax(input=logits,axis=1),
		"probabilities":tf.nn.softmax(logits,name="softmax_tensor")
	}

	if mode==tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

	loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

	if mode==tf.estimator.ModeKeys.TRAIN:
		optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=optimizer)

	eval_metric_ops={
		"accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])
	}

	return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	iris_classifier=tf.estimator.Estimator(model_fn=iris_model_fn,model_dir="./irisModel")
	tensors_to_log={"probabilities":"softmax_tensor"}
	logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)

	train_input_fn=tf.estimator.inputs.numpy_input_fn(
		x=X,
		y=Y,
		batch_size=10,
		num_epochs=None,
		shuffle=True)

	iris_classifier.train(
		input_fn=train_input_fn,
		steps=20000,
		hooks=[logging_hook])

	eval_input_fn=tf.estimator.inputs.numpy_input_fn(
		x=X,
		y=Y,
		num_epochs=1,
		shuffle=False)

	eval_results=iris_classifier.evaluate(input_fn=eval_input_fn)
	print (eval_results)

if __name__=="__main__":
	tf.app.run()
