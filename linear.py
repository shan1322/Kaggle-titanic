import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
datafile='D:/Program Scripts/Machine Learning/datasets/Advertising.xlsx'
book = pd.ExcelFile(datafile)
df=np.array(book.parse('Advertising'))
df5=(np.split(df,2))
df25=(np.split(df5[1],2))
dftrain=np.concatenate((df5[0],df25[0]),axis=0)#train set
dftest=df25[1] #test set
n_samples=df.size
X1=tf.placeholder(tf.float32,name='X1')
X2=tf.placeholder(tf.float32,name='X2')
X3=tf.placeholder(tf.float32,name='X3')
Y=tf.placeholder(tf.float32,name='Y')
w1 = tf.Variable(0.0, name='weights1')
w2 = tf.Variable(0.0, name='weights2')
w3 = tf.Variable(0.0, name='weights3')
b = tf.Variable(0.0, name='bias')
Y_pred=w1*X1+w2*X2+b
print(type(Y_pred))
loss=tf.square(Y-Y_pred,name='loss')
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	#writer = tf.summary.FileWriter('./graphs', sess.graph)
    total_loss=0
    for i in range(0,1):
        for p,q,r,s in dftrain:
            _, l = sess.run([optimizer, loss], feed_dict={X1:p,X2:q,Y:s})
            total_loss+=l
        #print('total loss',i,total_loss/n_samples)
    w1,w2,b = sess.run([w1,w2,b])
acc=[]
for f,g,h,i in dftest:
    temp=w1*f+w2*g+b
    diff=abs(i-temp)
    acc.append((abs(i-diff)/i)*100)
print("Accuracy on test data",np.mean(np.array(acc)))
X1,X2,Y= df.T[0],df.T[1],df.T[3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(X1,X2,Y,'bo')
ax.plot_trisurf(X1,X2,w1*X1+w2*X2+b)
plt.show()
