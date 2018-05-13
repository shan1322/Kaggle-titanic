import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
datafile='D:/Program Scripts/Machine Learning/datasets/titanic/train.xlsx'
book = pd.ExcelFile(datafile)
df=(book.parse('train'))
df['Sex'].replace('female',0,inplace=True)
df['Sex'].replace('male',1,inplace=True)
df=df.dropna(how='any')
df1=np.array(df)
n_samples=df.size
Pclass=tf.placeholder(tf.float32,name='Pclass')
Sex=tf.placeholder(tf.float32,name='Sex')
Age=tf.placeholder(tf.float32,name='Age')
Sib=tf.placeholder(tf.float32,name='Sib')
Parch=tf.placeholder(tf.float32,name='Parch')
Y=tf.placeholder(tf.float32,name='Y')
w1=tf.Variable(0.0,name='weight1')
w2=tf.Variable(0.0,name='weight2')
w3=tf.Variable(0.0,name='weight3')
w4=tf.Variable(0.0,name='weight4')
w5=tf.Variable(0.0,name='weight5')
bi=tf.Variable(0.0,name='bias')
theta=-(w1*Pclass+w2*Sex+w3*Age+w4*Sib+w5*Parch+bi)
ex=tf.exp(theta)
deno=1+ex
Y_pred=tf.divide(1,deno)
loss=-(tf.add(tf.multiply(Y,tf.log(Y_pred)),(1-Y)*tf.log(1-Y_pred)))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_loss=0
    for epchos in range(0,10):
        for a,b,c,d,e,f,g,h,i,j,k,l in df1:
            _,l=sess.run([optimizer,loss],feed_dict={Pclass:c,Sex:e,Age:f,Sib:g,Parch:h,Y:b})
            total_loss=+l
        print('loss',total_loss)
    w1,w2,w3,w4,w5,bi=sess.run([w1,w2,w3,w4,w5,bi])
    datafile='D:/Program Scripts/Machine Learning/datasets/titanic/test.xlsx'
    book = pd.ExcelFile(datafile)
    df=(book.parse('test'))
    df['Sex'].replace('female',0,inplace=True)
    df['Sex'].replace('male',1,inplace=True)
    df=df.dropna(how='any')
    df1=np.array(df)
    for a,b,c,d,e,f,g,h,i,j,k in df1:
        de=(w1*b+w2*d+w3*(e)+w4*f+w5*g+bi)
        x=math.exp(-de)
        y=x+1
        z=(1/y)
        print(math.log(z/(1-z)))
