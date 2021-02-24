
import pandas as pd
import random 
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
from time import perf_counter 
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Assigment 2.1

def function(m, n,max):                        
  X , y = make_blobs(n_samples=m, n_features=n, centers=2)    ### m tuple size  dimension size n 
  array = np.arange(start=0, stop=m, step=1)   
  gss = GroupShuffleSplit(n_splits=10, test_size=0.3)
  clf = Perceptron(tol=1e-3, random_state=0, max_iter=max)  
  total_score=0
  total_time= 0
  for i in range(10):
    time_thread_start = perf_counter()  #start time 
    train_set, test_set = next(gss.split(X=X, y=y, groups=array))                                        # split data randomly selected 70% traning 30% test with gss
    X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]
    clf.fit(X_train, y_train)                                                                     # fit by trainX and trainy
    score = clf.score(X_test, y_test)
    time_thread_end = perf_counter()  # end time
    error= 1-score
    total_score = total_score+error 
    time= time_thread_end-time_thread_start
    total_time = total_time + time
  av_score = total_score/10
  av_time= (total_time/10)*1000
  print("average error : ",av_score)
  print("average time : ",av_time,"mili second")

function(10000,100,100)

function(10000,1000,100)

function(100000,100,100)

function(250000,100,100)

function(10000,100,500)

function(10000,1000,500)

function(100000,100,500)

function(250000,100,500)

def function2(m, n,max):                             # for assigment 2.2
  X , y = make_blobs(n_samples=m, n_features=n, centers=2)    ### m tuple size  dimension size n
  array = np.arange(start=0, stop=m, step=1)   
  gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
  clf = Perceptron(tol=1e-3, random_state=0, max_iter=max)  
  for i in range(1):
    train_set, test_set = next(gss.split(X=X, y=y, groups=array))                                        # split data randomly selected 70% traning 30% test with gss
    X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]
    clf.fit(X_train, y_train)

  
  
  zdata = X_test[:,0]
  xdata = X_test[:,1]
  ydata = X_test[:,2]

  z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

  tmp = np.linspace(-15,15,30)
  x,y = np.meshgrid(tmp,tmp)

  fig = plt.figure()
  ax  = fig.add_subplot(111, projection='3d')
  ax.scatter3D(xdata, ydata, zdata, c=y_test, cmap=plt.cm.Paired);
  ax.plot_surface(x, y, z(x,y))
  ax.set_xlabel("x1")
  ax.set_ylabel('x2')
  ax.set_zlabel('x3')
  fig.savefig('asigment2.2.png')
  plt.show()

function2(500,10,100)

digit_X,digit_y = load_digits(return_X_y=True)                                  # for assigment 3.2 
array = np.arange(start=0, stop=len(digit_X), step=1)

gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
clf = MLPClassifier(hidden_layer_sizes=50, max_iter=100)
train_set, test_set = next(gss.split(X=digit_X, y=digit_y, groups=array))
X_train, X_test, y_train, y_test = digit_X[train_set], digit_X[test_set], digit_y[train_set], digit_y[test_set]

error_array = []

for i in range(100):
  clf.partial_fit(X_train,y_train, classes=np.unique(y_train))                          
  score =clf.score(X_test,y_test) 
  error = 1-score
  error_array.append(error)


fig = plt.figure()
plt.plot(error_array)
plt.title("Convergence of error with MLP")
plt.xlabel("iteration")
plt.ylabel("error")
fig.savefig('asigment3.1.png')

def tuple_create(n):
  tupple=()
  for i in range(1,n):
    number = 2**i
    tupple = tupple+(number,)
  reversed(tupple)
  return tupple

digit_X,digit_y = load_digits(return_X_y=True)                                  # for assigment 3.2 
array = np.arange(start=0, stop=len(digit_X), step=1)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
error_array=[]
train_score_array=[]
error_array.append(None)
train_score_array.append(None)
for i in range(1,11):
  hidden_layer= tuple_create(i)
  clf = MLPClassifier(hidden_layer_sizes=(hidden_layer), max_iter=100)
  train_set, test_set = next(gss.split(X=digit_X, y=digit_y, groups=array))
  X_train, X_test, y_train, y_test = digit_X[train_set], digit_X[test_set], digit_y[train_set], digit_y[test_set]
  clf.fit(X_train,y_train)
  score =clf.score(X_test,y_test) 
  error = 1-score
  error_array.append(error)
  train_error = clf.score(X_train, y_train)
  train_score= 1-train_error
  train_score_array.append(train_score)

fig = plt.figure()
plt.plot(error_array,"-o" ,color="red", label="Test")
plt.plot(train_score_array,"-x", color="blue", label="Train")
plt.title("Train & test scores as a function of hidden layer size")
plt.xlabel('Hidden Layer Size')
plt.ylabel('Score')
plt.legend()
fig.savefig('asigment3.1.png')
plt.show()
