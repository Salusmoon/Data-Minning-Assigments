import pandas as pd
import random 
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from time import perf_counter 
from sklearn.decomposition import PCA
from sklearn.utils.random import sample_without_replacement  
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import seaborn as sns
from skimage.restoration import denoise_tv_chambolle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

def function(m, n):                             # for assigment 2.1
  X , y = make_blobs(n_samples=m, n_features=n)    ### m tuple size  dimension size n
  array = np.arange(start=0, stop=m, step=1)
  reg = make_pipeline(StandardScaler(),  SGDRegressor(max_iter=1000))     # regresion
  gss = GroupShuffleSplit(n_splits=5, test_size=0.3)                      # slipt for train and test
  total_score=0
  total_time= 0
  for i in range(5):
    time_thread_start = perf_counter()  #start time 
    train_set, test_set = next(gss.split(X=X, y=y, groups=array))                                        # split data randomly selected 70% traning 30% test with gss
    X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    if score < 0:                                             # if score will be negative , I change score 
      score = 0
    error = 1-score
    total_score = total_score+error 
    time_thread_end = perf_counter()  # end time
    time= time_thread_end-time_thread_start
    total_time = total_time + time
  av_score = total_score/5
  av_time= (total_time/5)*1000
  print("average error : ",av_score)
  print("average time : ",av_time,"mili second")

function(10000, 100)          # 10000 tuple size  dimension size 100

function(10000, 1000)       # 10000 tuple size  dimension size 1000

function(10000, 2000)       # 10000 tuple size  dimension size 2000

function(100000, 100)         # 100000 tuple size  dimension size 100

function(250000, 100)         # 250000 tuple size  dimension size 100

function(500000, 100)       # 500000 tuple size  dimension size 100

def function2(m, n, number):                       # for assigment 2.2  ## mtuple size 10000 dimension size n
  array = np.arange(start=0, stop=m, step=1)
  X , y = make_blobs(n_samples=m, n_features=n)    
  array = np.arange(start=0, stop=m, step=1)
  pca = PCA(n_components=number)                                               # pca change dimension size 
  reg = make_pipeline(StandardScaler(),  SGDRegressor(max_iter=1000, tol=1e-3))   # regresion
  gss = GroupShuffleSplit(n_splits=5, test_size=0.3)                              # split for train and test 
  newX = pca.fit_transform(X)                                                  # create new X for given pca dimension size
  total_score = 0
  total_time = 0
  new_score = 0
  for i in range(5):  
      time_thread_start = perf_counter()  #start time 
      train_set, test_set = next(gss.split(X=newX, y=y, groups=array))                     # split data randomly selected 70% traning 30% test with gss
      X_train, X_test, y_train, y_test = newX[train_set], newX[test_set], y[train_set], y[test_set]
      reg.fit(X_train, y_train)
      score = reg.score(X_test, y_test)
      if score < 0:                                                                  # if score will be negative , I change score 
        score = 0
      error = 1-score
      total_score = total_score+error 
      new_score = new_score+ score
      time_thread_end = perf_counter()  # end time
      time= time_thread_end-time_thread_start
      total_time = total_time + time
  av_score = total_score/5
  av_time= total_time/5
  print((new_score))
  print("average error : ",av_score)
  print("average time : ",av_time,"second")

function2(10000, 2000, 500)               # 10000 tuple size  dimension size 2000 for 500 dimension

function2(10000, 2000,  100)          # 10000 tuple size  dimension size 2000 for 100 dimension

function2(10000, 2000,  10)         # 10000 tuple size  dimension size 2000 for 10 dimension

function2(10000, 2000,  4)          # 10000 tuple size  dimension size 2000 for 4 dimension

function2(10000, 2000,  1)          # 10000 tuple size  dimension size 2000 for 1 dimension

def function3(m, n, number):                                 # for assigment 2.1 
  X , y = make_blobs(n_samples=m, n_features=n)    ###mtuple size n dimension size 
  sample = sample_without_replacement(m,number)                                  # sample change tuple size
  reg = make_pipeline(StandardScaler(),  SGDRegressor(max_iter=1000, tol=1e-3))         # regresion
  gss = GroupShuffleSplit(n_splits=5, test_size=0.3)                                  # split for train and test
  newX = X[sample]                                                                   #create new X for given sample tuple size
  newY = y[sample]                                                                   # create new y for given sample tuple size
  array = np.arange(start=0, stop=number, step=1)
  total_score=0
  total_time=0
  for i in range(5):  
      time_thread_start = perf_counter()  #start time
      train_set, test_set = next(gss.split(X=newX, y=newY, groups=array))                        # split data randomly selected 70% traning 30% test with gss
      X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]
      reg.fit(X_train, y_train)
      score = reg.score(X_test, y_test)
      if score < 0:                                                                  # if score will be negative , I change score 
        score = 0
      error = 1-score
      total_score = total_score+error 
      time_thread_end = perf_counter()  # end time
      time= time_thread_end-time_thread_start
      total_time = total_time + time
  av_score = total_score/5
  av_time= total_time/5
  print("average error : ",av_score)
  print("average time : ",av_time,"second")

function3(500000, 100, 300000)          # 500000 tuple size  dimension size 100 for 300000 tuple

function3(500000, 100, 150000)          # 500000 tuple size  dimension size 100 for 150000 tuple

function3(500000, 100, 100000)          # 500000 tuple size  dimension size 100 for 100000 tuple

function3(500000, 100, 1000)          # 500000 tuple size  dimension size 100 for 1000 tuple

function3(500000, 100, 100)             # 500000 tuple size  dimension size 100 for 100 tuple

X, y = make_moons(n_samples=200)                                          # for assigment 3.1 
array = np.arange(start=0, stop=200, step=1)

gss = GroupShuffleSplit(n_splits=5, test_size=0.3)

train_set, test_set = next(gss.split(X=X, y=y, groups=array))                                            # split data randomly selected 70% traning 30% test with gss
X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]                


sgdc = SGDClassifier(loss= "log", max_iter=1000, tol=0.01)                                     # SGDC classifier method (logistic regression)

sgdc.fit(X_train, y_train)                                                                    # classify

with PdfPages("BinaryClassVisualization.pdf") as pdf:



  plt.title("Train")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.scatter(X_train[:,0], X_train[:,1],c=y_train,cmap=plt.cm.Paired)                       # plot for X_train set and colur by these class
  pdf.savefig()
  plt.close()

  xx = np.linspace(-1, 2, 10)
  yy = np.linspace(-1, 1, 10)

  X1, X2 = np.meshgrid(xx, yy)
  Z = np.empty(X1.shape)
  for (i, j), val in np.ndenumerate(X1):                                                 # we find hypotsesis line , other 2 lines are upper and lower size
      x1 = val
      x2 = X2[i, j]
      p = sgdc.decision_function([[x1, x2]])
      Z[i, j] = p[0]
  levels = [-1.0, 0.0, 1.0]
  linestyles = ['dashed', 'solid', 'dashed']
  colors = 'k'

  plt.title("Test")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
  plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.cm.Paired)
  pdf.savefig()
  plt.close()

digit_X,digit_y = load_digits(return_X_y=True)                                  # for assigment 3.2 
array = np.arange(start=0, stop=len(digit_X), step=1)

gss = GroupShuffleSplit(n_splits=5, test_size=0.3)
reg = make_pipeline(StandardScaler(),  SGDRegressor(max_iter=1000, tol=1e-3))
train_set, test_set = next(gss.split(X=digit_X, y=digit_y, groups=array))
X_train, X_test, y_train, y_test = digit_X[train_set], digit_X[test_set], digit_y[train_set], digit_y[test_set]


def function4(percent, targetX):                                                     #this function make noisy to given percentage
    percent = percent/100
    sample = sample_without_replacement(len(targetX),percent*len(targetX))
    change = random.sample(range(0,64),10)
    for i in change:
        targetX[sample][i] = abs(targetX[sample][i]-16)
    
    return targetX

testX = function4(25,X_test)
reg.fit(X_train, y_train)
score = reg.score(testX, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

testX = function4(50,X_test)
reg.fit(X_train, y_train)
score = reg.score(testX, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

testX = function4(75,X_test)
reg.fit(X_train, y_train)
score = reg.score(testX, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

trainX = function4(25,X_train)
reg.fit(trainX, y_train)
score = reg.score(X_test, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

trainX = function4(50,X_train)
reg.fit(trainX, y_train)
score = reg.score(X_test, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

trainX = function4(75,X_train)
reg.fit(trainX, y_train)
score = reg.score(X_test, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

testX = function4(25,X_test)
trainX = function4(25,X_train)
reg.fit(trainX, y_train)
score = reg.score(testX, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

testX = function4(50,X_test)
trainX = function4(50,X_train)
reg.fit(trainX, y_train)
score = reg.score(testX, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

testX = function4(75,X_test)
trainX = function4(75,X_train)
reg.fit(trainX, y_train)
score = reg.score(testX, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

testX = function4(50,X_test)                                      # denoised form of H
trainX = function4(50,X_train)

for i in range(len(testX)):
  testX[i] = denoise_tv_chambolle(testX[i], weight=0.5)

for i in range(len(trainX)):
  trainX[i] = denoise_tv_chambolle(trainX[i], weight=0.5)

reg.fit(trainX, y_train)
score = reg.score(testX, y_test)
if score < 0:
    score = 0
error = 1-score
print(error)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)

plt.gray()

ax[0].set_title("orginal")
ax[0].imshow(digit_X.reshape(-1,8,8)[8],cmap=plt.cm.gray_r,interpolation='nearest')            # image  show

change = random.sample(range(0,64),10)
for i in change:
    digit_X[8][i] = abs(digit_X[8][i]-16)

ax[1].set_title("noisy")
ax[1].imshow(digit_X.reshape(-1,8,8)[8],cmap=plt.cm.gray_r,interpolation='nearest')               # image  show


fig.savefig("Noising.pdf", bbox_inches='tight')

digit_X,digit_y = load_digits(return_X_y=True)                                          # for assigment 3.3
array = np.arange(start=0, stop=len(digit_X), step=1)

gss = GroupShuffleSplit(n_splits=5, test_size=0.3)
train_set, test_set = next(gss.split(X=digit_X, y=digit_y, groups=array))
X_train, X_test, y_train, y_test = digit_X[train_set], digit_X[test_set], digit_y[train_set], digit_y[test_set]
sgdc = SGDClassifier(loss= "log", max_iter=10000, tol=0.01)

sgdc.fit(X_train, y_train)
y_pred = sgdc.predict(X_test)                                                       # we predict to data for X_test
conf_mat = confusion_matrix(y_test, y_pred)                                   
conf_mat

table =plt.figure()
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
table.savefig("ConfusionMatrixHeatmap.pdf", bbox_inches='tight')





