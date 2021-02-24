import pandas as pd
import random 
import numpy as np
from random import randint
from sklearn.datasets import load_digits
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from skimage.util import random_noise
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_moons
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

## 2.1.1 Assigment
digit_X,digit_y = load_digits(return_X_y=True)  
array = np.arange(start=0, stop=len(digit_X), step=1)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
hidden_layers = (16, 8, 4,2)
 
 
for i in range(1):
  mlp = MLPClassifier(hidden_layer_sizes=(hidden_layers), max_iter=1000)              # mlp classification
  train_set, test_set = next(gss.split(X=digit_X, y=digit_y, groups=array))
  X_train, X_test, y_train, y_test = digit_X[train_set], digit_X[test_set], digit_y[train_set], digit_y[test_set]
  clf = BaggingClassifier(base_estimator=mlp,n_estimators=8, random_state=0)          # Bagging classification with 8 estimator
  clf.fit(X_train, y_train)
  row1=0
  row2=157
  i=1
  for estimator in clf.estimators_:                                               # each estimator classifies one of the 8 pieces of data.
    if i==8:
      row2=row2+1
    subsetX= X_train[row1:row2]                       # subset Datax(data/8)
    subsety= y_train[row1:row2]                        # subset Datay(data/8)
    clf.fit(subsetX,subsety)
    score =clf.score(X_test,y_test)
    number = 540*score
    print(int(number) ," out of 540 instances are correctly classified by learner #",i)
    row1= row1+157
    row2=row2+157
    i = i+1
  
  print("-----------------------------------------------")
  clf.fit(X_train, y_train)
  score =clf.score(X_test, y_test)
  number = 540*score
  print(int(number) ," out of 540 instances are correctly classified by bagging")

##for 2.1.2 Assigment
 
X, y = make_moons(n_samples=200)                                         
array = np.arange(start=0, stop=200, step=1)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3)                                       # random select rate
for i in range(X.shape[0]):
    X[i] = random_noise(X[i], mode='gaussian', var=0.2)                            #Add gaussian noise deviation value of 0.2.
 
train_set, test_set = next(gss.split(X=X, y=y, groups=array))                                            # split data randomly selected 70% traning 30% test with gss
X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]   
 
sgdc = SGDClassifier(loss="log", penalty="l2")                             # SGD solver with log
 
clf = AdaBoostClassifier(base_estimator=sgdc,n_estimators=4, random_state=0)  # ada bosst with sgs solver
clf.fit(X_train, y_train)
 
learner = 1
 
with PdfPages("BaseLearnerVisualization.pdf") as pdf:
 
  for estimator in clf.estimators_:
    #boundery
    xx = np.linspace(-1.5, 2, 10)
    yy = np.linspace(-1.5, 1.5, 10)
 
    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):                                                 # we find hypotsesis line , other 2 lines are upper and lower size
      x1 = val
      x2 = X2[i, j]
      p = clf.decision_function([[x1, x2]])
      Z[i, j] = p[0]
    levels = [-1.0, 0.0, 1.0]
    linestyles = ['dashed', 'solid', 'dashed']
    colors = 'k'
 
    plt.title("Learner"+str(learner))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.cm.Paired)
    learner= learner+1
    pdf.savefig()
    plt.close()

## 2.2 Assigment

dataX,dataY = load_breast_cancer(return_X_y=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
array = np.arange(start=0, stop=len(dataX), step=1)
train_set, test_set = next(gss.split(X=dataX, y=dataY, groups=array))
X_train, X_test, y_train, y_test = dataX[train_set], dataX[test_set], dataY[train_set], dataY[test_set]


# all classifier and score
sgdc = SGDClassifier(loss="log", penalty="l2")
sgdc.fit(X_train,y_train)
sgdc_scores = cross_val_score(sgdc, X_test, y_test, cv=5)
print("Accuracy obtained by learner SGDC is: ", sgdc_scores.mean())
mlp = MLPClassifier()
mlp.fit(X_train,y_train)
mlp_scores = cross_val_score(mlp, X_test, y_test, cv=5)
print("Accuracy obtained by learner MLP is: ", mlp_scores.mean())
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)
neigh_scores = cross_val_score(neigh, X_test, y_test, cv=5)
print("Accuracy obtained by learner KNN is: ", scores.mean())
print("-----------------------------------------------------")

# select which classifeir most consistent
if (sgdc_scores.mean() >= mlp_scores.mean()) and (sgdc_scores.mean() >= neigh_scores.mean()):
  bag = BaggingClassifier(base_estimator=sgdc,n_estimators=10, random_state=0)
  bag.fit(X_train, y_train)
  scores = cross_val_score(bag, X_test, y_test, cv=5)
  print("sgdc")
elif (mlp_scores.mean() >= sgdc_scores.mean()) and (mlp_scores.mean() >= neigh_scores.mean()):
  bag = BaggingClassifier(base_estimator=mlp,n_estimators=10, random_state=0)
  bag.fit(X_train, y_train)
  scores = cross_val_score(bag, X_test, y_test, cv=5)
  print("mlp")
else:
  bag = BaggingClassifier(base_estimator=neigh,n_estimators=10, random_state=0)
  bag.fit(X_train, y_train)
  scores = cross_val_score(bag, X_test, y_test, cv=5)
  print("neigh")
print("Accuracy obtained by ensemble learner is: ", scores.mean())

## hidden layer create method for 2.3
def tuple_create(n):                               
  tupple=()
  for i in range(1,n):
    number = 2**i
    tupple = tupple+(number,)
  reversed(tupple)
  return tupple

## 2.3 Assigment
dataX,dataY = load_breast_cancer(return_X_y=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
array = np.arange(start=0, stop=len(dataX), step=1)
train_set, test_set = next(gss.split(X=dataX, y=dataY, groups=array))
X_train, X_test, y_train, y_test = dataX[train_set], dataX[test_set], dataY[train_set], dataY[test_set]
scores=[]             # the array the scores will be in
bigest=[0]            # greatest value
for i in range(1,11):
  hidden_layer= tuple_create(i)
  mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer), max_iter=1000)   # Multi-layer Perceptron
  mlp.fit(X_train, y_train)
  score = mlp.score(X_test,y_test)
  scores.append(score)
  if score >= bigest[0]:                   # if the incoming score is greater than the value in the biggest array, it puts it in the biggest search
    bigest[0]= score                        
    index = i
  print("Parameter setting: l#",i, "Accuracy: ", score)
print("-----------------------")

hidden_layer= tuple_create(index)
mlp2 = MLPClassifier(hidden_layer_sizes=(hidden_layer), max_iter=1000)
bag = BaggingClassifier(base_estimator=mlp2,n_estimators=10, random_state=0)
bag.fit(X_train, y_train)
bag_score = bag.score(X_test,y_test)
print(index)
print("Ensemble Learning Accuracy: ", bag_score)

## 3 Assigment

X, y = make_moons(n_samples=200)                                          # for assigment 3.1 
array = np.arange(start=0, stop=200, step=1)
for i in range(X.shape[0]):
    X[i] = random_noise(X[i], mode='gaussian', var=0.3)                            #Add gaussian noise deviation value of 0.2.

gss = GroupShuffleSplit(n_splits=1, test_size=0.02)
array = np.arange(start=0, stop=len(X), step=1)
train_set, test_set = next(gss.split(X=X, y=y, groups=array))
X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]
neigh = KNeighborsClassifier(n_neighbors=5)     # knn classifier
neigh.fit(X_train, y_train)
pred = neigh.predict(X_test)
markers = ['x','.']
color=["blue","red"]
with PdfPages("kNN.pdf") as pdf:
  for i in range(4):                            # for test data

    for classes in range(2):                            # for class mark and color
          index= np.where(y_train == classes)       
          plt.scatter(X_train[index, 0], X_train[index, 1], c=color[classes],marker=markers[classes])
    plt.scatter(X_test[i][0], X_test[i][1],marker="+",c="green")
    plt.title("Test object #"+str(i)+" pred class: "+str(pred[i]))
    pdf.savefig()
    plt.close()









