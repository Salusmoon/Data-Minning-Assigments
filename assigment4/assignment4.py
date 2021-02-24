
import pandas as pd
import random 
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
import math  
from sklearn.utils.random import sample_without_replacement  
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from time import perf_counter 
from sklearn.datasets import make_moons
from random import randint

## Assigment 2.1
X, y = make_blobs(n_samples=20000, centers=5, n_features=2, random_state=0)


with PdfPages("OriginalData.pdf") as pdf:
  plt.title("Orginal data")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.scatter(X[:, 0], X[:, 1], c=y)
  pdf.savefig()
  plt.show()
  plt.close()


kmeans = KMeans(n_clusters=5, random_state=0,max_iter=1)
with PdfPages("KmeansDemonstration.pdf") as pdf:
  for i in range(4):
    kmeans.fit(X)
    Y_predict=kmeans.predict(X)
    plt.title("iteration"+ str(i))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(X[:, 0], X[:, 1], c=Y_predict, marker="x")
    pdf.savefig()
    plt.show()
    plt.close()
    intermediate_centers = kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=5,max_iter=1, init=intermediate_centers)

## Assigment 2.2
X, y = make_blobs(n_samples=20000, centers=5, n_features=2, random_state=0)


with PdfPages("ConvergenceCentroid.pdf") as pdf:
  plt.figure(figsize=(15,15))
  plt.title("Orginal data")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.scatter(X[:, 0], X[:, 1], c=y)
  pdf.savefig()
  plt.show()
  plt.close()


  kmeans = KMeans(n_clusters=5, random_state=0,max_iter=1)
  centers=[]
  for i in range(4):
    kmeans.fit(X)
    intermediate_centers = kmeans.cluster_centers_
    centers.append(intermediate_centers)
    kmeans = KMeans(n_clusters=5,max_iter=1, init=intermediate_centers)




  kmeans.fit(X)
  Y_predict=kmeans.predict(X)
  plt.figure(figsize=(20,20))
  plt.title("iteration: "+ str(i))
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.scatter(X[:, 0], X[:, 1], c=Y_predict, marker="x")

  for i in range(len(centers)):
    plt.scatter(centers[i][:,0], centers[i][:,1], c="black")


  pdf.savefig()
  plt.show()
  plt.close()

## Assigment 2.3.1
X, Y = make_blobs(n_samples=20000, centers=5, n_features=2, random_state=0)

array = np.arange(start=0, stop=20000, step=1)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3)                                       # random select rate


train_set, test_set = next(gss.split(X=X, y=Y, groups=array))                                            # split data randomly selected 70% traning 30% test with gss
X_train, X_test, y_train, y_test = X[train_set], X[test_set], Y[train_set], Y[test_set]   



X_train_copy = X_train
y_train_copy = y_train

with PdfPages("ClusterSampling.pdf") as pdf:
  
  #Original data plot
  plt.figure(figsize=(15,15))
  plt.title("Orginal Data")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.scatter(X_train_copy[:, 0], X_train_copy[:, 1], c=y_train_copy, marker="x")
  pdf.savefig()
  plt.show()
  plt.close()

  # 50 cluster plot
  kmeans = KMeans(n_clusters=40, random_state=0)
  kmeans.fit(X_train_copy)
  Y_predict= kmeans.predict(X_train_copy)
  plt.figure(figsize=(15,15))
  plt.title("Clustered data.Cluster : 40")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.scatter(X_train_copy[:, 0], X_train_copy[:, 1], c=Y_predict, marker="x")
  pdf.savefig()
  plt.show()
  plt.close()


  centers = kmeans.cluster_centers_


  np.cluster_density=[]
  for j in range(len(centers)):
    point= centers[j]
    distances=[]
    total=0
    for i in range(len(centers)):
      x= abs(centers[i][0]-point[0])
      y= abs(centers[i][1]-point[1])
      distance = math.sqrt((x*x)+(y*y))
      distances.append(distance)
    distances.sort()
    for number in range(10):
      total= total+distances[number+1]
    result= 1/(total/10)
    np.cluster_density.append(result)


  index=[]
  cluster=[]
  for i in range(10):
    maks= np.max(np.cluster_density)       
    mini= np.min(np.cluster_density)     
    ind= np.cluster_density.index(mini)   
    np.cluster_density[ind] = maks    
    index.append(ind)     
    cluster.append(mini)    

  data_index=[]
  for i in range(len(Y_predict)):   
    if Y_predict[i] in index:       
      data_index.append(i)

  # single-stage sampling
  sampling_x=X_train_copy[data_index]
  sampling_y= Y_predict[data_index]


  plt.figure(figsize=(15,15))
  plt.title("Single-stage sampling. Cluster : 10")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.scatter(sampling_x[:, 0], sampling_x[:, 1], c=sampling_y, marker="x")
  pdf.savefig()
  plt.show()
  plt.close()

  sample = sample_without_replacement(len(sampling_x),len(sampling_x)/2)
  #double stage sampling
  double_sample_x = sampling_x[sample]
  double_sample_y= sampling_y[sample]

  plt.figure(figsize=(15,15))
  plt.title("double-stage sampling")
  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.scatter(double_sample_x[:, 0], double_sample_x[:, 1], c=double_sample_y, marker="x")
  pdf.savefig()
  plt.show()
  plt.close()


mlp = MLPClassifier()

#Original data traininh
time_thread_start = perf_counter()  #start time 
mlp.fit(X_train_copy,y_train_copy)                            # Orginal data training
score=mlp.score(X_test, y_test)
time_thread_end = perf_counter()  # end time
time= time_thread_end-time_thread_start
print("(Original Data) Mean Testing Accuracy: ", score,  "Training Time: ", time ," ms")

#Single-stage training
time_thread_start = perf_counter()  #start time 
mlp.fit(sampling_x,sampling_y)
score=mlp.score(X_test, y_test)                            # Single-stage sampling data training
time_thread_end = perf_counter()  # end time
time= time_thread_end-time_thread_start
print("(Single-stage Clustering) Mean Testing ", score,  "Training Time: ", time ," ms")

#Double-stage training
time_thread_start = perf_counter()  #start time 
mlp.fit(double_sample_x,double_sample_y)                            # double-stage sampling data training
score=mlp.score(X_test, y_test)
time_thread_end = perf_counter()  # end time
time= time_thread_end-time_thread_start
print("(Double-stage Clustering) Mean Testing: ", score,  "Training Time: ", time ," ms")

## Assigment 3

with PdfPages("OutlierDetection.pdf") as pdf:

  noises=[0,0.05,0.1,0.25]    # noise levels
  for noise in range(len(noises)):
    
    X, y = make_moons(n_samples=200, noise=noises[noise])    

    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(X)
    y_predict= kmeans.predict(X)
    centers = kmeans.cluster_centers_
    
    # threshold 
    threshold=[]    # cluster  threshold value array
    for cluster in range(10):
      data_index=[]
      for j in range(len(y_predict)):
        if y_predict[j]== cluster:
          data_index.append(j)
      sampling_x=X[data_index]            # cluster X
      distances= []                         # cluster points distance for center
      for i in range(len(sampling_x)):
        x= abs(centers[cluster][0]-sampling_x[i][0])
        y= abs(centers[cluster][1]-sampling_x[i][1])
        distance = math.sqrt((x*x)+(y*y))
        distances.append(distance)           
      distances.sort()
      thresold_value = distances[-1]       # threshold value
      threshold.append(thresold_value)
    print("Setting Noise: ", noises[noise] )
    
    # random 5 points check through data
    random5 = []
    for i in range(5):
      ind = randint(0, 200)
      point_y = kmeans.predict([X[ind]])
      x1= abs(centers[point_y[0]][0]-X[ind][0])
      x2= abs(centers[point_y[0]][1]-X[ind][1]) 
      distance = math.sqrt((x1*x1)+(x2*x2))
      random5.append(X[ind])
      if distance >= threshold[point_y[0]]:
        print("point: ", X[ind][0], ", ", X[ind][1]," : ","It is outlier!")
      else:
        print("point: ", X[ind][0], ", ", X[ind][1]," : ","It is normal.")

    # random 5 points check randomly data
    random3=[]
    for i in range(3):
      random_point=[random.uniform(-1.5, 1.5), random.uniform(-1.0, 1.5)]
      point_y = kmeans.predict([random_point])
      x1= abs(centers[point_y[0]][0]-random_point[0])
      x2= abs(centers[point_y[0]][1]-random_point[1]) 
      distance = math.sqrt((x1*x1)+(x2*x2))
      random3.append(random_point)
      if distance >= threshold[point_y[0]]:
        print("point: ", random_point[0], ", ", random_point[1]," : ","It is outlier!")
      else:
        print("point: ", random_point[0], ", ", random_point[1]," : ","It is normal.")
    

    # draw plot
    markers=["x", "s","*"]
    labels=["data","center", "random 5 ", "random 3 "]
    plt.figure(figsize=(10,10))
    plt.title("Noise "+str(noises[noise]))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(X[:, 0], X[:, 1], c=y_predict, label=labels[0])
    plt.scatter(centers[:,0], centers[:,1], c="black", marker=markers[0] , label=labels[1])
    for i in range(5):
      plt.scatter(random5[i][0], random5[i][1], c="black", marker=markers[1], label=labels[2])
    for i in range(3):
      plt.scatter(random3[i][0], random3[i][1], c="black", marker=markers[2], label=labels[3])
    pdf.savefig()
    plt.legend()
    plt.show()
    plt.close()
