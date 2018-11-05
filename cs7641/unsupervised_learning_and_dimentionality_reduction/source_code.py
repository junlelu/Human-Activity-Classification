import csv
import math
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd

class assignment4:
  def __init__(self):
    # data processing
    self.dataSetPath = './data_set/'
    self.dataSetName = ""
    self.csv_delimiter = ','
    self.data = None
    self.allFeatures = []
    self.allTarget = []

    # not used
    self.XTrain = None
    self.XTest = None
    self.YTrain = None
    self.YTest = None

    # k-mean clustering
    self.kNum = range(1,21)
    self.kmean = None
    self.kmeanRD = None
    # expectation maximization
    self.em = None
    self.emRD = None
    # PCA
    self.pca = None
    self.pcaDims = range(1,21)

    # ICA
    self.icaDims = range(1,21)
    self.ica = None

    # RP
    self.rp = None
    self.rpDims = range(1,21)

    # TSVD
    self.tsvd = None
    self.tsvdDims = range(1,10)

  def read_data_voice(self,dataName):
    with open(self.dataSetPath+dataName, 'r', encoding="utf8") as file:
        reader = csv.reader(file,delimiter=self.csv_delimiter)
        self.data = list(reader)
    print("Reading data set: '{}'".format(self.dataSetPath+dataName))
    print('Number of instances: {}'.format(len(self.data)))
    print('Number of attributes: {}'.format(len(self.data[0])-1))

  def read_data_haptX(self,dataName):
    self.data = None
    with open(self.dataSetPath+dataName, 'r', encoding="utf8") as file:
        reader = csv.reader(file,delimiter=',')
        self.data = list(reader)

    print(len(self.data)) 
    for elim in self.data:
      feature = []
      for i in elim:
          feature.append(i)
      self.allFeatures.append(feature)
    print("Reading data set: '{}'".format(self.dataSetPath+dataName))
    print('Number of instances: {}'.format(len(self.allFeatures)))
    print('Number of attributes: {}'.format(len(self.allFeatures[0])))
  def read_data_haptY(self,dataName):
    self.data = None
    with open(self.dataSetPath+dataName, 'r', encoding="utf8") as file:
        reader = csv.reader(file,delimiter=',')
        self.data = list(reader)
    for elim in self.data:
      self.allTarget.append(elim)
    print("Reading data set: '{}'".format(self.dataSetPath+dataName))
    print('Number of instances: {}'.format(len(self.allTarget)))
    print('Number of attributes: {}'.format(len(self.allTarget[0])))

    self.allFeatures = np.asarray(self.allFeatures,dtype=np.float32)
    self.allTarget = np.asarray(self.allTarget,dtype=np.float32)
    self.allTarget = self.allTarget.ravel()
    
  def split_data_to_train_test(self,testSize=0.3):
    # in case the data set are very different in format
    sample_len = len(self.data[0])
    for elem in self.data:
        feature = elem[0:sample_len-1]
        feature_vector = []
        for f in feature:
            feature_vector.append(float(f))
        self.allFeatures.append(feature_vector)
        if elem[-1] == '0':
            val = 0
        else:
            val = 1
        self.allTarget.append((float(val)))
    self.allFeatures = np.asarray(self.allFeatures,dtype=np.float32)
    self.allTarget = np.asarray(self.allTarget,dtype=np.float32)
    self.XTrain,self.XTest,self.YTrain,self.YTest = train_test_split(self.allFeatures,self.allTarget,test_size = testSize, random_state=42) 
    print('Total X train data -> {}%'.format(int((len(self.XTrain)/len(self.data))*100)),'Size:',len(self.XTrain))
    print('Total X test data -> {}%'.format(int((len(self.XTest)/len(self.data))*100)),'Size:',len(self.XTest))
    print('Total Y train data -> {}%'.format(int((len(self.YTrain)/len(self.data))*100)),'Size:',len(self.YTrain))
    print('Total Y test data -> {}%'.format(int((len(self.YTest)/len(self.data))*100)),'Size',len(self.YTest))

  def get_max_idx(self,input):
    maxVal = input[0]
    maxIdx = 0
    for i in range(1,len(input)):
      if input[i] > maxVal:
        maxIdx = i
        maxVal = input[i]
    return maxIdx

  def pairwiseDistCorr(self,X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

  def k_mean_cluster(self):
    print("-"*50)
    print('{}: K-mean clustering'.format(self.dataSetName))
    
    dataX = StandardScaler().fit_transform(self.allFeatures)
    scores = []
    confusionMatrix = []
    self.kmean = KMeans(random_state=5, max_iter=1000)
    for i in self.kNum:
      self.kmean.set_params(n_clusters=i)
      self.kmean.fit(dataX)
      scores.append(sm.accuracy_score(self.allTarget,self.kmean.labels_))
      confusionMatrix.append(sm.confusion_matrix(self.allTarget,self.kmean.labels_ ))
    bestScoreIdx = self.get_max_idx(scores) 
    print("Accuracy score:{0:.2f}".format(scores[bestScoreIdx]))
    print("Confusion Matrix:",confusionMatrix[bestScoreIdx])

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('# of Clusters')
    plt.title('K-mean Cluster ({})'.format(self.dataSetName))

    plt.style.context('seaborn-whitegrid')
    plt.xticks(self.kNum)
    plt.plot(self.kNum,scores)
    plt.grid()
    plt.draw()
    plt.savefig('./{}_KMEAN.png'.format(self.dataSetName)) 
    print("-"*50)

  def k_mean_cluster_reduced(self,n_clusters,reduced_data,name):
    print("-"*50)
    print('{}: K-mean clustering {}'.format(self.dataSetName,name))
    dataX = StandardScaler().fit_transform(self.allFeatures) 
    self.kmeanRD = KMeans(n_clusters=n_clusters, random_state=5, max_iter=1000)
    self.kmeanRD.fit(reduced_data)

    print("Accuracy score:{0:.2f}".format(sm.accuracy_score(self.allTarget, self.kmeanRD.labels_)))
    print("Confusion Matrix:")
    print(sm.confusion_matrix(self.allTarget,self.kmeanRD.labels_))

    print("-"*50)

  def expectation_maximization_reduced(self,n_components,reduced_data,name):
    print("-"*50)
    print('{}: Expectation maximization {}'.format(self.dataSetName,name))
     
    self.emRD = GaussianMixture(n_components=n_components,random_state=5)
    self.emRD.fit(reduced_data)
    y_predict = self.emRD.predict(reduced_data)

    print("Accuracy score:{0:.2f}".format(sm.accuracy_score(self.allTarget, y_predict)))
    print("Confusion Matrix:")
    print(sm.confusion_matrix(self.allTarget,y_predict))
    print("-"*50)

  def expectation_maximization(self):
    print("-"*50)
    print('{}: Expectation maximization'.format(self.dataSetName))
    dataX = StandardScaler().fit_transform(self.allFeatures)
    scores = []
    confusionMatrix = []
    self.em = GaussianMixture(random_state=5)
    for i in self.kNum:
      self.em.set_params(n_components=i)
      self.em.fit(dataX)
      y_predict = self.em.predict(dataX)
      scores.append(sm.accuracy_score(self.allTarget, y_predict))
      confusionMatrix.append(sm.confusion_matrix(self.allTarget,y_predict))

    bestScoreIdx = self.get_max_idx(scores)
    print("Accuracy score:{0:.2f}".format(scores[bestScoreIdx]))
    print("Confusion Matrix:")
    print(confusionMatrix[bestScoreIdx])

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('# of Clusters')
    plt.title('Expectation Maximum Cluster ({})'.format(self.dataSetName))

    plt.style.context('seaborn-whitegrid')
    plt.xticks(self.kNum)
    plt.plot(self.kNum,scores)
    plt.grid()
    plt.draw()
    plt.savefig('./{}_EM.png'.format(self.dataSetName)) 
    print("-"*50)

  def PCA(self):
    print("-"*50)
    print('{}: Principal component analysis '.format(self.dataSetName))    
    
    dataX = StandardScaler().fit_transform(self.allFeatures)

    self.pca = PCA(random_state=5)
    grid ={'pca__n_components':self.pcaDims}
    mlp = MLPClassifier(max_iter=2000,alpha=1e-5,early_stopping=False,random_state=5,hidden_layer_sizes=[17]*11)
    pipe = Pipeline([('pca',self.pca),('NN',mlp)])
    search = GridSearchCV(pipe,grid,verbose=2,cv=5)
    search.fit(dataX, self.allTarget)

    print("Best number PCA components:",search.best_params_)
    
    self.pca.fit(dataX)
    var=np.cumsum(np.round(self.pca.explained_variance_ratio_, decimals=3)*100) 

    plt.figure()
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis ({})'.format(self.dataSetName))
    plt.xticks(self.pcaDims)
    plt.style.context('seaborn-whitegrid')
    plt.plot(var)
    plt.grid()
    plt.draw()
    plt.savefig('./{}_PCA_VA.png'.format(self.dataSetName)) 

    plt.figure()
    plt.ylabel('Score')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis Grid Search ({})'.format(self.dataSetName))
    plt.xticks(self.pcaDims)
    plt.ylim([0,1]) 
    plt.style.context('seaborn-whitegrid')
    plt.plot(self.pcaDims,search.cv_results_['mean_test_score'])
    plt.grid()
    plt.draw()
    plt.savefig('./{}_PCA_GS.png'.format(self.dataSetName))

    print("-"*50)

  def ICA(self):
    print("-"*50)
    print('{}: Independent component analysis '.format(self.dataSetName))    

    dataX = StandardScaler().fit_transform(self.allFeatures)
    self.ica = FastICA(random_state=5,max_iter =6000)
    # kurtosis
    kurt = []
    for dim in self.icaDims:
      self.ica.set_params(n_components=dim)
      tmp = self.ica.fit_transform(dataX)
      tmp = pd.DataFrame(tmp)
      tmp = tmp.kurt(axis=0)
      kurt.append(tmp.abs().mean())

    # grid search
    grid ={'ica__n_components':self.icaDims}
    mlp = MLPClassifier(max_iter=2000,alpha=1e-5,early_stopping=False,random_state=5,hidden_layer_sizes=[17]*11)
    pipe = Pipeline([('ica',self.ica),('NN',mlp)])
    search = GridSearchCV(pipe,grid,verbose=2,cv=5)
    search.fit(dataX, self.allTarget)
    print("Best number ICA components:",search.best_params_)

    plt.figure()
    plt.ylabel('Kurtosis')
    plt.xlabel('# of Features')
    plt.title('ICA Analysis ({})'.format(self.dataSetName))
    plt.xticks(self.icaDims)
    plt.style.context('seaborn-whitegrid')
    plt.plot(kurt)
    plt.grid()
    plt.draw()
    plt.savefig('./{}_kurtosis.png'.format(self.dataSetName))

    plt.figure()
    plt.ylabel('Score')
    plt.xlabel('# of Features')
    plt.title('ICA Analysis Grid Search ({})'.format(self.dataSetName))
    plt.xticks(self.icaDims)
    plt.style.context('seaborn-whitegrid')
    plt.plot(self.icaDims,search.cv_results_['mean_test_score'])
    plt.grid()
    plt.draw()
    plt.savefig('./{}_ICA_GS.png'.format(self.dataSetName))
    print("-"*50)

  def RP(self):
    print("-"*50)
    print('{}: Random Projection'.format(self.dataSetName))
    dataX = StandardScaler().fit_transform(self.allFeatures)
    disCorr = []
    self.rp = SparseRandomProjection(random_state=5)
    for dim in self.rpDims:
      self.rp.set_params(n_components=dim)
      disCorr.append(self.pairwiseDistCorr(self.rp.fit_transform(dataX), dataX))
    print(disCorr)

    # grid search
    grid ={'rp__n_components':self.rpDims}
    mlp = MLPClassifier(max_iter=2000,alpha=1e-5,early_stopping=False,random_state=5,hidden_layer_sizes=[17]*11)
    pipe = Pipeline([('rp',self.rp),('NN',mlp)])
    search = GridSearchCV(pipe,grid,verbose=2,cv=5)
    search.fit(dataX, self.allTarget)
    print("Best number RP components:",search.best_params_)

    plt.figure()
    plt.ylabel('Distance')
    plt.xlabel('# of Features')
    plt.title('RP Analysis ({})'.format(self.dataSetName))
    plt.xticks(self.rpDims)
    plt.style.context('seaborn-whitegrid')
    plt.plot(disCorr)
    plt.grid()
    plt.draw()
    plt.savefig('./{}_distance.png'.format(self.dataSetName))

    plt.figure()
    plt.ylabel('Score')
    plt.xlabel('# of Features')
    plt.title('RP Analysis Grid Search ({})'.format(self.dataSetName))
    plt.xticks(self.rpDims)
    plt.style.context('seaborn-whitegrid')
    plt.plot(search.cv_results_['mean_test_score'])
    plt.grid()
    plt.draw()
    plt.savefig('./{}_RP_GS.png'.format(self.dataSetName))
    print("-"*50) 

  def TSVD(self):
    print("-"*50)
    print('{}: TruncatedSVD'.format(self.dataSetName))
    dataX = StandardScaler().fit_transform(self.allFeatures)
    self.tsvd = TruncatedSVD(random_state=5)

    # grid search
    grid ={'tsvd__n_components':self.tsvdDims}
    mlp = MLPClassifier(max_iter=2000,alpha=1e-5,early_stopping=False,random_state=5,hidden_layer_sizes=[17]*11)
    pipe = Pipeline([('tsvd',self.tsvd),('NN',mlp)])
    search = GridSearchCV(pipe,grid,verbose=2,cv=5)
    search.fit(dataX, self.allTarget)
    print("Best number TSVD components:",search.best_params_)

    self.tsvd.fit(dataX)
    var=np.cumsum(np.round(self.tsvd.explained_variance_ratio_, decimals=3)*100) 

    plt.figure()
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('TSVD Analysis ({})'.format(self.dataSetName))
    plt.xticks(self.tsvdDims)
    plt.style.context('seaborn-whitegrid')
    plt.plot(var)
    plt.grid()
    plt.draw()
    plt.savefig('./{}_TSD_VA.png'.format(self.dataSetName))

    plt.figure()
    plt.ylabel('Score')
    plt.xlabel('# of Features')
    plt.title('TSVD Analysis Grid Search ({})'.format(self.dataSetName))
    plt.xticks(self.tsvdDims)
    plt.style.context('seaborn-whitegrid')
    plt.plot(search.cv_results_['mean_test_score'])
    plt.grid()
    plt.draw()
    plt.savefig('./{}_TSVD_GS.png'.format(self.dataSetName))
    print("-"*50) 
def main():
  runit = 1
  if runit:
    run = assignment4()
    run.read_data_voice('voice.csv')
    run.dataSetName = 'Voice'
    run.split_data_to_train_test(testSize=0.3)
    dataX = StandardScaler().fit_transform(run.allFeatures)

    ''' 
    run.PCA()
    run.ICA()
    run.RP()
    '''
    run.TSVD()
    run.k_mean_cluster()
    run.expectation_maximization()
    pcaCom = 15
    icaCom = 15
    rpCom = 15
    tsvdCom = 15
    k = 2
    reducedDataPCA = PCA(n_components=pcaCom,random_state=5).fit_transform(dataX)
    run.k_mean_cluster_reduced(k,reducedDataPCA,'PCA')
    run.expectation_maximization_reduced(k,reducedDataPCA,'PCA')

    reducedDataICA = FastICA(n_components=icaCom,random_state=5).fit_transform(dataX)
    run.k_mean_cluster_reduced(k,reducedDataICA,'ICA')
    run.expectation_maximization_reduced(k,reducedDataICA,'ICA') 

    reducedDataRP = SparseRandomProjection(n_components=rpCom,random_state=5).fit_transform(dataX)
    run.k_mean_cluster_reduced(k,reducedDataRP,'RP')
    run.expectation_maximization_reduced(k,reducedDataRP,'RP') 

    reducedDataTSVD = TruncatedSVD(random_state=5,n_components=tsvdCom).fit_transform(dataX)
    run.k_mean_cluster_reduced(k,reducedDataTSVD,'TSVD')
    run.expectation_maximization_reduced(k,reducedDataTSVD,'TSVD') 

  run_hapt = assignment4()
  run_hapt.read_data_haptX('HAPT_X.csv')
  run_hapt.read_data_haptY('HAPT_Y.csv')
  run_hapt.dataSetName = 'HAPT'
  dataX = StandardScaler().fit_transform(run_hapt.allFeatures)

  run_hapt.kNum = range(1,20,5)
  run_hapt.pcaDims = range(1,561,25)
  run_hapt.icaDims = range(1,561,25)
  run_hapt.rpDims = range(1,561,25)
  run_hapt.tvsdDims = range(1,561,25)

  #run_hapt.k_mean_cluster()
  run_hapt.expectation_maximization()

  run_hapt.PCA()
  run_hapt.ICA()
  run_hapt.RP()
  run_hapt.TSVD()



  pcaCom = 15
  icaCom = 15
  rpCom = 15
  tsvdCom = 15
  k = 2
  reducedDataPCA = PCA(n_components=pcaCom,random_state=5).fit_transform(dataX)
  run_hapt.k_mean_cluster_reduced(k,reducedDataPCA,'PCA')
  run_hapt.expectation_maximization_reduced(k,reducedDataPCA,'PCA')

  reducedDataICA = FastICA(n_components=icaCom,random_state=5).fit_transform(dataX)
  run_hapt.k_mean_cluster_reduced(k,reducedDataICA,'ICA')
  run_hapt.expectation_maximization_reduced(k,reducedDataICA,'ICA') 

  reducedDataRP = SparseRandomProjection(n_components=rpCom,random_state=5).fit_transform(dataX)
  run_hapt.k_mean_cluster_reduced(k,reducedDataRP,'RP')
  run_hapt.expectation_maximization_reduced(k,reducedDataRP,'RP') 

  reducedDataTSVD = TruncatedSVD(random_state=5,n_components=tsvdCom).fit_transform(dataX)
  run_hapt.k_mean_cluster_reduced(k,reducedDataTSVD,'TSVD')
  run_hapt.expectation_maximization_reduced(k,reducedDataTSVD,'TSVD') 

  

  print("All done")
  plt.show()
if __name__ == "__main__":
    main()