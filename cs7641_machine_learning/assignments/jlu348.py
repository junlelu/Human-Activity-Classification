import csv
import math

from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt
import numpy as np
import time


cv_fold = 10
# toggle this to run different dataset 
abalone = False

if abalone:
    savefig_path = './data_set1_graph/'
    file_path = './data_set/messidor_features.arff' 
    csv_delimiter = ','
    data_name = 'Diabete'
else:
    file_path = './data_set/voice.csv'
    csv_delimiter = ','
    data_name = 'Voice'
'''
Go to def main() for for information!
Data set links:
https://www.kaggle.com/primaryobjects/voicegender/home
https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
'''

def readData():
    file_name = file_path
    with open(file_name, 'r', encoding="utf8") as file:
        reader = csv.reader(file,delimiter=csv_delimiter)
        csv_list = list(reader)
    print('Read in file -> {}'.format(file_name))
    print('Total data size -> {}'.format(len(csv_list)))
    print('Sample size -> {}'.format(len(csv_list[0])))
    return csv_list

def getTrainTestSets(data,test_size,partition=True):
    features = []
    target = []
    # in case the data set are very different in format
    if abalone:
        sample_len = len(data[0])
        for elem in data:
            feature = elem[0:sample_len]
            feature_vector = []
            for f in feature:
                feature_vector.append(float(f))
            features.append(feature_vector)
            target.append(float(elem[-1]))
    else:
        sample_len = len(data[0])
        for elem in data:
            feature = elem[0:sample_len-1]
            feature_vector = []
            for f in feature:
                feature_vector.append(float(f))
            features.append(feature_vector)
            if elem[-1] == 'male':
                val = 0
            else:
                val = 1
            target.append((float(val)))

    features = np.asarray(features,dtype=np.float32)
    target = np.asarray(target,dtype=np.float32)
    X_train,X_test,Y_train,Y_test = train_test_split(features,target,test_size = test_size, random_state=42) 
    print('Total X train data -> {}%'.format(int((len(X_train)/len(data))*100)),'Size:',len(X_train))
    print('Total X test data -> {}%'.format(int((len(X_test)/len(data))*100)),'Size:',len(X_test))
    print('Total Y train data -> {}%'.format(int((len(Y_train)/len(data))*100)),'Size:',len(Y_train))
    print('Total Y test data -> {}%'.format(int((len(Y_test)/len(data))*100)),'Size',len(Y_test))
    if partition:
        return X_train,X_test,Y_train,Y_test
    else:
        return features,target

def plotIt(x_var,cv_scores,train_scores,test_scores,method):
    method_to_title = {'decisionTree':'Decision Tree','adaBoost':'Adaptive Boosting', 'KNN':'K-nearest Neighbors', 'MLP':'Neural Network',
        'SVM':'Support Vector Machine'}   
    xlabel = {'decisionTree':'Maximum Depth','adaBoost':'Number of Estimators', 'KNN':'K-nearst', 'MLP':'Number of Layers',
        'SVM':'Support Vector Machine'}    
    plt.figure() 
    if method == 'SVM':
        n_groups = 3
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.8
        
        plt.bar(index, cv_scores, bar_width ,
                        alpha=opacity,
                        color='r',
                        label='Cross Validation Score')
        
        plt.bar(index + bar_width+0.01, train_scores, bar_width,
                        alpha=opacity,
                        color='b',
                        label='Training Score')

        plt.bar(index + (bar_width+0.01)*2, test_scores, bar_width,
                        alpha=opacity,
                        color='g',
                        label='Test Score')
        xlabel = 'Kernels'
        ylabel = 'Accuracy' 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim([0,1.01])
        plt.title('{} Vs. {}({})'.format(ylabel,xlabel,data_name))
        plt.xticks(index + bar_width, ('RBF', 'Sigmoid', 'Linear'))
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.draw()
        plt.savefig(savefig_path+method)
    else:
        line_width = 4.0
        plt.plot(x_var, cv_scores, 'r',label='Cross Validation Score',linewidth=line_width)
        plt.plot(x_var, train_scores, 'b',label='Training Score',linewidth=line_width)
        plt.plot(x_var, test_scores, 'g',label='Testing Score',linewidth=line_width)
        #print(train_scores)
        plt.legend(shadow=True, fancybox=True)
        plt.title('Accuracy Vs. {}({})'.format(method_to_title[method],data_name))
        plt.xlabel(xlabel[method])
        plt.ylabel('Accuracy')
        plt.ylim([max(min(cv_scores+train_scores+test_scores),0.1)-0.1,1.01])
        plt.grid()
        plt.draw()
        plt.savefig(savefig_path+method)
    return plt

def decisionTree(X_train,X_test,Y_train,Y_test,max_depth):
    model = tree.DecisionTreeClassifier(max_depth = max_depth)
    model.fit(X_train,Y_train) 
    cv_score = cross_val_score(model,X_train,Y_train,cv=cv_fold).mean()
    train_score = model.score(X_train,Y_train)
    test_score = model.score(X_test,Y_test)
    
    return cv_score,train_score,test_score

def adaBoost(X_train,X_test,Y_train,Y_test,n_estimators):
    model = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=(n_estimators + 1) * 10)
    model.fit(X_train,Y_train) 
    cv_score = cross_val_score(model,X_train,Y_train,cv=cv_fold).mean()
    train_score = model.score(X_train,Y_train)
    test_score = model.score(X_test,Y_test)
    return cv_score,train_score,test_score

def KNN(X_train,X_test,Y_train,Y_test,k):
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train,Y_train) 
    cv_score = cross_val_score(model,X_train,Y_train,cv=cv_fold).mean()
    train_score = model.score(X_train,Y_train)
    test_score = model.score(X_test,Y_test)
    return cv_score,train_score,test_score

def MLP(X_train,X_test,Y_train,Y_test,layers=10,neuron=10):
    layer_param = []
    for i in range(layers):
        layer_param.append(neuron)
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layer_param), random_state=1)
    model.fit(X_train,Y_train) 
    cv_score = cross_val_score(model,X_train,Y_train,cv=cv_fold).mean()
    train_score = model.score(X_train,Y_train)
    test_score = model.score(X_test,Y_test)
    return cv_score,train_score,test_score

def SVM(X_train,X_test,Y_train,Y_test,kernel):
    if kernel == 'rbf':
        model = svm.SVC() # rbf
    elif kernel == 'sigmoid':
        model = svm.SVC(kernel="sigmoid")
    elif kernel == 'linear':
        model = svm.LinearSVC()
    else:
        print('Error: no such kernel {}.'.format(kernel))
    model.fit(X_train,Y_train) 
    cv_score = cross_val_score(model,X_train,Y_train,cv=cv_fold).mean()
    train_score = model.score(X_train,Y_train)
    test_score = model.score(X_test,Y_test)
    return cv_score,train_score,test_score

def callClassifier(X_train,X_test,Y_train,Y_test,var,method = 'decisionTree'):
    if method == 'decisionTree':
        return decisionTree(X_train,X_test,Y_train,Y_test,var)
    elif method == 'adaBoost':
        return adaBoost(X_train,X_test,Y_train,Y_test,var)
    elif method == 'KNN':
        return KNN(X_train,X_test,Y_train,Y_test,var)
    elif method == 'MLP':
        return MLP(X_train,X_test,Y_train,Y_test,layers=var,neuron=30) 
    elif method == 'SVM':
        return SVM(X_train,X_test,Y_train,Y_test,var)
    else:
        print('Error: the method {} does not exist.'.format(method))

def tryClassifier(X_train,X_test,Y_train,Y_test,max_var,method = 'decisionTree',):
    method_to_title = {'decisionTree':'Decision Tree','adaBoost':'Adaptive Boosting', 'KNN':'K-nearest Neighbors', 'MLP':'Neural Network',
        'SVM':'Support Vector Machine'}    
    cv_scores = []
    train_scores = []
    test_scores = []
    x_var = []
    print('Info: Performing analysis on classfier:',method_to_title[method])
    start_time = time.time()
    if method == 'SVM':
        kernels = ['rbf', 'sigmoid', 'linear']
        for kernel in kernels:
            print('Info:', method_to_title[method],'Running kernel:{}'.format(kernel))
            cv_score,train_score,test_score = callClassifier(X_train,X_test,Y_train,Y_test,kernel,method=method)
            cv_scores.append(cv_score)
            train_scores.append(train_score)
            test_scores.append(test_score)

            print('Info:', method_to_title[method],'kernel:{}'.format(kernel), '->',100,'%...Done!')
        plotIt(x_var,cv_scores,train_scores,test_scores,method)

    else:
        for x in range(1,max_var):
            cv_score,train_score,test_score = callClassifier(X_train,X_test,Y_train,Y_test,x,method=method)
            x_var.append(x)
            cv_scores.append(cv_score)
            train_scores.append(train_score)
            test_scores.append(test_score)
            percent_done = (x/max_var)*100
            if percent_done % 20 == 0 and percent_done > 19:
                print('Info:', method_to_title[method], '->',int((x/float(max_var))*100),'%...')
        print('Info:', method_to_title[method], '->',100,'%...Done!')
        plotIt(x_var,cv_scores,train_scores,test_scores,method)
    print('Info:', method_to_title[method],'spend %s seconds ---' % round((time.time() - start_time),2))

def learningCurve(features,target,test_size,param):
    plot_learning_curve(features, target, test_size, param, ylim=(0.0, 1.01), n_jobs=4)
    return 0
# Adapted this function from the following link
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(X, y,test_size,param,ylim=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 30% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=test_size, random_state=0)
    estimator_color = {'decison':['C0','C1','Decison Tree'],'adaBoost':['C2','C3','AdaBoost'],'knn':['C4','C5','KNN'],'mlp':['C6','C7','MLP'],'svm':['C8','C9','SVM']}

    # parameters
    max_depth = param['depth']
    ada_estimators = param['ada_estimator']
    k = param['k']  
    layers = param['layers']
    neuron = param['neuron']
    layer_param = []
    
    # decison tree
    estimators = []
    estimator = tree.DecisionTreeClassifier(max_depth = max_depth)
    estimators.append([estimator,'decison'])

    # AdaBoost
    estimator = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=ada_estimators)
    estimators.append([estimator,'adaBoost'])
    # KNN
    estimator = KNeighborsClassifier(n_neighbors=k)
    estimators.append([estimator,'knn'])

    # MLP
    for i in range(layers):
        layer_param.append(neuron)
    estimator = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layer_param), random_state=1)
    estimators.append([estimator,'mlp'])

    # SVM
    estimator = svm.SVC()
    estimators.append([estimator,'svm'])

    # Get all data
    plot_vector = []
    
    for (estimator,es_type) in estimators:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        #train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        #test_scores_std = np.std(test_scores, axis=1)
        plot_vector.append([train_sizes, train_scores_mean,test_scores_mean,estimator_color[es_type]])
    plt.grid()

    #plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                train_scores_mean + train_scores_std, alpha=0.1,
    #                color="r")
    #plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Plot
    for vector in plot_vector:
        train_sizes,train_scores_mean,test_scores_mean,info = vector
        train_color,test_color,method = info
        plt.plot(train_sizes, train_scores_mean, 'o-', color=train_color,
            label="{} Training score".format(method))
        plt.plot(train_sizes, test_scores_mean, 'o-', color=test_color,
            label="{} Cross-validation score".format(method))

    plt.legend(loc="best")
    plt.title('Learning Curve Vs. Data Size({})'.format(data_name))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Data Size")
    plt.ylabel("Score")
    plt.draw()
    plt.savefig(savefig_path+'learning_curve')
    return plt

def main():
    '''
    Instruction:
        'test_size_percent' -> partion size and for cv test size in learning curve
        'methods' -> modify it to run desires algorithm
        'hyper_parameters' -> modify it to update the iteration values
    '''
    # parse data
    print('Info: Reading data...')
    test_size_percent = 0.30
    all_data = readData()
    X_train,X_test,Y_train,Y_test = getTrainTestSets(all_data,test_size = test_size_percent)
    features,target = getTrainTestSets(all_data,test_size = test_size_percent,partition=False)
    print('Info: Reading data...Done!\n') 

    # 5 algorithms
    print('Info: Performing model complexity analysis!')
    methods = ['decisionTree','adaBoost','KNN','MLP','SVM']
    hyper_parameters = {'decisionTree':100,'adaBoost':50,'KNN':100,'MLP':20,'SVM':None}
    # quick graph format check
    #hyper_parameters = {'decisionTree':10,'adaBoost':10,'KNN':10,'MLP':2,'SVM':None}
    for method in methods:
        tryClassifier(X_train,X_test,Y_train,Y_test,hyper_parameters[method],method)
        print()
    print('Info: Completed model complexity analysis...Done!\n')

    print('Info: Performing learning curve analysis!')
    # learning curve
    param = {'depth':10,'ada_estimator':10,'k':10,'layers':20,'neuron':30}

    learningCurve(features,target,test_size_percent,param)   
    print('Info: Performing learning curve analysis...Done!')
    # print info 
    method_to_title = {'decisionTree':'Decision Tree','adaBoost':'Adaptive Boosting', 'KNN':'K-nearest Neighbors', 'MLP':'Neural Network',
        'SVM':'Support Vector Machine'}
    print('Info: Completed analysis on:')
    for i in range(0,len(methods)):
        print('  {}. '.format(i+1),method_to_title[methods[i]])
    
    plt.show()

# run the program
main()