import csv
import math
import random
from sklearn.model_selection import train_test_split
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
cv_fold = 10
# toggle this to run different dataset 

csv_delimiter = ','
data_name = 'Voice'

def readData(file_path):
    file_name = file_path
    with open(file_name, 'r', encoding="utf8") as file:
        reader = csv.reader(file,delimiter=csv_delimiter)
        csv_list = list(reader)
    print('Read in file -> {}'.format(file_name))
    print('Total data size -> {}'.format(len(csv_list)))
    print('Instance size -> {}'.format(len(csv_list[0])))
    return csv_list

# split data into train or testing set 
def getTrainTestSets(data,test_size,partition=True):
    features = []
    target = []
    # in case the data set are very different in format
    sample_len = len(data[0])
    for elem in data:
        feature = elem[0:sample_len-1]
        feature_vector = []
        for f in feature:
            feature_vector.append(float(f))
        features.append(feature_vector)
        # keep this format in case the label is not numeric
        if elem[-1] == '0':
            val = 0
        else:
            val = 1
        target.append((float(val)))

    features = np.asarray(features,dtype=np.float32)
    target = np.asarray(target,dtype=np.float32)

    X_train,X_test,Y_train,Y_test = train_test_split(features,target,test_size = test_size, random_state = random.randint(1,101)) 
    if partition:
        return X_train,X_test,Y_train,Y_test
    else:
        return features,target

def generateDataset(features,target,fname):
    f = open("./dataset/{}".format(fname), "w")
    for i in range(0,len(features)):
        instance = ''
        for pt in features[i]: 
            instance += str(pt) + ','
        instance += str(target[i])+'\n'
        f.write(instance)

def main():
    '''
    Instruction:
        'test_size_percent' -> partion size and for cv test size in learning curve
        'methods' -> modify it to run desires algorithm
        'hyper_parameters' -> modify it to update the iteration values
    '''
    # parse data
    file_path = './dataset/voice.csv'
    print('Info: Reading data...')
    test_size_percent = 0.30
    all_data = readData(file_path)
    X_train,X_test,Y_train,Y_test = getTrainTestSets(all_data,test_size = test_size_percent)
    features,target = getTrainTestSets(all_data,test_size = test_size_percent,partition=False)
    print('Info: Reading data...Done!\n') 

    generateDataset(X_train,Y_train,'voice_train_data.txt')
    generateDataset(X_test,Y_test,'voice_test_data.txt')

    #file_path = './dataset/voice_temp_data.txt'
    print('Info: Reading data...')
    test_size_percent = 0.7
    #all_data = readData(file_path)
    shuffle(all_data)
    shuffle(all_data)
    shuffle(all_data)
    shuffle(all_data)
    shuffle(all_data)
    X_train,X_test,Y_train,Y_test = getTrainTestSets(all_data,test_size = test_size_percent)
    features,target = getTrainTestSets(all_data,test_size = test_size_percent,partition=False)
    print('Info: Reading data...Done!\n') 

    generateDataset(X_train,Y_train,'voice_cv_data.txt')
    generateDataset(X_test,Y_test,'voice_temp_data.txt')
main()