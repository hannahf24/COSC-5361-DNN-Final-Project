import copy
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

#filepaths 
trainFile= "UNSW_NB15_training-set.parquet"
testFile= "UNSW_NB15_testing-set.parquet"
outputDIR = "outputs"
os.makedirs(outputDIR, exist_ok=True) 

#variable setup
seed= 42
valSize= 0.15
batchSize= 512
maxEpochs= 100
patience= 10
lr= 1e-3
weightDecay= 1e-4
dropout= 0.3
aeEpochs= 100
aeLR= 1e-3

#balance classes
CLASS_WEIGHT_MODE= "balanced_sqrt"

# UNSW-NB15 features are extremely heavy-tailed (bytes, loads, rates). log1p before
# standardization is common in the literature and stabilises training. Set False to follow
# the proposal exactly (StandardScaler only).
USE_LOG1P = True

#load and preprocess data
drop= ["id", "label"]
target= "attack_cat"
categorical= ["proto", "service", "state"]

#method to  set random seeds for reproducibility
def set_seed(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#method to load data 
def load_data(path):
    df= pd.read_parquet(path)
    df.columns= df.columns.str.strip()
    df[target] = (df[target].fillna("Normal").astype(str).str.strip()
                  .replace({"Backdoors": "Backdoor", "": "Normal"}))

    for col in categorical:
        df[col] = df[col].astype(str).str.strip()

    return df.drop(columns= [col for col in drop if col in df.columns])

#method to turn data into arrays 
def transform(df, fit=False):
    num= df[numeric].to_numpy(dtype= np.float64)
    if USE_LOG1P:
        num= np.sign(num) * np.log1p(np.abs(num))

    num= scaler.fit_transform(num) if fit else scaler.transform(num)
    cat= ohe.transform(df[categorical])
    x= np.hstack([num, cat]).astype(np.float32)
    y= labelEncoder.transform(df[target]).astype(np.int64)
    return x, y

#loader method
def make_loader(x, y, shuffle):
    dataset= TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size= batchSize, shuffle= shuffle, generator= torch.Generator().manual_seed(seed) if shuffle else None)

set_seed()
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

#load data
fullTrain= load_data(trainFile)
test= load_data(testFile)

#print data shape ad first logisitics 
print("Train: ", fullTrain.shape, "Test: ", test.shape)
print("\nTraining Set Class Distribution:")
dist= fullTrain[target].value_counts()
print(pd.DataFrame({"count": dist, "percent": (100* dist/ dist.sum()).round(2)}))    

#encode labels and train/validation split
labelEncoder= LabelEncoder().fit(fullTrain[target])
classNames= list(labelEncoder.classes_)
nClasses= len(classNames)

train, val= train_test_split( fullTrain, test_size= valSize, stratify= fullTrain[target], random_state= seed)
numeric= [col for col in train.columns if col not in categorical + [target]]
ohe= OneHotEncoder(handle_unknown= "ignore", sparse_output= False).fit(train[categorical])
scaler= StandardScaler() 

#transform train, validation, and test sets
xTrain, yTrain= transform(train, fit=True)
xVal, yVal= transform(val)
xTest, yTest= transform(test)
nFeatures= xTrain.shape[1]

print(f"Features after encoding: {nFeatures}")
print(f"Train: {xTrain.shape}, Val: {xVal.shape}, Test: {xTest.shape}")

#send data to loader 
trainLoader= make_loader(xTrain, yTrain, shuffle= True)
valLoader= make_loader(xVal, yVal, shuffle= False)
testLoader= make_loader(xTest, yTest, shuffle= False)

#class weights fro weighted cross-entropy from trianing split
counts= np.bincount(yTrain, minlength= nClasses).astype(np.float64)
if CLASS_WEIGHT_MODE == "none":
    w= np.ones(nClasses)
else:
    w= counts.sum()/ (nClasses * np.maximum(counts, 1))
    if CLASS_WEIGHT_MODE == "balanced_sqrt":
        w= np.sqrt(w)
classWeights= torch.tensor(w, dtype=torch.float32, device=DEVICE)
print("\nClass Weights: ", dict(zip(classNames, np.round(w, 3))))