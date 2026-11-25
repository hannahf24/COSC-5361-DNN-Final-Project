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
testFile= "UNSW_NB15_test-set.parquet"
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