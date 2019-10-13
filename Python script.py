import os

os.getcwd()
os.chdir("C:/Users/gopin/Documents/R/CustomerTransaction")

import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("train.csv")

savedData = data

data1 = savedData.iloc[:, 1:202]

# Detect outliers & delete(Casual)
for i in data1.columns[1:202]:
    q75, q25 = np.percentile(data1.loc[:,i],[75,25])
    iqr = q75 - q25
    innerfence = q25 - (float(iqr) * 1.5)
    outerfence = q75 + (float(iqr) * 1.5)
    data1 = data1.drop(data1[data1.loc[:,i] < innerfence].index)
    data1 = data1.drop(data1[data1.loc[:,i] > outerfence].index)
# Replace with NA
for i in data1.columns[1:202]:
    q75, q25 = np.percentile(data1.loc[:,i],[75,25])
    iqr = q75 - q25
    innerfence = q25 - (iqr * 1.5)
    outerfence = q75 + (iqr * 1.5)
    data1.loc[data1[i] < innerfence,:i] = np.nan
    data1.loc[data1[i] > outerfence,:i] = np.nan

# Impute using Mean method
for i in data1.columns[1:202]:
    data1[i] = data1[i].fillna(data1[i].mean())

# Decision Tree Classification
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data1['target'] = data['target'].replace(1,'yes')
data1['target'] = data['target'].replace(0,'no')

x = data1.iloc[:, 1:202]
y = data1.iloc[:, 0]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

clf = tree.DecisionTreeClassifier(criterion = 'entropy').fit(xTrain,yTrain)

clf_predicted = clf.predict(xTest)

from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(yTest, clf_predicted)
confusionMatrix = pd.crosstab(yTest, clf_predicted)

TN = confusionMatrix.iloc[0,0]
FN = confusionMatrix.iloc[1,0]
TP = confusionMatrix.iloc[1,1]
FP = confusionMatrix.iloc[0,1]
totalObservations = (TN + FN + TP + FP)

# Accuracy
accuracy = ((TP + TN) * 100)/totalObservations

# Precision
precision = (TP * 100)/(TP + FP)

# Recall
recall = (TP * 100)/(TP + FN)

# False Negative Rate
fnRate = (FN * 100)/(FN + TP)










