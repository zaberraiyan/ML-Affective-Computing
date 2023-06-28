#Project 1 Affective Computing 
#Zaber Raiyan Choudhury

#RUN extract.py first to get the npy files x.npy and y.npy

#Classifier code: SVM == Support Vector Machines
#                  RF == Random Forest
#                  DT == Decision Tree
#
#Datatype code:   original == original data
#                   center == center translated data
#                        X == x axis rotation by 180 degree
#                        Y == y axis rotation by 180 degree
#                        Z == z axis rotation by 180 degree
#import lib
from sklearn import svm, datasets
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#augmentation matrixes for rotation
#sin_pi = 0 #exactly 0
#cos_pi = -1 #exactly -1

#X rotation matrix
def mx():
    rotX = [[1, 0, 0],[0, -1, 0 ],[0, 0, -1]]
    rotX = np.array(rotX)
    return rotX
    ##print(rotX)

#Y rotation matrix
def my():
    rotY = [[-1, 0, 0],[0, 1, 0 ],[0, 0, -1]]
    rotY = np.array(rotY)
    return rotY
    ##print(rotY)

#Z rotation matrix
def mz():
    rotZ = [[-1, 0, 0],[0, -1, 0 ],[0, 0, 1]]
    rotZ = np.array(rotZ)
    return rotZ
    ##print(rotZ)

#rotation X
def rotX (X):
    x_rotx = []
    for data in X:
        data = np.matmul(data, mx()) #multiply matrix
        x_rotx.append(data)

    x_rotx = np.array(x_rotx)
    X = x_rotx
    return X

#rotation Y
def rotY(X):
    x_roty = []
    for data in X:
        data = np.matmul(data, my())
        x_roty.append(data)

    x_roty = np.array(x_roty)
    X = x_roty
    return X

#rotation Z
def rotZ(X):
    x_rotz = []
    for data in X:
        data = np.matmul(data, mx())
        x_rotz.append(data)

    x_rotz = np.array(x_rotz)
    X = x_rotz
    return X

#centered
def centered_translation(X):
    x_centered = []
    for data in X:
        mean = np.mean(data)
        data = np.subtract(data,mean)
        x_centered.append(data)

    x_centered = np.array(x_centered)
    return x_centered

#3D data plot for question 5
def scatterplot3D(X):
    data = X

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], alpha = 0.2,color='green')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

#Output for train
def PrintEvalMetrics(pred, indices, y):
#manually merge predictions and testing labels from each of the folds to make confusion matrix
  finalPredictions = []
  groundTruth = []
  for p in pred:
    finalPredictions.extend(p)
  for i in indices:
    groundTruth.extend(y[i])
  print(confusion_matrix(groundTruth, finalPredictions))
  print("Precision: ", precision_score(groundTruth, finalPredictions, average='macro'))
  print("Recall: ", recall_score(groundTruth, finalPredictions, average='macro'))
  print("Accuracy: " , accuracy_score(groundTruth, finalPredictions))

#Confusion matrix plot
def display_cm(pred, indices, y):
  hash = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise" ]
  
  finalPredictions = []
  groundTruth = []
  for p in pred:
    finalPredictions.extend(p)
  for i in indices:
    groundTruth.extend(y[i])
  cm = confusion_matrix(groundTruth, finalPredictions)

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= hash)
  disp.plot()
  plt.show()

#model
def CrossFoldValidation(X,y, classifier="DT"):

    if classifier == "SVM":  
        clf = svm.SVC()
    if classifier == "RF":
        clf = RandomForestClassifier()
    if classifier == "DT":
        clf = DecisionTreeClassifier()
    
    pred=[]
    test_indices=[]
    #4-fold cross validation
    kf = KFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        #train classifier
        clf.fit(X[train_index], y[train_index])
        #get predictions and save
        pred.append(clf.predict(X[test_index]))
        #save current test index
        test_indices.append(test_index)
    return pred, test_indices, y


parser = argparse.ArgumentParser(description='Project 1')
parser.add_argument('classifier', nargs='?', type=str, default='DT', help='Classifier type; if none given, DT is default.')
parser.add_argument('datatype', nargs='?', type=str, default='original', help='datatype type; if none given, original is default.')
args = parser.parse_args()

#load data saved from extract.py
X = np.load('x_train.npy')
y = np.load('y_train.npy')

#augmentation
if args.datatype == "X":
    X = rotX(X)
if args.datatype == "Y":
    X = rotY(X)
if args.datatype == "Z":
    X = rotZ(X)
if args.datatype == "center":
    X = centered_translation(X)

#reshaping for classifier
X = np.reshape(X,(60402, 249))

##scatterplot3D(X)      #uncomment to find 3D plot of data

#Run classifier
pred, test_indices, y = CrossFoldValidation(X, y, args.classifier)
#Output
PrintEvalMetrics(pred, test_indices, y)
display_cm(pred, test_indices, y)

