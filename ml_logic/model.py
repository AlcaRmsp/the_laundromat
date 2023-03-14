
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score, accuracy_score, mean_absolute_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


def new_x_train_y_train(data_new):
    X_sampled_train = data_new.drop('isFraud', axis = 1)
    y_sampled_train = data_new['isFraud']

    return X_sampled_train, y_sampled_train

def modellingLR(X_sampled_train, y_sampled_train, X_test, y_test):
    #state the model
    model_LR = LogisticRegression()

    #instantiate the model
    model_LR.fit(X_sampled_train, y_sampled_train)

    #Predict the y
    y_pred_LR = model_LR.predict(X_test)

    #Scores
    recall_LR = recall_score(y_test, y_pred_LR)
    accuracy_LR = accuracy_score(y_test, y_pred_LR)

    #Confusion matrix
    cm_LR = confusion_matrix(y_test, y_pred_LR)
    confusion_matrix_LR = pd.DataFrame(data=cm_LR, columns=['Actual Positive:1', 'Actual Negative:0'],
                                    index=['Predict Positive:1', 'Predict Negative:0'])
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confusion_matrix_LR, annot=True, fmt='d', cmap = 'PuBu', ax = ax)
    plt.savefig('Confusion matrix for LR model')

    return model_LR, recall_LR, accuracy_LR

def modellingDTC(X_sampled_train, y_sampled_train, X_test, y_test):
    #state the model
    model_DTC = DecisionTreeClassifier(max_depth=2, random_state=2)

    #instantiate the model
    model_DTC.fit(X_sampled_train, y_sampled_train)

    #Predict the y
    y_pred_DTC = model_DTC.predict(X_test)

    #Scores
    recall_DTC = recall_score(y_test, y_pred_DTC)
    accuracy_DTC = accuracy_score(y_test, y_pred_DTC)

    #Confusion matrix
    cm_DTC = confusion_matrix(y_test, y_pred_DTC)
    confusion_matrix_DTC = pd.DataFrame(data=cm_DTC, columns=['Actual Positive:1', 'Actual Negative:0'],
                                    index=['Predict Positive:1', 'Predict Negative:0'])
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confusion_matrix_DTC, annot=True, fmt='d', cmap = 'PuBu', ax = ax)
    plt.savefig('Confusion matrix for DTC model')

    return model_DTC, recall_DTC, accuracy_DTC

def modellingRFC(X_sampled_train, y_sampled_train, X_test, y_test):
    #state the model
    model_RFC = RandomForestClassifier()

    #instantiate the model
    model_RFC.fit(X_sampled_train, y_sampled_train)

    #Predict the y
    y_pred_RFC = model_RFC.predict(X_test)

    #Scores
    recall_RFC = recall_score(y_test, y_pred_RFC)
    accuracy_RFC = accuracy_score(y_test, y_pred_RFC)

    #Confusion matrix
    cm_RFC = confusion_matrix(y_test, y_pred_RFC)
    confusion_matrix_RFC = pd.DataFrame(data=cm_RFC, columns=['Actual Positive:1', 'Actual Negative:0'],
                                    index=['Predict Positive:1', 'Predict Negative:0'])
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confusion_matrix_RFC, annot=True, fmt='d', cmap = 'PuBu', ax = ax)
    plt.savefig('Confusion matrix for RFC model')

    return model_RFC, recall_RFC, accuracy_RFC

def modellingXGBC(X_sampled_train, y_sampled_train, X_test, y_test):
    #state the model
    model_XGBC = XGBClassifier()

    #instantiate the model
    model_XGBC.fit(X_sampled_train, y_sampled_train)

    #Predict the y
    y_pred_XGBC = model_XGBC.predict(X_test)

    #Scores
    recall_XGBC = recall_score(y_test, y_pred_XGBC)
    accuracy_XGBC =accuracy_score(y_test, y_pred_XGBC)

    #confusion matrix
    cm_XGBC = confusion_matrix(y_test, y_pred_XGBC)
    confusion_matrix_XGBC = pd.DataFrame(data=cm_XGBC, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confusion_matrix_XGBC, annot=True, fmt='d', cmap = 'PuBu', ax = ax)

    return model_XGBC, recall_XGBC, accuracy_XGBC

def modellingSVM(X_sampled_train, y_sampled_train, X_test, y_test):
    #state the model
    model_SVM = svm.SVC(kernel='linear')

    #instatiate the model
    model_SVM.fit(X_sampled_train, y_sampled_train)

    #Predict the y
    y_pred_SVM = model_SVM.predict(X_test)

    #Scores
    recall_SVM = recall_score(y_test, y_pred_SVM)
    accuracy_SVM = accuracy_score(y_test, y_pred_SVM)

    #confusion matrix
    cm_SVM = confusion_matrix(y_test, y_pred_SVM)
    confusion_matrix_SVM = pd.DataFrame(data=cm_SVM, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confusion_matrix_SVM, annot=True, fmt='d', cmap = 'PuBu', ax = ax)

    return model_SVM, recall_SVM, accuracy_SVM
