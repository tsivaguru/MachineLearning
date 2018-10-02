# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 01:05:43 2018

@author: shiva
"""

# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
  
# Function importing Dataset 
def importdata(): 
    balance_data = pd.read_csv('VehicleClassi.csv')
      
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset Header Info: ",balance_data.head()) 
    print("Dataset Size: ",balance_data.size)
    print("Dataset Info:",balance_data.info())
    print("Dataset Sample:",balance_data.sample(5))
    print("Dataset Statistical Summary:",balance_data.describe())
    print("Dataset Number of Null:",balance_data.isnull().sum())
    print("Dataset Columns:",balance_data.columns)
    return balance_data 
  
# Function to split the dataset 
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 1:4] 
    Y = balance_data.values[:, 0] 
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 
      
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
  
  
# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = train_using_entropy(X_train, X_test, y_train) 
   
    
    # Operational Phase 
    print("Results Using Gini Index:") 
    print(clf_gini)
      
    print("Results Using Entropy Index:") 
    print(clf_entropy)
      
      
# Calling main function 
if __name__=="__main__": 
    main() 
