# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 23:36:48 2022

@author: Alpha
"""
#1 Import the packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier




#2 Import data

df = pd.read_csv(r"C:\Users\Aaron\Downloads\garments_worker_productivity.csv")
df.info()
df.shape()
df.isnull().sum()

#3 Data visualisation
plt.figure(figsize=(8,4)
sns.barplot(data-df, y='incentive', x='department', color='darkblue')

#4 Analysis part
plt.figure(figsize=(10,4)
sns.boxplot(data = df)
plt.xticks(rotation = 90)
plt.title('Box Plot')

plt.figure(figsize=(10,5)
           sns.countplot(x='department', hue='quarter' data=df)
           df['Department'].values_counts().plot.pie()

#5 Data training
df.drop(["date"], axia =1, inplace=True)
label_encoder = preprocessing.LabelEncoder()
for columns in df.columns:
df[columns]= labels_encoder.fit_transform(df[column])
df

X = dr.drop('no_of_style_change' axis=1)
y= df['no_of_style_change']
X_train, X_test, y_train, y_test = train_test_split(X, y, test size=0, 20)

#6 Random Forest classifier
rfc = RandomForestClassifier(n_estimators=600)
model= rfc.fit(X_train, y_train)
y_pred_lreg = rfc.predict(X_test)
logred_accuracy = round(accuracy_score(y_test, y_pred_lreg) * 100,2)
print('Accuracy,  logreg_accuracy, '%')
     
predictions= rfc.predict(X_test)
true_labels = y_test
cf_matrix = confusion_matrix(true_labels, prediction)
plt.figure(figsize=(7.4))
heatmap = sns.heatmap(cf.matrix, annot=True, cmap='Blues', fmt='g', 
                      xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))



















           