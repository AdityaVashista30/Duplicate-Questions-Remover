# -*- coding: utf-8 -*-
"""
@author: Aditya Vashista
"""
"""This file is purely used to create and save various classification models for classifying weather 
a given pair of questions is similar or not

A total of 5 classifiaction models are created here. This is done inorder to increase accuracy 
by combining models to analyse duplicate questions in questions_reducer.py (main) file 

Training Data used here is on openly available Quora dataset of question pairs 
Download link: http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
            or: https://www.kaggle.com/c/quora-question-pairs/data"""

#importing imp libraries
import pandas as pd
import data_preprocessing as dfp
import gensim
import numpy as np
import joblib

#importing KeyedVector model: 'GoogleNews-vectors-negative300.bin.gz'
#Only first 800000 enteries selected as total size of this model is 3.5GB
#To be downloaded from: https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True,limit=800000)

#Getting training dataset: it has 5 pre-filled features: question1,question2,id,qid1,qid2
data=pd.read_csv('train.csv')
data=data.drop(['id', 'qid1', 'qid2'], axis=1)

#adding features/data preprocessing using functions from data_preprocessing.py
data,fs_1=dfp.data_transform(data)
data,fs_2=dfp.fuzzy_features(data)
data,fs_3=dfp.wmd_features(data,model)
data,fs_4=dfp.distance_features(data,model)

y=data.is_duplicate.values
x=data[fs_1+fs_2+fs_3+fs_4]

#data standardisation and cleaning and creating StandardScaler for this task(to be saved for future use)
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
x = x.replace([np.inf, -np.inf], np.nan).fillna(0).values
x= scaler.fit_transform(x)

#saving data for future use to test models
pd.DataFrame(x).to_csv('model_input.csv',index=False)
pd.DataFrame(y).to_csv('model_input_results.csv',index=False)

#spliting in train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


"""BUILDING MODELS: """
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

"""Model 1: XGBoost Classidier"""
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
model1=XGBClassifier()
model1.fit(x_train,y_train)

# Predicting the Test set results
y_pred = model1.predict(x_test)

# Making the Confusion Matrix and accuracies
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator = model1, X = x_train, y = y_train, cv = 15)
accuracies.mean()


"""Model 2: Light GBM Classifier"""
import lightgbm as lgbm
model2 =lgbm.LGBMClassifier()
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
cm2 = confusion_matrix(y_test, y_pred2)
accuracies2 = cross_val_score(estimator = model2, X = x_train, y = y_train, cv = 15)
accuracies2.mean()



"""Model 3: ANN Classifier-1"""
import tensorflow 

ann=tensorflow.keras.models.Sequential() 
ann.add(tensorflow.keras.layers.Dense(units=10,activation="relu"))
ann.add(tensorflow.keras.layers.Dense(units=15,activation="relu"))
ann.add(tensorflow.keras.layers.Dense(units=10,activation="relu"))
ann.add(tensorflow.keras.layers.Dense(units=5,activation="relu"))
ann.add(tensorflow.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(x_train,y_train,batch_size = 32, epochs = 300)

y_pred3=ann.predict(x_test)
y_pred3.resize(len(x_test))
y_pred3=(y_pred3>0.5)
cm3 = confusion_matrix(y_test, y_pred3)



"""Model 4: Random Forest Classifier"""
from sklearn.ensemble import RandomForestClassifier
model4=RandomForestClassifier(n_estimators=35,criterion="entropy",random_state=0)
model4.fit(x_train,y_train)
y_pred4=model4.predict(x_test)
cm4 = confusion_matrix(y_test, y_pred4)
accuracies4 = cross_val_score(estimator = model4, X = x_train, y = y_train, cv = 15)
accuracies4.mean()


"""Model 5: Cat boost classifier"""
import catboost as ctb
model5 = ctb.CatBoostClassifier()
model5.fit(x_train, y_train)
y_pred5=model5.predict(x_test)
cm5 = confusion_matrix(y_test, y_pred5)
accuracies5 = cross_val_score(estimator = model5, X = x_train, y = y_train, cv = 15)
accuracies5.mean()


#saving models and standard scaller
joblib.dump(scaler,'sc.pkl')
joblib.dump(model1,'xgboost_model.pkl')
joblib.dump(model2,'lgbm_model.pkl')
joblib.dump(model4,'rf_model.pkl')
joblib.dump(model5,'catboost_model.pkl')
ann.save('ann_model.h5')