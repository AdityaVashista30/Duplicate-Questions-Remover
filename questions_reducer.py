# -*- coding: utf-8 -*-
"""
@author: Aditya Vashista

Main file to detect duplicate questions in a set of questions and return output file having distinct questions only
"""

"""It uses previously three created files:
    1) data_preprocessing.py: to add features,process data for inputs for predictions
    2) 5 models stored from models.py to predict the duplicity is set of questions
    3) data_framer.py: To create data frame from input file of questions suitable for predictions
                        To create single column dataframe of distinct questions to be saved in seperate file"""

#importing important libraries
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
from statistics import mode 
import data_preprocessing as dpp
import data_framer as dF
import gensim
from os import system


#taking important user inputs: location of input file and dir where output file is to be stored 
input_file=input("Enter path/storage location of CSV file containing questions: \n")
output_file=input("Enter path/storage location where you want to save new file having reduced questions: \n")

#opening input file
df=pd.read_csv(input_file, encoding= 'unicode_escape')
originalQ=len(df) #number of input questions

print("\n Number of Questions before processing: ",originalQ)
print("Please wait for a while...")
print("The model is looking for unique set questions.....")


#loading previously saved models for predictions
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True,limit=800000)

model1=load('models\\xgboost_model.pkl')
model2=load('models\\catboost_model.pkl')
model3=load('models\\lgbm_model.pkl')
model4=load('models\\rf_model.pkl')
model5 =load_model('models\\ann_model.h5')


#formating input dataframe to another data frame suitable for further processing
data,d,dQ=dF.make_dataset(df)
ques_ids=data[['q1_id','q2_id']] #seperating out question id pairs (to be qiven input for output_dataset() function)


#data preprocessing and adding features
data,fs_1=dpp.data_transform(data)
data,fs_2=dpp.fuzzy_features(data)
data,fs_3=dpp.wmd_features(data,model)
data,fs_4=dpp.distance_features(data,model)

x=data[fs_1+fs_2+fs_3+fs_4]
x=dpp.data_standard(x)


#predicting outputs from various individul models
yP1=model1.predict(x)
yP2=model2.predict(x)
yP3=model3.predict(x)
yP4=model4.predict(x)
yP5=model5.predict(x)
yP5.resize(len(x))
yP5=yP5>0.5
yP5=np.multiply(yP5, 1)

#taking mode of all 5 predictions from 5 models inorder to get max accuracy and storing them in seperate array
yF=[]
for i in range(len(x)):
    l=[yP1[i],yP2[i],yP3[i],yP4[i],yP5[i]]
    yF.append(mode(l))

yF=np.array(yF)

#getting list of uniques set of questions from data_framer.output_dataset()
quesF=dF.output_dataset(yF,d,dQ,ques_ids)
finalQ=len(quesF) #number of distinct questions

#Saving output file
pd.DataFrame(quesF,columns=['Questions']).to_csv(output_file+'\\unique_questions.csv',index=False)


#final outputs:
print("\n Task Completed!!!")
print("CSV file containing reduced and unique number of questions stored at desired folder as unique_questions.csv")
print("\n Questions in new file: ",finalQ)
print("Removed ",(originalQ-finalQ)," duplicate questions")

system("pause")
