# -*- coding: utf-8 -*-
"""
@author: Aditya Vashista

This file is created for the sole purpose of creating a dataframe (and data prediction helper objects)
from input file of questions whhich is 1)suitable for data preprocessing and predictions and 
2) to store a single column of refined questions

"""

import pandas as pd

"""This function converts single column of available questions into a data frame of 4 columns
The 4 columns are: question1,question2,q1_id and q2_id
The 2 out of 5 colums are input for data preprocessing to be done by data_preprocessing.py: 'question1' & 'question2'
The other 2 colums are used by the output_dataset() function to store set of unique questions


Thifunctions returns 3 objects:
    1) DataFreame of question pairs having n(n-1)/2 rows for input data set of n questions
    2)A dictionary d: maped with question id and their status of duplicity checked: 
                    False= not checked; True= checked (to be used in final step: output_dataset())
    3)Dictionary dQ: mapping unique question ids to their corresponding Questions,
                    to be used in final step: output_dataset() for storing unique questions
"""

def make_dataset(data):
    df=pd.DataFrame(columns=['question1','question2','q1_id','q2_id'])
    d={}
    dQ={}
    d[len(data)]=False
    for i in range(len(data)-1):
        d[i+1]=False
        q1=data.iloc[i][0]
        q1=q1.replace(u'\xa0', ' ')
        dQ[i+1]=q1
        for j in range(i+1,len(data)):
           q2=data.iloc[j][0]
           q2=q2.replace(u'\xa0', ' ') 
           df=df.append({'question1':q1,'question2':q2,'q1_id':i+1,'q2_id':j+1},ignore_index=True)
    
    q1=data.iloc[len(data)-1][0]
    q1=q1.replace(u'\xa0', ' ')
    dQ[len(data)]=q1
    return df,d,dQ


"""This function takes in dictionaries: d,dQ (explained above); 
                    yF: output of grouped predictions wether the question pairs are duplicate or not
                    ques_ids: dataframe of question ids pair of questions corresponding to whick yF is calculated        

Output is list of distinct questions                                                             
"""
def output_dataset(yF,d,dQ,ques_ids):
    quesF=[]
    for i in range(len(yF)):
        #0 means the question pair ae different; 1 they are same
        if yF[i]==0:
            #adding question to output fial if q1 is not been checked yet
            if  d[ques_ids['q1_id'][i]]==False:
                quesF.append(dQ[ques_ids['q1_id'][i]])
                d[ques_ids['q1_id'][i]]=True #marking question so that it can't be added again 
                
        else:
            #marking both questions checked since dulicity is present and the won't be added
            if  d[ques_ids['q1_id'][i]]==True:
                d[ques_ids['q2_id'][i]]=True
            
            elif  d[ques_ids['q2_id'][i]]==True:
                d[ques_ids['q1_id'][i]]=True
            
            else:
                quesF.append(dQ[ques_ids['q1_id'][i]])
                d[ques_ids['q1_id'][i]]=True
                d[ques_ids['q2_id'][i]]=True
    if d[len(d)]==False:
        quesF.append(dQ[len(d)])
    return quesF





