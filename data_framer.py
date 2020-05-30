# -*- coding: utf-8 -*-
"""
@author: Aditya Vashista
"""

import pandas as pd

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

def output_dataset(yF,d,dQ,ques_ids):
    quesF=[]
    for i in range(len(yF)):
        if yF[i]==0:
            if  d[ques_ids['q1_id'][i]]==False:
                quesF.append(dQ[ques_ids['q1_id'][i]])
                d[ques_ids['q1_id'][i]]=True
        else:
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





