# -*- coding: utf-8 -*-
"""
TASK 2

@author: Eran Schenker
"""

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

directory_path='/Users/Eran Schenker/Desktop/Galvanize/Oscar/Data_(Researcher)_Case_Study_Datasets'

# 1. Let's read in the data 
# claim_ccs_2 = pd.read_csv(os.path.join(directory_path,'claim_ccs_2.csv'))
labels_0 = pd.read_csv(os.path.join(directory_path,'labels.csv'))
drugs_0 = pd.read_csv(os.path.join(directory_path,'prescription_drugs.csv'))

## adding year 
drugs_0 = drugs_0.assign(date_svc_datetype = pd.to_datetime(drugs_0['date_svc']))
drugs_0 = drugs_0.assign(year = drugs_0['date_svc_datetype'].dt.year)
drugs_0 = drugs_0.assign(month = drugs_0['date_svc_datetype'].dt.month)
drugs_0 = drugs_0.assign(day = drugs_0['date_svc_datetype'].dt.day)
# min(claim_0['date_svc_datetype']),max(claim_0['date_svc_datetype'])

# subset to only labeled members
keys = labels_0[['member_id','year']].drop_duplicates()
drugs_1  = pd.merge(keys,drugs_0,how = 'left',
                       on = ['member_id','year'])

drugs_2 = drugs_1.dropna()
#drugs_2['year'].value_counts()

# filtering out irrleivant years
drugs_2 = drugs_2[drugs_2['year']>=2015.0]

'''' ####################### Feature Engineering ############'''
drugs_2['DRUG_CATEGORY'] = drugs_2['drug_category']

## indicator features
drugs_2 = pd.get_dummies(drugs_2, columns=['drug_category'],drop_first=False)

## count_prescriptions feature     
cols = [i for i in drugs_2.columns if 'drug_category_' in i]
agg_funcs = ['sum' for i in range(len(cols))]
dic_agg = dict(zip(cols,agg_funcs))
dic_agg['record_id']='count'

grp_drugs_0 = ( drugs_2.groupby(['member_id','year'],as_index = False)
               .agg(dic_agg)
               .rename(columns = {'record_id':'count_prescriptions'} ) )

## count_distinct_drug_types feature
for i in cols:
    grp_drugs_0[i] = grp_drugs_0[i].apply(lambda x: 1 if x>0 else 0)
grp_drugs_0 = grp_drugs_0.assign( count_distinct_drug_types = grp_drugs_0[cols].sum(axis=1))


'''## join dataset with labels ## '''
df_0 = pd.merge(grp_drugs_0,labels_0,on = ['member_id','year'] ,how = 'left' )

e= sns.lmplot(x = 'count_distinct_drug_types',y = 'weighted_risk_score',
              data = df_0.sample(frac = 0.05),aspect=1.5, markers='+', x_jitter=True, y_jitter=True)

if 1==0: ## switch statement to True to save file and read again
    df_0.to_csv(os.path.join(directory_path,'final_df.csv'))
    df_0 = pd.read_csv(os.path.join(directory_path,'final_df.csv'))


'''################# Modeling #########################'''

'''importing models '''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

'''importing metrics '''
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

df_1 = df_0.drop(['Unnamed: 0','count_claims',
               'set_ccs_categories', 'sum_comorbidity', 'count_claims_adj',
               'weighted_risk_score', 'log_weighted_risk_score',
               'prcrnk_weighted_risk_score', 'comorbidity_risk_score',
               'log_comorbidity_risk_score', 'prcrnk_comorbidity_risk_score'], axis =1)
    
### splitting to train set 2016-2017 and test set 2018
year_1_2 = df_1[(df_1['year']<2018) & (df_1['year']>2015)]
train = year_1_2
year_3 = df_1[df_1['year']==2018]
test , test_untouched  = train_test_split( year_3 , test_size = 0.5, random_state = 101)

len(year_3),len(test),len(test_untouched),len(year_1_2),len(train)

# Xtrain = train
Xtrain = train.drop(['member_id','year','risk_label','comorbidity_risk_label'],axis = 1)
ytrain = train['risk_label']
Xtest = test.drop(['member_id','year','risk_label','comorbidity_risk_label'],axis = 1)
ytest = test['risk_label']

'''###################
 instantiating and running a basic RF model with preliminary hyper parameters 
 #################'''

int_params={'n_estimators':500,'min_samples_split':4,'max_features':30,'max_depth':5}

rfc = RandomForestClassifier(random_state=42,n_estimators=int_params['n_estimators'],
                             min_samples_split=int_params['min_samples_split'],
                             max_features=int_params['max_features'],
                             max_depth=int_params['max_depth'])
rfc.fit(Xtrain,ytrain)
predy=rfc.predict(Xtest)
predprob=rfc.predict_proba(Xtest)

''' ########## results #############'''
cm=confusion_matrix(predy,ytest)
cm

print(classification_report(ytest,predy))	

''' ### importance of leading features ###'''
most=rfc.feature_importances_[rfc.feature_importances_>0.01]
collnum=np.where(rfc.feature_importances_>0.01)
coll=Xtrain.columns[collnum]
plt.subplots(figsize=(16,7))
plt.barh(coll,most,)
j=plt.title('Features Importance')

if 1==1: ## switch statement to False to cancel plot
    plt.subplots(figsize=(8,4))
    print('relative frequency of weighted risk score')
    plt.subplot(1,2,1)
    plt.title('all members')
    df_0['weighted_risk_score'][df_0['drug_category_Analgesics - Opioid']== 0].hist(bins=22,density=True,color = 'navy',)
    plt.subplot(1,2,2)
    plt.title('members with opioid perscriptions')
    e= df_0['weighted_risk_score'][df_0['drug_category_Analgesics - Opioid']> 0].hist(bins=22,density=True)
    print('example of 2 members with opioid prescription and their set ccs of categories')
    df_0['set_ccs_categories'][df_0['drug_category_Analgesics - Opioid']> 0].head(2)

