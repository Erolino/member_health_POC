# -*- coding: utf-8 -*-
"""
TASK 1

@author: Eran Schenker
"""

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

directory_path='/Users/Eran Schenker/Desktop/Galvanize/Oscar/Data_(Researcher)_Case_Study_Datasets'

# 1. Let's read in the data 
ccs_0=pd.read_csv(os.path.join(directory_path,'ccs.csv'))
claim_0 = pd.read_csv(os.path.join(directory_path,'claim_lines.csv'))
drugs_0 = pd.read_csv(os.path.join(directory_path,'prescription_drugs.csv'))

## adding year 
claim_0 = claim_0.assign(date_svc_datetype = pd.to_datetime(claim_0['date_svc']))
claim_0 = claim_0.assign(year = claim_0['date_svc_datetype'].dt.year)
claim_0 = claim_0.assign(month = claim_0['date_svc_datetype'].dt.month)
claim_0 = claim_0.assign(day = claim_0['date_svc_datetype'].dt.day)
# min(claim_0['date_svc_datetype']),max(claim_0['date_svc_datetype'])

## let's get rid of anyting below 2015 for now
claim_1 = claim_0[claim_0['year']>=2015.0]

# generating seqence of days from start
minn=min(claim_1['date_svc_datetype'])
claim_1['start_date_diag']=minn
claim_1['day_from_start']= claim_1['date_svc_datetype']- claim_1['start_date_diag']
claim_1['day_from_start'] = claim_1['day_from_start'].dt.days+1

## adjsuting to the diagnosis format of CCS before joining
claim_1['diag_code'] = claim_1['diag1'].apply(lambda x: x.replace('.',''))

claim_2 = claim_1.drop(['date_svc','diag1'],axis = 1) 

## joining
claim_ccs_0 = pd.merge(claim_2,ccs_0,how = 'left',
                       left_on = ['diag_code'],
                       right_on = ['diag'] ,suffixes=('_claims', '_ccs'))

## fillna with 'unknown'
claim_ccs_1 = claim_ccs_0.fillna('unknown')

''' 
#############  Building Risk Score ############
'''

claim_ccs_2 = claim_ccs_1
claim_ccs_2['CCS_1_DESC'] = claim_ccs_2['ccs_1_desc'] 

## creating indicator columns for each "ccs_1_desc"
claim_ccs_2 = pd.get_dummies(claim_ccs_2,columns=['ccs_1_desc'],drop_first=False)

## building dictionary to be used in aggregation 
cols = [i for i in claim_ccs_2.columns if 'ccs_1_desc_' in i] ## all indicator columns to be aggregated
agg_funcs = ['sum' for i in range(len(cols))]  ## function to be used -"sum"
dic_agg = dict(zip(cols,agg_funcs)) 
dic_agg['record_id']='count' ## adding count function for 'records'
dic_agg['CCS_1_DESC']=lambda x: set(x)  ## creating a set ccs_1_desc for the member

## aggregating
grp_claim_0 = ( claim_ccs_2.groupby(['member_id','year'],as_index = False)
               .agg(dic_agg)
               .rename(columns = {'record_id':'count_claims','CCS_1_DESC':'set_ccs_categories'}) )

## creating an indicator whether a member had at least one type of ccs_1_desc throughout the year 
for i in cols:
    grp_claim_0[i] = grp_claim_0[i].apply(lambda x: 1 if x>0 else 0)
    
## to drop ill defined and unknown from 'comorbidity' calculation
if 1==1:
    cols = [i for i in claim_ccs_2.columns if 'ccs_1_desc_' in i and 
           'ill-defined' not in i and
            'unknown' not in i]
else:
    cols = cols

grp_claim_0 = grp_claim_0.assign( sum_comorbidity = grp_claim_0[cols].sum(axis=1))

'''######### Calculating risk scores ######### '''

top = round(grp_claim_0['count_claims'].quantile(0.99),5)

grp_claim_0 = grp_claim_0.assign( count_claims_adj = 
                                 grp_claim_0['count_claims'].apply(lambda x: top if x>top else x))

# weighted score  
grp_claim_0 = grp_claim_0.assign( weighted_risk_score = 
                                 grp_claim_0['count_claims_adj']* 0.25 + grp_claim_0['sum_comorbidity'] * 0.75)

# log weighted score
grp_claim_0 = grp_claim_0.assign( log_weighted_risk_score = 
                                 np.log10(grp_claim_0['weighted_risk_score']) )

# prc_rank of weighted score 
grp_claim_0 = grp_claim_0.assign( prcrnk_weighted_risk_score = 
                                 grp_claim_0['weighted_risk_score'].rank(pct=True) )

# classes 0 - low, 1 - med , 2 - high
grp_claim_0['risk_label'] = np.where(grp_claim_0['prcrnk_weighted_risk_score']>=0.2,1,0)
grp_claim_0['risk_label'] = np.where(grp_claim_0['prcrnk_weighted_risk_score']>=0.6,2,
                                                grp_claim_0['risk_label'])

# risk score (only comorbidity)
grp_claim_0 = grp_claim_0.assign( comorbidity_risk_score = 
                                 grp_claim_0['sum_comorbidity']) 

# log (only comorbidity)
grp_claim_0 = grp_claim_0.assign( log_comorbidity_risk_score = 
                                 np.log10(grp_claim_0['comorbidity_risk_score']) ) 

# prc_rank (only comorbidity)
grp_claim_0 = grp_claim_0.assign( prcrnk_comorbidity_risk_score = 
                                 grp_claim_0['comorbidity_risk_score'].rank(pct=True) )

# classes 0 - low, 1 - med , 2 - high
grp_claim_0['comorbidity_risk_label'] = np.where(grp_claim_0['prcrnk_comorbidity_risk_score']>=0.33,1,0)
grp_claim_0['comorbidity_risk_label'] = np.where(grp_claim_0['prcrnk_comorbidity_risk_score']>=0.9,2,
                                                grp_claim_0['comorbidity_risk_label'])

''' plotting transformation from risk score to risk label'''
if 1==1: ## change statement to false to cancel ploting 

    plt.subplots(figsize=(16,4))
    plt.subplot(1,3,1)
    plt.title('weighted risk score')
    grp_claim_0['weighted_risk_score'].hist(bins = 100, color = 'purple')
    plt.subplot(1,3,2)
    plt.title('percent rank weighted risk score')
    grp_claim_0['prcrnk_weighted_risk_score'].hist(bins = 30, color = 'navy')
    e=plt.subplot(1,3,3)
    plt.title('risk_label')
    e= grp_claim_0['risk_label'].hist(bins = 6)


### droping columns before saving

colls = ['ccs_1_desc_Certain conditions originating in the perinatal period',
       'ccs_1_desc_Complications of pregnancy; childbirth; and the puerperium',
       'ccs_1_desc_Congenital anomalies',
       'ccs_1_desc_Diseases of the blood and blood-forming organs',
       'ccs_1_desc_Diseases of the circulatory system',
       'ccs_1_desc_Diseases of the digestive system',
       'ccs_1_desc_Diseases of the genitourinary system',
       'ccs_1_desc_Diseases of the musculoskeletal system and connective tissue',
       'ccs_1_desc_Diseases of the nervous system and sense organs',
       'ccs_1_desc_Diseases of the respiratory system',
       'ccs_1_desc_Diseases of the skin and subcutaneous tissue',
       'ccs_1_desc_Endocrine; nutritional; and metabolic diseases and immunity disorders',
       'ccs_1_desc_Infectious and parasitic diseases',
       'ccs_1_desc_Injury and poisoning', 'ccs_1_desc_Mental Illness',
       'ccs_1_desc_Neoplasms',
       'ccs_1_desc_Residual codes; unclassified; all E codes [259. and 260.]',
       'ccs_1_desc_Symptoms; signs; and ill-defined conditions and factors influencing health status',
       'ccs_1_desc_unknown']

grp_claim_1 = grp_claim_0.drop(colls,axis = 1)

grp_claim_1.info()

if 1==0: ## switch statement to True if would like to save this file 
    grp_claim_1.to_csv(os.path.join(directory_path,'labels.csv'))