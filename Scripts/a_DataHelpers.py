# -*- coding: utf-8 -*-
"""1_DataHelpers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cMLTUr4GWlWebVE-ipbjCDRMVzJA92K1
"""



"""## Imports"""

import io
import os
import gzip
import json
import scipy
import random
import warnings
import numpy as np
import numba as nb
import pandas as pd
import datetime as dt
import seaborn as sns
from tqdm import tqdm
import itertools
import scipy.stats as st
import scipy.sparse as sparse
from scipy.linalg import orth
import matplotlib.pyplot as plt
from numpy import linalg as lin
warnings.filterwarnings('ignore')
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from datetime import datetime as dt_dt
from scipy.sparse.linalg import spsolve
from numpy.linalg import qr as QR_decomp
from collections import OrderedDict
from scipy.sparse import csr_matrix, find
from pandas.api.types import CategoricalDtype

"""## Functions"""



"""### 1.Data Split"""

def getPivotMonths(DF,time_column,N_TMonths):
    pivotMonths_list = []
    ts = DF[time_column]
    for n in range(1,N_TMonths+1):
        pivotMonth = ts.max() - pd.DateOffset(months=n)
        pivotMonths_list.append(pivotMonth)
    return pivotMonths_list

def Time_DataSplit(DF,time_column,pivotMonths_list,N_TMonths,n_train): 
    ΔA_list = []
    ts = DF[time_column]
    A0_df = DF.loc[ts < pivotMonths_list[-1]] 
    ΔA1 = DF.loc[ts >= pivotMonths_list[0]]
    ΔA_list.append(ΔA1)
    for i in range(N_TMonths-1):
        ΔA = DF.loc[(ts >= pivotMonths_list[i+1]) & (ts < pivotMonths_list[i])]
        ΔA_list.append(ΔA)
    ΔA_list =  ΔA_list[::-1]          ##reverse order..
    ΔA_train = ΔA_list[:n_train]  
    ΔA_test = ΔA_list[n_train:]   
    return A0_df,ΔA_train,ΔA_test



def TestTrain_DataSplit(DF,user_column,time_column,ΔA_test):
    AllDF_list  =  []
    PSITest_list = []
    HOLDOUT_list = []
    UserItemDF_list = []
    ts = DF[time_column]
    for test_ in ΔA_test:
        train_ = DF.loc[ts < test_[time_column].min()]
        test_sorted = test_.sort_values(time_column)
        test_idx = [x[-1] for x in test_sorted.index.groupby(test_sorted[user_column]).values()]
    
        holdout = test_.loc[test_idx]
        psi_test = test_.drop(test_idx)
        all_df_ = DF.loc[ts <= test_[time_column].max()] 
        useritem_df = pd.concat([train_,test_.drop(test_idx)], axis = 0)  ##remove holdout items
        
        PSITest_list.append(psi_test)
        HOLDOUT_list.append(holdout)
        AllDF_list.append(all_df_)
        UserItemDF_list.append(useritem_df)

    return AllDF_list, PSITest_list, HOLDOUT_list,UserItemDF_list

"""### 2.Rating Matrices"""

def SingleRatingMatrix(DF,user_column,product_column,rows_,cols_):  ##rows_ = n_users,cols_ = n_items
    rows0 = DF[user_column].values
    cols0 = DF[product_column].values
    data  = np.broadcast_to(1., DF.shape[0]) # ignore ratings

    A0_Rating_matrix = coo_matrix((data, (rows0, cols0)), shape=(rows_, cols_)).tocsr()
    if A0_Rating_matrix.nnz < len(data):
        # there were duplicates accumulated by .tocsr() -> need to make it implicit
        A0_Rating_matrix = A0_Rating_matrix._with_data(np.broadcast_to(1., A0_Rating_matrix.nnz), copy=False)

    return A0_Rating_matrix


def AllRatingMatrices(DFList,user_column,product_column,rows_ ,cols_):
    Rating_matrix_list = []
    for df in DFList:
        df_Mat = SingleRatingMatrix(df,user_column,product_column,rows_, cols_)
        Rating_matrix_list.append(df_Mat)
    return Rating_matrix_list               #return the list of Rating matrices


#######################################################################
      ##get Rating Matrices based on single step interaction only 
def SingleStep_RatMat(DF,user_column,item_column):  ##rows_ = n_users,cols_ = n_items
    rows_ = DF[user_column].nunique() 
    cols_ = DF[item_column].nunique() 
    
    rows0 = DF[user_column].values
    cols0 = DF[item_column].values
    data  = np.broadcast_to(1., DF.shape[0]) # ignore ratings

    A0_Rating_matrix = coo_matrix((data, (rows0, cols0)), shape=(rows_, cols_)).tocsr()
    if A0_Rating_matrix.nnz < len(data):
        # there were duplicates accumulated by .tocsr() -> need to make it implicit
        A0_Rating_matrix = A0_Rating_matrix._with_data(np.broadcast_to(1., A0_Rating_matrix.nnz), copy=False)

    return A0_Rating_matrix

def All_SingleStepRatMat(DFList,user_column,item_column):
    Rating_matrix_list = []
    for df in DFList:
        df_Mat = SingleStep_RatMat(df,user_column,item_column)
        Rating_matrix_list.append(df_Mat)
    return Rating_matrix_list               #return the list of Rating matrices


#######################################################################
#######################################################################
def SingleStepRatMat_2(DF,U,V,user_column,item_column):  ##rows_ = n_users,cols_ = n_items
    rows_ = U.shape[0]
    cols_ = V.shape[0] 
    
    rows0 = DF[user_column].values
    cols0 = DF[item_column].values
    data  = np.broadcast_to(1., DF.shape[0]) # ignore ratings

    A0_Rating_matrix = coo_matrix((data, (rows0, cols0)), shape=(rows_, cols_)).tocsr()
    if A0_Rating_matrix.nnz < len(data):
        A0_Rating_matrix = A0_Rating_matrix._with_data(np.broadcast_to(1., A0_Rating_matrix.nnz), copy=False)

    return A0_Rating_matrix

def AllSingleStepRatMat_2(DFList,Ulist,Vlist,user_column,item_column):
    Rating_matrix_list = []
    for DF,U,V, in zip(DFList,Ulist,Vlist):
        df_Mat = SingleStepRatMat_2(DF,U,V,user_column,item_column)
        Rating_matrix_list.append(df_Mat)
    return Rating_matrix_list              

###########################################################################
###########################################################################

def psiStep_RatMat(DF,All_DF,user_column,item_column):  ##rows_ = n_users,cols_ = n_items
    rows_ = All_DF[user_column].nunique() 
    cols_ = All_DF[item_column].nunique() 
    
    rows0 = DF[user_column].values
    cols0 = DF[item_column].values
    data  = np.broadcast_to(1., DF.shape[0]) # ignore ratings

    A0_Rating_matrix = coo_matrix((data, (rows0, cols0)), shape=(rows_, cols_)).tocsr()
    if A0_Rating_matrix.nnz < len(data):
        # there were duplicates accumulated by .tocsr() -> need to make it implicit
        A0_Rating_matrix = A0_Rating_matrix._with_data(np.broadcast_to(1., A0_Rating_matrix.nnz), copy=False)

    return A0_Rating_matrix

def psiAllStep_RatMat(DFList,All_DF_list,user_column,product_column):
    Rating_matrix_list = []
    for df,all_df in zip(DFList,All_DF_list):
        df_Mat = psiStep_RatMat(df,all_df,user_column,product_column)
        Rating_matrix_list.append(df_Mat)
    return Rating_matrix_list               #return the list of Rating matrices

"""### 3.Find new users and items"""

def Find_NewUsersItems(AllDF_start,AllDF_list,user_column,item_column,N_steps=8):
    New_usersList = []
    prev_users_1 =  AllDF_start[user_column].unique()       #
    new_step_users_1 = AllDF_list[0][user_column].unique()
    new_users_1 = np.setdiff1d(new_step_users_1,prev_users_1)  #elements in 'new_step_users' not in 'all_prev_users' == new_users
    New_usersList.append(new_users_1)

    New_itemsList = []
    prev_items_1 =     AllDF_start[item_column].unique()            #
    new_step_items_1 = AllDF_list[0][item_column].unique()
    new_items_1 = np.setdiff1d(new_step_items_1,prev_items_1)  #elements in 'new_step_items' not in 'all_prev_items' == new_items
    New_itemsList.append(new_items_1)

    for i in range(N_steps-1):
        prev_users = AllDF_list[i][user_column].unique()    ## i == for SVD
        step_users=  AllDF_list[i+1][user_column].unique()    ##(i+1) == for incremental 
        new_users = np.setdiff1d(step_users,prev_users)  ##elements in 'new_step_users' not in 'all_prev_users' == new_users
        New_usersList.append(new_users)

        prev_items = AllDF_list[i][item_column].unique()    ## i == for SVD
        step_items=  AllDF_list[i+1][item_column].unique()    ##(i+1) == for incremental 
        new_items = np.setdiff1d(step_items,prev_items)  ##elements in 'new_step_users' not in 'all_prev_users' == new_users
        New_itemsList.append(new_items)
    return New_usersList,New_itemsList

"""### 4.Dataset Adjustments """

def get_NEWHoldout(HOLDOUT_list,userID_dict,itemID_dict,AllUpdtUSERS_,AllUpdtITEMS_,user_col,item_col):
    newHOLDOUT_LIST =  []
    for DF,Updt_Users,Updt_Items in tqdm(zip(HOLDOUT_list,AllUpdtUSERS_,AllUpdtITEMS_)):
        df = DF.copy()
        newHOLDOUT_ =   df.loc[(df[item_col].isin(Updt_Items)) & (df[user_col].isin(Updt_Users))]#
        newHOLDOUT_ =   newHOLDOUT_[[user_col,item_col]]

        prevUser_ID =   newHOLDOUT_[user_col].values  ##
        prevItems_ID =  newHOLDOUT_[item_col].values   #
        Updted_UserID = [userID_dict.get(user) for user in prevUser_ID]   
        Updted_ItemID = [itemID_dict.get(item) for item in prevItems_ID]

        newHOLDOUT_['Updated_UserID'] = Updted_UserID
        newHOLDOUT_['Updated_ItemID'] = Updted_ItemID
        newHOLDOUT_LIST.append(newHOLDOUT_)
    return  newHOLDOUT_LIST


def adjustedAllDF(AllDF_list,userID_dict,itemID_dict,AllUpdtUSERS_,AllUpdtITEMS_,user_col,item_col):
    newAllDF_list =[]
    for DF, Updt_Users, Updt_Items in tqdm(zip(AllDF_list,AllUpdtUSERS_,AllUpdtITEMS_)):
        df = DF.copy()
        df = df[[user_col,item_col]]
        allnew_df     = df.loc[(df[item_col].isin(Updt_Items)) & (df[user_col].isin(Updt_Users))]

        prevUser_ID =   allnew_df[user_col].values  
        prevItems_ID =  allnew_df[item_col].values   
        Updted_UserID = [userID_dict.get(user) for user in prevUser_ID]   
        Updted_ItemID = [itemID_dict.get(item) for item in prevItems_ID]
        allnew_df['Updated_UserID'] = Updted_UserID
        allnew_df['Updated_ItemID'] = Updted_ItemID
        newAllDF_list.append(allnew_df)
    return newAllDF_list

def adjustedPSI_DF(PSITest_list,userID_dict,itemID_dict,AllUpdtUSERS_,AllUpdtITEMS_,user_col,item_col):
    new_PSIDFlist =[]
    for DF, Updt_Users, Updt_Items in tqdm(zip(PSITest_list,AllUpdtUSERS_,AllUpdtITEMS_)):
        df = DF.copy()
        df = df[[user_col,item_col]]
        new_PSIdf     = df.loc[(df[item_col].isin(Updt_Items)) & (df[user_col].isin(Updt_Users))]

        prevUser_ID =   new_PSIdf[user_col].values  
        prevItems_ID =  new_PSIdf[item_col].values   
        Updted_UserID = [userID_dict.get(user) for user in prevUser_ID]   
        Updted_ItemID = [itemID_dict.get(item) for item in prevItems_ID]
        new_PSIdf['Updated_UserID'] = Updted_UserID
        new_PSIdf['Updated_ItemID'] = Updted_ItemID
        new_PSIDFlist.append(new_PSIdf)
    return new_PSIDFlist



def ADJUST_mainDF(AMZB_DF,userID_dict,itemID_dict,AllUpdtUSERS_,AllUpdtITEMS_,userCol,itemCol):
    Updt_ItemsLast = AllUpdtITEMS_[-1]
    Updt_UsersLast = AllUpdtUSERS_[-1]
    newAMZB_DF = AMZB_DF.loc[(AMZB_DF[itemCol].isin(Updt_ItemsLast)) & (AMZB_DF[userCol].isin(Updt_UsersLast))]
    prevUser_ID =  newAMZB_DF[userCol].values  
    prevItems_ID = newAMZB_DF[itemCol].values   
    Updted_UserID = [userID_dict.get(user) for user in prevUser_ID]   
    Updted_ItemID = [itemID_dict.get(item) for item in prevItems_ID]
    newAMZB_DF['Updated_UserID'] = Updted_UserID
    newAMZB_DF['Updated_ItemID'] = Updted_ItemID
    return newAMZB_DF