# -*- coding: utf-8 -*-
"""2_AlgoFunctions.ipynb

"""
## Imports"""

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


import implicit
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

"""### 1.Regular PSI """

##INPUTS: factorization of the rank-r matrix Y(0) = USV and the increment ΔA 
def integrator(U0,S0,V0,ΔA):
  K1 = U0 @ S0 + ΔA @ V0               #1st step is to find K1 from inital inputs...
  U1,S1_cap =  QR_decomp(K1)           #compute the QR_decomposition of K1 
  S0_tilde = S1_cap - U1.T @ ΔA @ V0
  L1 = V0 @ S0_tilde.T + ΔA.T @ U1
  V1,S1_T =  QR_decomp(L1)             #compute the QR_decomposition of L1
  S1 = S1_T.T
  return U1,S1,V1


def getStartingValues(A0,k):
  U, S, VT = svds(A0,k=k)
  V = VT.T
  S = np.diag(S)
  return U,S,V

def integratorOnMat(A0,ΔA_train_matrix,ΔA_test_matrix,k):
  U,S,V = getStartingValues(A0,k)          ##technically the starting point U, S, V here are U0, S0, V0T
  for ΔA in ΔA_train_matrix:
    U,S,V = integrator(U,S,V,ΔA)            ##the last U,S,V from this ΔA_train are the starting elements for the ΔA_test  
    
  V_list = []  
  for ΔA in ΔA_test_matrix:
    U,S,V = integrator(U,S,V,ΔA)
    V_list.append(V)
  return V_list

def last_psiTrainMat(A0,ΔA_train_matrix,k):
  U,S,V = getStartingValues(A0,k)          ##technically the starting point U, S, V here are U0, S0, V0T
  for ΔA in tqdm(ΔA_train_matrix):
    U,S,V = integrator(U,S,V,ΔA)            ##the last U,S,V from this ΔA_train are the starting elements for the ΔA_test      
  return U,S,V


"""### PureSVD """
def pureSVD(SVDRatingMatrices, k):
    Vsvd_list = []
    for Rating_Mat in SVDRatingMatrices:
        Usvd, Ssvd, VTsvd = svds(Rating_Mat, k=k)
        Vsvd = VTsvd.T     
        Vsvd_list.append(Vsvd)
    return Vsvd_list


"""### implicit ALS#####"""

def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)

                                                  ##takes rating matrix and masks a certain 20% 
                                                  ##of the user-item interaction
def TrainTest(ratings_matrix,pct_mask=0.2):
    test_set = ratings_matrix.copy()              #making a copy of the rating matrix as text_set
    test_set[test_set!=0]=1                       #storing this as a binary preference matrix 

    train_set = ratings_matrix.copy() 
    nonzero_inds = train_set.nonzero()            #get indices where interaction actually exist
    nonzero_pairs = list(zip(nonzero_inds[0],     #user-item indices into list
                          nonzero_inds[1]))   
    random.seed(0) 
    num_samples = int(np.ceil(pct_mask*len(nonzero_pairs)))  #num of samples to mask
    samples = random.sample(nonzero_pairs, num_samples) 


    user_inds = [i[0] for i in samples]             #Get the user row indices
    item_inds = [i[1] for i in samples]            # Get the item column indices
    train_set[user_inds, item_inds] = 0            # Assign all of the randomly chosen user-item pairs to zero
    train_set.eliminate_zeros()                    # Get rid of zeros in sparse array storage after update to save space
    return train_set, test_set, list(set(user_inds)) # Output the unique list of use


def mean_auc(train_set,test_set,latent_features,masked_users) :
    user_AUC = []
    item_vectors = latent_features[1]                     #item latent features: matrix_V    
    for user in masked_users:
        train_row  = train_set[user,:].toarray().reshape(-1)  ##for each user in the training set
        zero_index = np.where(train_row==0)                   ##get the index where interaction has not yet occurred 
                                                              #extract the user latent features... 
        user_vector = latent_features[0][user,:]              #get me each row in the predicted rating matrix--> 
                                                              #user latent features: matrix_U
        pred_ = user_vector.dot(item_vectors).toarray()[0,zero_index].reshape(-1)  #user prediction
        actual_ = test_set[user,:].toarray()[0,zero_index].reshape(-1) 
                                                            #to test with our system recommending popular item for every user
        user_AUC.append(auc_score(pred_,actual_))
        AVG_test_AUC = float('%.6f'%np.mean(user_AUC)) 
    return AVG_test_AUC


def tune_ALS(train_set,test_set,validation_set,als_param_grid):
    best_auc   = 0
    best_model = None
    for i in tqdm(range(len(als_param_grid))):
        alpha, reg, rank, iter = list(als_param_grid)[i]   
        ALSmodel = implicit.als.AlternatingLeastSquares(
                   factors=rank, regularization=reg, iterations=iter,use_gpu = False)
        #train ALS model
        Item_UsersMAT = (train_set.T * alpha).astype('double')
        ALSmodel.fit(Item_UsersMAT, show_progress=True)                
        user_vecs = ALSmodel.user_factors
        item_vecs = ALSmodel.item_factors                                              
        ##############        ##############                              
        user_vecs_csr = sparse.csr_matrix(user_vecs)       #converting the ALS output to csr_matrix ,
        item_vecs_csr = sparse.csr_matrix(item_vecs.T)     #and transposing th eitem vectors:
        latent_features = [user_vecs_csr,item_vecs_csr]
        test_auc  =  mean_auc(train_set,test_set,latent_features,validation_set)   #masked_users == VALIDATION SET

        print('latent factors= {} ,regularization = {}:, n_iter = {}, alpha = {}, AUC= {}'.format(rank,reg,iter,alpha,test_auc))
        if test_auc > best_auc:
           best_auc  = test_auc
           best_rank = rank
           best_reg  = reg
           best_iter = iter
           best_alpha = alpha
           best_model = ALSmodel
    print('\n Best model; latent factors= {} , regularization = {}, n_iter = {}, alpha = {}, AUC = {}'.format(best_rank,
                                                                              best_reg, best_iter,best_alpha,best_auc))
    return best_rank, best_reg, best_iter,best_alpha,best_auc,best_model




def nonzeros(m, row):
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]


def least_squares_cg(Cui, X, Y, factors, U1, U2,reglr,cg_steps=3):
    YtY = Y.T.dot(Y) + reglr * np.eye(factors)
    for u in range(U1, U2):
        # start from previous iteration
        x = X[u]   #single user vec

        # calculate residual r = (YtCuPu - (YtCuY.dot(Xu), without computing YtCuY
        r = -YtY.dot(x)
        for i, confidence in nonzeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            # calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            # standard CG update
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x


def alternating_least_squares_cg(Cui,nuser_list,nItem_list,factors,alpha,reglr,iter):
    Cui = Cui*alpha
    users, items = Cui.shape  
    X = np.random.rand(users, factors) * 0.01  # initialize factors randomly
    Y = np.random.rand(items, factors) * 0.01
    U_list, V_list = [],[]
    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()
    for U1,U2,I1,I2 in tqdm(zip(nuser_list,nuser_list[1:],nItem_list,nItem_list[1:])):
        for iteration in range(iter):
            least_squares_cg(Cui, X, Y,factors,U1, U2,reglr)   ##for users..
            least_squares_cg(Ciu, Y, X,factors,I1, I2,reglr)   ##for items....
        U_list.append(X)
        V_list.append(Y)

    return U_list, V_list
     


def getiALS_VUlist(bst_Model,UserItem_RatMAT,alpha):
    V_list = []
    U_list = []
    for RatMAT in tqdm(UserItem_RatMAT):
        Item_UsersMAT = (RatMAT.T*alpha).astype('double')
        bst_Model.fit(Item_UsersMAT, show_progress=True)                
        user_vecs = bst_Model.user_factors
        item_vecs = bst_Model.item_factors
        V_list.append(item_vecs)
        U_list.append(user_vecs)
    return V_list,U_list 

"""### 2.Incremental Update:

#### Row Update:
"""

def Updt_RowMatrix(DF,user_column,item_column,itemID_dict,rows_,cols_):  ##rows_ = n_users,cols_ = n_items
    rows0 = DF[user_column].values                       

    Original_itemID = DF[item_column].values                   ##== dict_key..
    cols0 = [itemID_dict.get(item) for item in Original_itemID]   ##== dict_values || get the updated_ids ...          
    data  = np.broadcast_to(1., DF.shape[0]) # ignore rating
    A0_Rating_matrix = coo_matrix((data, (rows0, cols0)), shape=(rows_, cols_)).tocsr()
    if A0_Rating_matrix.nnz < len(data):
        A0_Rating_matrix = A0_Rating_matrix._with_data(np.broadcast_to(1., A0_Rating_matrix.nnz), copy=False)

    return A0_Rating_matrix    

def getRow_Mat(DF_curr,newUSER_id,In_DomainListITEMS,Vi,itemID_dict,user_column,item_column):  #Same num items: as prev step      
    n_cols = Vi.shape[0]                                                   
    new_Userrow = DF_curr[DF_curr[user_column]==newUSER_id] #get  all items bought by the newUser so far 
    itemsbought = new_Userrow[item_column].unique().tolist()         #get all items the newuser has bought
    Out_DomainItem_ = np.setdiff1d(itemsbought,In_DomainListITEMS)   #get newItems the newUsers interacted with

    if len(Out_DomainItem_)== 0:          #if there is no NewItems in NewUser interaction: then update user
       new_Userrow['new_userId'] = 0
       newUser_mat = Updt_RowMatrix(new_Userrow,'new_userId',item_column,itemID_dict,1,n_cols)
       A_row = newUser_mat.todense()        

    else:    #if there is(are) NewItem(s) in the items NewUser has interacted with: update users with only the old items
       In_DomainItem_ = np.intersect1d(itemsbought, In_DomainListITEMS)   #get old items users interacted with only
       newUser_oldItems  = new_Userrow.loc[new_Userrow[item_column].isin(In_DomainItem_) ]  #incase its more than 1 item
       newUser_oldItems['new_userId'] = 0
       newUser_mat = Updt_RowMatrix(newUser_oldItems,'new_userId',item_column,itemID_dict,1,n_cols)
       A_row = newUser_mat.todense()                 #row_matrix of new_user

    return A_row 

def row_update(U0,S0,V0,A_row,k,Forced_Orth=False):
    Sn = len(S0)
    Sdiag = np.eye(Sn)*S0    #diag matrix  (r X r)
    V = U0
    U = V0
    S = Sdiag
    A = A_row.T
    rank = U.shape[1]
    m = np.dot(U.T,A)         #m = UTA || A : Matrix of additional data
    p = A - np.dot(U,m)       #p = (A - UUTA) || [1-UUT]A 
    P = orth(p)               #Orthogonal basis of p  
    Ra = np.dot(P.T,p)

    z = np.zeros(m.shape)

    upper_ = np.hstack((Sdiag,m)) 
    lower_ = np.hstack((z.T,Ra))    
    K = np.vstack((upper_,lower_))  #Eqn-9 || K = [(S m);(0, Ra)]
    U1,S1,V1 = lin.svd(K)           ##Full k_decomposition  
    Uupdt = U1[:,:rank]             ##clip K decomposition to rank
    Supdt = np.diag(S1[:rank])
    Vupdt = V1[:,:rank]
                                             ##get update
    U_updt = np.dot(np.hstack((U,P)),Uupdt)  ##[U P]U`
    n = Vupdt.shape[0] 
    Vp_    = np.dot(V,Vupdt[:rank,:])
    V_updt = np.vstack((Vp_, Vupdt[rank:n, :]))

    if Forced_Orth:
       Uq, Ur = lin.qr(U_updt) 
       Vq, Vr = lin.qr(V_updt)
       Ur_S = np.dot(Ur,Supdt)
       K_orth = np.dot(Ur_S,Vr.T)
       U2,S2,V2 = lin.svd(K_orth)

       Supdt = np.diag(S2)
       U_updt = np.dot(Uq,U2)
       V_updt = np.dot(Vq,V2)
    
    U_updated = np.array(V_updt)
    S_updated = Supdt
    V_updated = np.array(U_updt)
    return U_updated,S_updated,V_updated

"""#### Column Update:"""

def Updt_ColMatrix(DF,user_column,item_column,userID_dict,rows_,cols_):  ##rows_ = n_users,cols_ = n_items
    cols0 = DF[item_column].values

    Original_UserID = DF[user_column].values                        ##== dict_key..
    rows0 = [userID_dict.get(user) for user in Original_UserID]     ##== dict_values || get the updated_ids ...  
    data  = np.broadcast_to(1., DF.shape[0]) # ignore ratings

    A0_Rating_matrix = coo_matrix((data, (rows0, cols0)), shape=(rows_, cols_)).tocsr()
    if A0_Rating_matrix.nnz < len(data):
        A0_Rating_matrix = A0_Rating_matrix._with_data(np.broadcast_to(1., A0_Rating_matrix.nnz), copy=False)

    return A0_Rating_matrix


def getCol_Mat(DF_curr,newITEM_id,In_DomainListUSERS,Ui,userID_dict,user_column,item_column):  #Same num users: as the previous step     
    n_rows = Ui.shape[0]    
    new_Itemcol = DF_curr[DF_curr[item_column]==newITEM_id]   #get all item_df                                                   
    users_whobought = new_Itemcol[user_column].unique().tolist()         #get all users that bought the newitem 
    Out_DomainUser_ = np.setdiff1d(users_whobought,In_DomainListUSERS)   #get newUsers that bought the new item 

    if len(Out_DomainUser_)== 0:   ##if No new user bought the new item: update item with the indomain users
       new_Itemcol['new_itemId'] = 0
       newItem_mat = Updt_ColMatrix(new_Itemcol,user_column,'new_itemId',userID_dict,n_rows,1)
       A_col = newItem_mat.todense()                
 
    else:                   #if there is(are) NewUser(s) that bought the newItems: update items with only the old users
       In_DomainUser_ = np.intersect1d(users_whobought,In_DomainListUSERS) 
       newItem_oldUser  = new_Itemcol.loc[new_Itemcol[user_column].isin(In_DomainUser_)] #get the old  users that bought the item with only
       newItem_oldUser['new_itemId'] = 0
       newItem_mat = Updt_ColMatrix(newItem_oldUser,user_column,'new_itemId',userID_dict,n_rows,1)
       A_col = newItem_mat.todense()              
       
    return A_col

def colunm_update(U0,S0,V0,A_column,k,Forced_Orth=False):
    Sn = len(S0)
    Sdiag = np.eye(Sn)*S0    #diag matrix  (r X r)
    V = V0
    U = U0
    S = Sdiag
    A = A_column
    rank = U.shape[1]
    m = np.dot(U.T,A)         #m = UTA || A : Matrix of additional data
    p = A - np.dot(U,m)       #p = (A - UUTA) || [1-UUT]A 
    P = orth(p)               #Orthogonal basis of p  
    Ra = np.dot(P.T,p)
    z = np.zeros(m.shape)

    upper_ = np.hstack((Sdiag,m)) 
    lower_ = np.hstack((z.T,Ra))    
    K = np.vstack((upper_,lower_))  #Eqn-9 || K = [(S m);(0, Ra)]
    U1,S1,V1 = lin.svd(K)           ##Full k_decomposition  
    Uupdt = U1[:,:rank]             ##clip K decomposition to rank
    Supdt = np.diag(S1[:rank])
    Vupdt = V1[:,:rank]
                                             ##get update
    U_updt = np.dot(np.hstack((U,P)),Uupdt)  ##[U P]U`
    n = Vupdt.shape[0] 
    Vp_    = np.dot(V,Vupdt[:rank,:])
    V_updt = np.vstack((Vp_, Vupdt[rank:n, :]))

    if Forced_Orth:
       Uq, Ur = lin.qr(U_updt) 
       Vq, Vr = lin.qr(V_updt)
       Ur_S = np.dot(Ur,Supdt)
       K_orth = np.dot(Ur_S,Vr.T)
       U2,S2,V2 = lin.svd(K_orth)

       Supdt = np.diag(S2)
       U_updt = np.dot(Uq,U2)
       V_updt = np.dot(Vq,V2)
    
    U_updated = np.array(U_updt)
    S_updated = Supdt
    V_updated = np.array(V_updt)
    return U_updated,S_updated,V_updated

"""#### RowCol Update"""

def UsersItems_RatPair(DF_curr,newUSER_id,newITEM_id,In_DomainListUSERS,In_DomainListITEMS,userID_dict,itemID_dict,Ui,Vi,user_column,item_column):    
    new_Userrow = DF_curr[DF_curr[user_column]==newUSER_id] #get  all items bought by the newUser so far 
    n_cols = Vi.shape[0]                           #n_cols = newITEM_id : num_cols in prev update == PREV['productId'].nunique() == newItem_Id
    itemsbought = new_Userrow[item_column].unique().tolist()         #get all items the newuser has bought
    Out_DomainItem_ = np.setdiff1d(itemsbought,In_DomainListITEMS)   #get newItems the newUsers interacted with

    if len(Out_DomainItem_)== 0:          #if there is no NewItems in NewUser interaction: then update user
       new_Userrow['new_userId'] = 0
       newUser_mat = Updt_RowMatrix(new_Userrow,'new_userId',item_column,itemID_dict,1,n_cols)
       A_row = newUser_mat.todense()        #row_matrix of new_user
 
    else:    #if len(Out_DomainItem_) != 0: #if there is(are) NewItem(s) in the items NewUser has interacted with: update users with only the old items
       In_DomainItem_ = np.intersect1d(itemsbought, In_DomainListITEMS)   #get old items users interacted with only
       newUser_oldItems  = new_Userrow.loc[new_Userrow[item_column].isin(In_DomainItem_) ]  #incase its more than 1 item
       newUser_oldItems['new_userId'] = 0
       newUser_mat = Updt_RowMatrix(newUser_oldItems,'new_userId',item_column,itemID_dict,1,n_cols)
       A_row = newUser_mat.todense()                 #row_matrix of new_user

    new_Itemcol = DF_curr[DF_curr[item_column]==newITEM_id]   #get all users that bought the newItem so far 
    n_rows = Ui.shape[0] +1                                   #Now num of user has increased by 1 ||n_rows: Numof users in last update
    users_whobought = new_Itemcol[user_column].unique().tolist()         #get all users that bought the newitem 
    Out_DomainUser_ = np.setdiff1d(users_whobought,In_DomainListUSERS)   #get newUsers that bought the new item 

    if len(Out_DomainUser_)== 0:   ##if No new user bought the new item: update item with the indomain users
       new_Itemcol['new_itemId'] = 0
       newItem_mat = Updt_ColMatrix(new_Itemcol,user_column,'new_itemId',userID_dict,n_rows,1)
       A_col = newItem_mat.todense()                
 
    else:       #if there is(are) NewUser(s) that bought the newItems: update items with only the old users
       In_DomainUser_ = np.intersect1d(users_whobought,In_DomainListUSERS) 
       newItem_oldUser  = new_Itemcol.loc[new_Itemcol[user_column].isin(In_DomainUser_)] #get the old  users that bought the item with only
       newItem_oldUser['new_itemId'] = 0
       newItem_mat = Updt_ColMatrix(newItem_oldUser,user_column,'new_itemId',userID_dict,n_rows,1)
       A_col = newItem_mat.todense()              
    return A_row, A_col

def getRowCol_psiupdt(U_prev,S_prev,V_prev, A_row,A_col,k=50,Forced_Orth=False):  #Alternating ... 
    U_Row,S_Row,V_Row = row_update(U_prev,S_prev,V_prev,A_row,k,Forced_Orth)    ##user_updt b4 item_updt
    U_ColRow,S_ColRow,V_ColRow = colunm_update(U_Row,S_Row,V_Row,A_col,k,Forced_Orth)  

    return  U_ColRow,S_ColRow,V_ColRow

"""#### Check Deffred Status"""

def ITEMS_defferredStatus(DF_curr,ITEMS_list,defferredItem_list,In_DomainListUSERS,user_column,item_column): #check if item only have new (out-domain) users
    newITEM_id = ITEMS_list[0]
    new_Itemcol = DF_curr[DF_curr[item_column]==newITEM_id]             #get newitem_df 
    users_whobought = new_Itemcol[user_column].unique().tolist()        #get all users that bought the newitem 
    In_DomainUser_ = np.intersect1d(users_whobought,In_DomainListUSERS) #get the old  users that bought the item with only

    if len(In_DomainUser_) == 0:   #if there is(are) NO OldUser(s) that bought the newItems:
       status = True 
       defferredItem_list.append(ITEMS_list.pop(ITEMS_list.index(newITEM_id))) #remove the item from the itemlist and put into defferred list

    else:           #if there is(are)  OldUser(s) that  bought the newItems: 
       status = False  

    defferredItem_LIST = defferredItem_list
    return status, ITEMS_list,defferredItem_LIST 


def USERS_defferredStatus(DF_curr,USERS_list,defferredUsers_list,In_DomainListITEMS,user_column,item_column): #check if item only have new (out-domain) users
    newUSER_id = USERS_list[0]
    new_Userrow = DF_curr[DF_curr[user_column]==newUSER_id]        #get  newUser_df items bought by the newUser so far  
    itemsbought = new_Userrow[item_column].unique().tolist()       #get all items the newuser has bought
    In_DomainItem_ = np.intersect1d(itemsbought, In_DomainListITEMS)  #get old items users interacted with only

    if len(In_DomainItem_) == 0:   #if there is(are) NO OldItem(s) that the user bought:
       status = True 
       defferredUsers_list.append(USERS_list.pop(USERS_list.index(newUSER_id))) #remove the item from the itemlist and put into defferred list
      
    else:          #if there is(are)  OldUser(s) that  bought the newItems: 
       status = False  
      
    defferredUsers_LIST = defferredUsers_list
    return status, USERS_list,defferredUsers_LIST

"""#### Get All V_list"""

def getV_listUpdate(DF_curr,Defferred_Items,Defferred_Users,ITEMS_list,USERS_list,In_DomainListUSERS,In_DomainListITEMS,
                 userID_dict,itemID_dict,Ui,Si,Vi,userCol,itemCol,k,Forced_Orth):

    item_len = len(ITEMS_list)
    user_len = len(USERS_list)
    for i in range(max(item_len,user_len)):
        Item_isCold = True   
        User_isCold = True 
        while (User_isCold) & (len(USERS_list) !=  0) :                #check if untill status==0
              User_isCold, USERS_list, Defferred_Users = USERS_defferredStatus(DF_curr,USERS_list,Defferred_Users,
                                                                         In_DomainListITEMS,userCol,itemCol)          

        while (Item_isCold) & (len(ITEMS_list) !=  0) :           #check if untill status==0
              Item_isCold, ITEMS_list, Defferred_Items = ITEMS_defferredStatus(DF_curr,ITEMS_list,Defferred_Items,
                                                                         In_DomainListUSERS,userCol,itemCol)          

        if (len(ITEMS_list) !=  0) &  (len(USERS_list) !=  0):      #if items & users are still available in  after the checks..:
           newITEM_id = ITEMS_list[0]                               #do row:col update   
           newUSER_id = USERS_list[0]
           userID_dict[newUSER_id] = len(userID_dict.values())   #updated the userId
           itemID_dict[newITEM_id] = len(itemID_dict.values())   #updated the itemId
           A_row,A_col = UsersItems_RatPair(DF_curr,newUSER_id,newITEM_id,In_DomainListUSERS,In_DomainListITEMS,userID_dict,
                                            itemID_dict,Ui,Vi,userCol,itemCol)

           Ui,Si,Vi = getRowCol_psiupdt(Ui,Si,Vi, A_row,A_col,k,Forced_Orth)  
           In_DomainListUSERS.append(newUSER_id)        #LstUpdted_User = newUSER_id
           In_DomainListITEMS.append(newITEM_id)  

           ITEMS_list.pop(ITEMS_list.index(newITEM_id))  #remove the updated item from the list   
           USERS_list.pop(USERS_list.index(newUSER_id))  #remove the updated user from the list     
       
           item_len = len(ITEMS_list)
           user_len = len(USERS_list)
                                                                
        if (len(USERS_list) !=  0) & (len(ITEMS_list) ==  0) :    #if item cataloge is empty and users are still available                                                         
           newUSER_id = USERS_list[0]                             #switch to row update  after exhausting available items
           userID_dict[newUSER_id] = len(userID_dict.values())   #updated the userId

           A_row = getRow_Mat(DF_curr,newUSER_id,In_DomainListITEMS,Vi,itemID_dict,userCol,itemCol)    #update rest of available users         
           Ui,Si,Vi = row_update(Ui,Si,Vi,A_row,k,Forced_Orth)

           In_DomainListUSERS.append(newUSER_id)  
           USERS_list.pop(USERS_list.index(newUSER_id))  #remove the updated user from the list
           user_len = len(USERS_list)

        if (len(ITEMS_list) !=  0) & (len(USERS_list) ==  0) : #if user cataloge is empty and items are still available   
           newITEM_id = ITEMS_list[0]                          #switch to col update  after exhausting available users 
           itemID_dict[newITEM_id] = len(itemID_dict.values()) #updated the itemId                                       
           A_col = getCol_Mat(DF_curr,newITEM_id,In_DomainListUSERS,Ui,userID_dict,userCol,itemCol)    #update rest of available items         
           Ui,Si,Vi = colunm_update(Ui,Si,Vi,A_col,k,Forced_Orth)
           In_DomainListITEMS.append(newITEM_id)   
           ITEMS_list.pop(ITEMS_list.index(newITEM_id))  #remove the updated item from the list
           item_len = len(ITEMS_list)
    return Defferred_Items,Defferred_Users,In_DomainListUSERS,In_DomainListITEMS,userID_dict,itemID_dict,Ui,Si,Vi 

 
"""#### All Step Update"""

def SingleStep_UPDATE(DF_curr,Defferred_Items,Defferred_Users,ITEMS_list,USERS_list,In_DomainListUSERS,In_DomainListITEMS,
                     userID_dict,itemID_dict,U_list,S_list,V_list,userCol,itemCol,k,Forced_Orth):
    Ui,Si,Vi = U_list[-1],S_list[-1],V_list[-1]
    DItems_1, DUsers_1,In_DomainUSERS,In_DomainITEMS,userID_dict,itemID_dict,U1,S1,V1 = getV_listUpdate(DF_curr,Defferred_Items,Defferred_Users,ITEMS_list,USERS_list,In_DomainListUSERS,In_DomainListITEMS,userID_dict,itemID_dict,Ui,Si,Vi,userCol,itemCol,k,Forced_Orth)

    ITEMS_list = DItems_1  #transfer deferred into item&user lists
    USERS_list = DUsers_1
    DefferredItems_2 = []
    DefferredUsers_2 = []
    DItems_2, DUsers_2,In_DomainUSERS,In_DomainITEMS,userID_dict,itemID_dict,U2,S2,V2 = getV_listUpdate(DF_curr,DefferredItems_2,DefferredUsers_2,ITEMS_list,USERS_list,In_DomainUSERS,In_DomainITEMS,userID_dict,itemID_dict,U1,S1,V1,userCol,itemCol,k,Forced_Orth)
    
    U_list.append(U2)
    S_list.append(S2)
    V_list.append(V2)
    UpdtUSERS_list = In_DomainUSERS.copy()
    UpdtITEMS_list = In_DomainITEMS.copy()
    return DItems_2, DUsers_2,In_DomainUSERS,In_DomainITEMS,userID_dict,itemID_dict,UpdtUSERS_list,UpdtITEMS_list,U_list,S_list,V_list


def ALLSTEPs_UPDATE(AllDF_start,AllDF_list,New_itemsList,New_usersList,U_list,S_list,V_list,userCol,itemCol,Nsteps=8,k=50,Forced_Orth=False):
    DF_curr = AllDF_list[0][[userCol,itemCol]]

    Defferred_Items = []
    Defferred_Users = []
    AllUpdtUSERS_List = []
    AllUpdtITEMS_List = []

    ITEMS_list =  New_itemsList[0].tolist() ##out of domain
    USERS_list =  New_usersList[0].tolist()
    In_DomainListUSERS =  AllDF_start[userCol].unique().tolist()     
    In_DomainListITEMS =  AllDF_start[itemCol].unique().tolist()
    userID_dict = {item: idx for idx, item in enumerate(In_DomainListUSERS)}
    itemID_dict = {item: idx for idx, item in enumerate(In_DomainListITEMS)}
    DItems_, DUsers_,In_DomainUSERS,In_DomainITEMS,userID_dict,itemID_dict,UpdtUSERS_,UpdtITEMS_,U_list,S_list,V_list = SingleStep_UPDATE(DF_curr,Defferred_Items,Defferred_Users,ITEMS_list,USERS_list,In_DomainListUSERS,In_DomainListITEMS,
                                                               userID_dict,itemID_dict,U_list,S_list,V_list,userCol,itemCol,k,Forced_Orth)
    AllUpdtUSERS_List.append(UpdtUSERS_)
    AllUpdtITEMS_List.append(UpdtITEMS_)
    for i in tqdm(range(1,Nsteps)):
        Defferred_Items = []
        Defferred_Users = []
      
        ITEMS_list =  New_itemsList[i].tolist()+DItems_ ##append previously defferred items to newlist
        USERS_list =  New_usersList[i].tolist()+DUsers_

        DF_curr_ = AllDF_list[i][[userCol,itemCol]]
        DItems_, DUsers_,In_DomainUSERS,In_DomainITEMS,userID_dict,itemID_dict,UpdtUSERS_,UpdtITEMS_,U_list,S_list,V_list= SingleStep_UPDATE(DF_curr_,Defferred_Items,Defferred_Users,ITEMS_list,USERS_list,In_DomainUSERS,In_DomainITEMS,userID_dict,
                                              itemID_dict,U_list,S_list,V_list,userCol,itemCol,k,Forced_Orth)
        AllUpdtUSERS_List.append(UpdtUSERS_)
        AllUpdtITEMS_List.append(UpdtITEMS_)

    return DItems_, DUsers_,In_DomainUSERS,In_DomainITEMS,userID_dict,itemID_dict,AllUpdtUSERS_List,AllUpdtITEMS_List,U_list,S_list,V_list 
    
