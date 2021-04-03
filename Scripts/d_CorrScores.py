# -*- coding: utf-8 -*-
"""d_CorrScores.ipynb"""


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

"""### 1.Corr Scores """



##weight Jaccard for correlation 
def no_copy_csr_matrix(data, indices, indptr, shape, dtype):
        # set data and indices manually to avoid index dtype checks
        # and thus prevent possible unnecesssary copies of indices
    matrix = csr_matrix(shape, dtype=dtype)
    matrix.data = data
    matrix.indices = indices
    matrix.indptr = indptr
    return matrix

def build_rank_weights_matrix(recommendations, shape):
    recommendations = np.atleast_2d(recommendations)
    n_users, topn = recommendations.shape
    weights_arr = 1. / np.arange(1, topn+1) # 1 / rank
    weights_mat = np.lib.stride_tricks.as_strided(weights_arr, (n_users, topn), (0, weights_arr.itemsize))

    data = weights_mat.ravel()
    indices = recommendations.ravel()
    indptr = np.arange(0, n_users*topn + 1, topn)

    weight_matrix = no_copy_csr_matrix(data, indices, indptr, shape, weights_arr.dtype)
    return weight_matrix

def rank_weighted_jaccard_index(inds1, inds2):
    shape = inds1.shape[0], max(inds1.max(), inds2.max())+1
    weights1 = build_rank_weights_matrix(inds1, shape)
    weights2 = build_rank_weights_matrix(inds2, shape)
    jaccard_index = weights1.minimum(weights2).sum(axis=1) / weights1.maximum(weights2).sum(axis=1)
    return np.asarray(jaccard_index).squeeze()



def Updt_getAll_AvgCorr(All_PRED,SVDTrain_list,Corr_steps_,user_name):
    All_Corr_List = []
    for step in Corr_steps_:
        prev_users =   SVDTrain_list[step-1][user_name].unique()
        PREV_StepPred = All_PRED[step-1][prev_users,:]
        CURR_StepPred = All_PRED[step][prev_users,:]
        corr_result = rank_weighted_jaccard_index(PREV_StepPred, CURR_StepPred)  ##corr values for all users at each dual step
        All_Corr_List.append(corr_result) 
    AUserC_arry = np.array(All_Corr_List).T      
    return AUserC_arry      #corr values for all users at every step 


#####All Ranks CorrrScores 
def updtCorr_4AllRanks(UserItem_MatList,UserItem_List,Corr_steps_,V_list,start_value,MAX_RANK,increment,N):
    print("Correlation for Allusers: ")
    All_UsersCorrUPDT = []
    for Mat_,UIlist_,V_ in tqdm(zip(UserItem_MatList,UserItem_List,V_list)): 
        ALLUpdt_Pred =  getALLTopNPred_ALLUSERS(Mat_,V_,N)   
        AUsersCorr_ = Updt_getAll_AvgCorr(ALLUpdt_Pred,UIlist_,Corr_steps_,user_name)    #Avg_AAUsersPSI
        All_UsersCorrUPDT.append(AUsersCorr_)
    #np.savez_compressed(SAVE_name,All_UsersCorrPSI) 
    return All_UsersCorrUPDT

##############################################################################



