#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 17:15:45 2025

@author: jliu6
"""

import os
import torch
import pandas as pd
from torch.utils.data import Subset
from torch.utils.data import Dataset
import re
import time
from itertools import chain
import random
import ast
import matplotlib.pyplot as plt
import numpy as np
import h5py
import io
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from tqdm import tqdm
from pathlib import Path

def get_feature_idexes(method, include_tumor_fraction = True):
    
    if method == 'retccl':
        selected_feature = [str(i) for i in range(0,2048)] 
    elif method == 'uni1': 
        selected_feature = [str(i) for i in range(0,1024)] 
    elif method == 'uni2' or method == 'prov_gigapath':
        selected_feature = [str(i) for i in range(0,1536)] 
    elif method == 'virchow2':
        selected_feature = [str(i) for i in range(0,2560)] 
    elif method == 'hoptimus0':
        selected_feature = [str(i) for i in range(0,1536)] 

    if include_tumor_fraction == True:
        selected_feature = selected_feature + ['TUMOR_PIXEL_PERC'] 
        
    return selected_feature


def load_model_ready_data(
    data_path: str,
    cohort: str,
    pixel_overlap: int,
    fe_method: str,
    tumor_frac: float
 ):
    """
    Load preprocessed model-ready data for a given cohort.

    Parameters
    ----------
    data_path :     data directory
    cohort :        Name of the cohort to load data for
    pixel_overlap : Pixel overlap 
    fe_method :     Feature extraction method
    tumor_frac :    Tumor fraction threshold used during preprocessing.

    Returns
    -------
    dict or None
        Loaded model-ready data if found, otherwise None.
    """

    feature_path = os.path.join(
        data_path,
        cohort,
        f"IMSIZE250_OL{pixel_overlap}",
        f"feature_{fe_method}",
        f"TFT{tumor_frac}"
    )

    data_file = os.path.join(feature_path, f"{cohort}_data.pth")

    try:
        return torch.load(data_file)
    except FileNotFoundError:
        return None
    

    
def get_cohort_data(data_path, cohort_name, fe_method, tumor_frac):
    
    #Load data
    data_ol100 = load_model_ready_data(data_path, cohort_name, 100, fe_method, tumor_frac) #overlap 100
    data_ol0   = load_model_ready_data(data_path, cohort_name, 0, fe_method, tumor_frac) #overlap 0
    
    #Combine
    data_comb = {'OL100': data_ol100, 'OL0': data_ol0}
    
    return data_comb

# def combine_cohort_data(data_dir, id_data_dir, cohort_name, fe_method, tumor_frac):
    
#     #norm and nonorm data
#     stnorm0 = get_cohort_data(data_dir, "z_nostnorm_" + cohort_name, fe_method, tumor_frac)
#     stnorm1 = get_cohort_data(data_dir, cohort_name, fe_method, tumor_frac)

#     ################################################
#     # Combine stnorm and nostnorm 
#     ################################################
#     ol100_union = combine_data_from_stnorm_and_nostnorm(stnorm0['OL100'], stnorm1['OL100'], method = 'union')
#     ol0_union   = combine_data_from_stnorm_and_nostnorm(stnorm0['OL0'], stnorm1['OL0'], method = 'union')


#     cdata = {'stnorm0_OL100':     stnorm0['OL100'],
#                 'stnorm0_OL0':       stnorm0['OL0'],
#                 'stnorm1_OL100':     stnorm1['OL100'],
#                 'stnorm1_OL0':       stnorm1['OL0'],
#                 'Union_OL100': ol100_union, 
#                 'Union_OL0':  ol0_union}

    
#     return cdata


def get_matching_tile_index(indata, common_tiles_xy):
    # Get row indices in df1
    match_mask = indata['TILE_COOR_ATLV0'].isin(common_tiles_xy)
    df_match = indata[match_mask]
    df_nomatch = indata[~match_mask]
    match_index = df_match.index.tolist()
    nomatch_index = df_nomatch.index.tolist()
    
    return  match_index, nomatch_index

def get_larger_tumor_fraction_tile(tf1, tf2, match_index1, match_index2):
    
    tensor1 = tf1[match_index1]
    tensor2 = tf2[match_index2]
    
    
    mask_tensor1 = tensor1 >= tensor2
    
    # Get indices where tensor1 "wins"
    indices_tensor1 = mask_tensor1.nonzero(as_tuple=True)[0]
    
    # Get indices where tensor2 "wins"
    indices_tensor2 = (~mask_tensor1).nonzero(as_tuple=True)[0]
    
    final_idx1  = [match_index1[i] for i in indices_tensor1.tolist()]
    final_idx2  = [match_index2[i] for i in indices_tensor2.tolist()]
    
    return final_idx1, final_idx2


# def combine_data_from_stnorm_and_nostnorm(indata_stnorm, indata_stnorm_no, method = 'union'):
    
#     # indata_stnorm = data_ol100_opx_stnorm0
#     # indata_stnorm_no = data_ol100_opx_stnorm1
    
#     comb_data_list = []
#     #get the data with more IDs
#     if len(indata_stnorm) >= len(indata_stnorm_no):
#         main_data = indata_stnorm
#         nomain_data = indata_stnorm_no
#     else:
#         main_data = indata_stnorm_no
#         nomain_data = indata_stnorm

#     for i in range(len(main_data)):

#         # Unpack the tuple
#         features1, labels1, tf1, other_info1, sample_id1, patient_id1 = main_data[i]
        
#         #find the data in indata2
#         index = next((i for i, entry in enumerate(nomain_data) if entry[-2] == sample_id1), None)
        
    
        
#         if index is not None: #if the ID is in both data, take the union or combine
#             features2, labels2, tf2, other_info2, sample_id2, patient_id2 = nomain_data[index]
#             if method == 'combine_all':
#                 comb_data = (
#                     features1,
#                     labels1,
#                     tf1,
#                     other_info1,
#                     sample_id1,
#                     patient_id1,
                    
#                     features2,
#                     labels2,
#                     tf2,
#                     other_info2,
#                     sample_id2,
#                     patient_id2,
#                 )
#             elif method == 'union':
#                 #Intersect
#                 common_tiles = set(other_info1['TILE_COOR_ATLV0']).intersection(other_info2['TILE_COOR_ATLV0'])
#                 match_index1, nomatch_index1 = get_matching_tile_index(other_info1, common_tiles)
#                 match_index2, nomatch_index2 = get_matching_tile_index(other_info2, common_tiles)
#                 match_index1_tokeep, match_index2_tokeep = get_larger_tumor_fraction_tile(tf1, tf2,match_index1, match_index2)
                
#                 final_idx_tokeep1 = match_index1_tokeep + nomatch_index1
#                 final_idx_tokeep2 = match_index2_tokeep + nomatch_index2
                
                
#                 #Get updated info
#                 feature = torch.concat([features1[final_idx_tokeep1], features2[final_idx_tokeep2]])
#                 tf = torch.concat([tf1[final_idx_tokeep1], tf2[final_idx_tokeep2]])
#                 labels = labels1 
#                 other_info = pd.concat([other_info1.iloc[final_idx_tokeep1], other_info2.iloc[final_idx_tokeep2]])
#                 sample_id = sample_id1
#                 patient_id = patient_id1
                
#                 comb_data = (
#                     feature,
#                     labels,
#                     tf,
#                     other_info,
#                     sample_id,
#                     patient_id
#                 )

#         else: #only take all the things in main data
#             if method == 'combine_all':
#                 comb_data = (
#                     features1,
#                     labels1,
#                     tf1,
#                     other_info1,
#                     sample_id1,
#                     patient_id1,
#                     [],
#                     [],
#                     [],
#                     [],
#                     [],
#                     []
                    
#                 )
#             elif method == 'union':
#                  comb_data = (
#                      features1,
#                      labels1,
#                      tf1,
#                      other_info1,
#                      sample_id1,
#                      patient_id1
#                  )
                 
#         comb_data_list.append(comb_data)
        
#     return comb_data_list   


def get_selected_labels(mutation_name, train_cohort):
    
    all_labels = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"]    
    
    if mutation_name == 'MT':
        if 'TCGA' in train_cohort:
            selected = ["AR" , "HR2", "PTEN","RB1","TP53","MSI"]   #without TMB
        else:
            selected = ["AR",  "HR2", "PTEN","RB1","TP53","TMB","MSI"]   
    else:
            selected = mutation_name.split('_')

    #get label index in all labels
    selected_index = []
    for l in selected:
        selected_index.append(all_labels.index(l))
    
    return selected, selected_index

def update_label(indata, select_label_index):
    r'''
    Input:   Model Ready data
    Returns: Model ready data, List of ids
    -------
    '''
    
    #Update label
    indata_final = [(x[0], x[1][:,select_label_index], x[2], x[3],x[4],x[5]) for x in indata]

    print(f'DS: {len(indata_final)}')

    return indata_final


def get_final_model_data_v2(opx, tcga, nep, id_data_dir, train_cohort, mutation, fe_method, tumor_frac, s_fold):
    
    ################################################
    #Get Train, test, val data
    ################################################    
    
    
    train_cohort_map = {
        # #stain normed
        # 'OPX': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': opx['stnorm1_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'TCGA_PRAD': {
        #     'train_cohort1': 'TCGA_PRAD',
        #     'model_data1': tcga['stnorm1_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'Neptune': {
        #     'train_cohort1': 'Neptune',
        #     'model_data1': nep['stnorm1_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        
        # #no stain normed
        # 'z_nostnorm_OPX': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': opx['stnorm0_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'z_nostnorm_TCGA_PRAD': {
        #     'train_cohort1': 'TCGA_PRAD',
        #     'model_data1': tcga['stnorm0_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'z_nostnorm_Neptune': {
        #     'train_cohort1': 'Neptune',
        #     'model_data1': nep['stnorm0_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        
        # #two corhot, no st
        # 'z_nostnorm_OPX_TCGA': {
        #     'train_cohort1': 'z_nostnorm_OPX',
        #     'model_data1': opx['stnorm0_OL100'],
        #     'train_cohort2': 'z_nostnorm_TCGA_PRAD',
        #     'model_data2': tcga['stnorm0_OL100'],
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'z_nostnorm_OPX_NEP': {
        #     'train_cohort1': 'z_nostnorm_OPX',
        #     'model_data1': opx['stnorm0_OL100',
        #     'train_cohort2': 'z_nostnorm_Neptune',
        #     'model_data2': data_nep_stnorm0,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'z_nostnorm_TCGA_NEP': {
        #     'train_cohort1': 'z_nostnorm_TCGA_PRAD',
        #     'model_data1': data_tcga_stnorm0,
        #     'train_cohort2': 'z_nostnorm_Neptune',
        #     'model_data2': data_nep_stnorm0,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        
        # #two corhot, st
        # 'OPX_TCGA': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm1,
        #     'train_cohort2': 'TCGA_PRAD',
        #     'model_data2': data_tcga_stnorm1,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'Neptune',
        #     'ext_data_nost':  data_nep_stnorm0['OL0'],
        #     'ext_data_st':    data_nep_stnorm1['OL0'],
        #     'ext_data_union': data_nep_stnorm10_union['OL0']
        # },
        # 'OPX_NEP': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm1,
        #     'train_cohort2': 'Neptune',
        #     'model_data2': data_nep_stnorm1,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'TCGA_PRAD',
        #     'ext_data_nost':  data_tcga_stnorm0['OL0'],
        #     'ext_data_st':    data_tcga_stnorm1['OL0'],
        #     'ext_data_union': data_tcga_stnorm10_union['OL0']
        # },
        # 'TCGA_NEP': {
        #     'train_cohort1': 'TCGA_PRAD',
        #     'model_data1': data_tcga_stnorm1,
        #     'train_cohort2': 'Neptune',
        #     'model_data2': data_nep_stnorm1,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'OPX',
        #     'ext_data_nost':  data_opx_stnorm0['OL0'],
        #     'ext_data_st':    data_opx_stnorm1['OL0'],
        #     'ext_data_union': data_opx_stnorm10_union['OL0']
        # },
        
        
        #Dict
        #'stnorm0_OL100', 'stnorm0_OL0', 'stnorm1_OL100', 'stnorm1_OL0', 'Union_OL100', 'Union_OL0']

        
        #two cohort, union of st and no-st
        'union_STNandNSTN_OPX_TCGA': {
            'train_cohort1': 'OPX',
            'model_data1': {'OL100': opx['Union_OL100'], 'OL0': opx['Union_OL0']},
            'train_cohort2': 'TCGA_PRAD',
            'model_data2': {'OL100': tcga['Union_OL100'], 'OL0': tcga['Union_OL0']},
            'train_cohort3':  None,
            'model_data3': None,
            'ext_cohort':  'Neptune',
            'ext_data_nost':  nep['stnorm0_OL0'],
            'ext_data_st':    nep['stnorm1_OL0'],
            'ext_data_union': nep['Union_OL0'],

        },
        # 'union_STNandNSTN_OPX_NEP': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm10_union,
        #     'train_cohort2': 'Neptune',
        #     'model_data2': data_nep_stnorm10_union,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'TCGA_PRAD',
        #     'ext_data_nost':  data_tcga_stnorm0['OL0'],
        #     'ext_data_st':    data_tcga_stnorm1['OL0'],
        #     'ext_data_union': data_tcga_stnorm10_union['OL0'],
        # },
        # 'union_STNandNSTN_TCGA_NEP': {
        #     'train_cohort1': 'TCGA_PRAD',
        #     'model_data1': data_tcga_stnorm10_union,
        #     'train_cohort2': 'Neptune',
        #     'model_data2': data_nep_stnorm10_union,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'OPX',
        #     'ext_data_nost':  data_opx_stnorm0['OL0'],
        #     'ext_data_st':    data_opx_stnorm1['OL0'],
        #     'ext_data_union': data_opx_stnorm10_union['OL0'],
        # },
        
        # #three cohort, union of st and no-st
        # 'union_STNandNSTN_OPX_TCGA_NEP': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm10_union,
        #     'train_cohort2': 'TCGA_PRAD',
        #     'model_data2': data_tcga_stnorm10_union,
        #     'train_cohort3':  'Neptune',
        #     'model_data3': data_nep_stnorm10_union,
        #     'ext_cohort':  None,
        #     'ext_data_nost':  None,
        #     'ext_data_st':    None,
        #     'ext_data_union': None,
        # },
        
        
        # #two cohort, combine and later sampling
        # 'comb_STNandNSTN_OPX_TCGA': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm10_comb,
        #     'train_cohort2': 'TCGA_PRAD',
        #     'model_data2': data_tcga_stnorm10_comb,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # }
    }
    
    if train_cohort in train_cohort_map:
        
        config = train_cohort_map[train_cohort]
        train_cohort1 = config['train_cohort1']
        model_data1 = config['model_data1']
        train_cohort2 = config['train_cohort2']
        model_data2 = config['model_data2']
        train_cohort3 = config['train_cohort3']
        model_data3 = config['model_data3']
        ext_cohort = config['ext_cohort']
        ext_data_st0 = config['ext_data_nost']
        ext_data_st1 = config['ext_data_st']
        ext_data_union = config['ext_data_union']
        
        
    else:
        raise ValueError(f"Unknown training cohort: {train_cohort}")
    
    #Get Train, validation and test
    split_data = get_final_split_data(id_data_dir,
                                    train_cohort1,
                                    model_data1, 
                                    train_cohort2, 
                                    model_data2, 
                                    train_cohort3,
                                    model_data3,
                                    tumor_frac, 
                                    s_fold)
    
    train_data, _ = split_data["train"]
    val_data, _ = split_data["val"]
    test_data, _ = split_data["test"]
    test_data1, _ = split_data["test1"]
    test_data2, _ = split_data["test2"]
    test_data3, _ = split_data["test3"]
    

    #Get sanple ID:
    train_ids =  [x[-2] for x in train_data]
    val_ids =  [x[-2] for x in val_data]
    test_ids =  [x[-2] for x in test_data]
    test_ids1 =  [x[-2] for x in test_data1]
    test_ids2 =  [x[-2] for x in test_data2]
    test_ids3 =  [x[-2] for x in test_data3]
    
    if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
        ext_ids0 =  [x[-2] for x in ext_data_st0]
        ext_ids1 =  [x[-2] for x in ext_data_st1]
        ext_ids =  [x[-2] for x in ext_data_union]
    else:
        ext_ids0 = []
        ext_ids1 = []
        ext_ids = []        
    
    
    #Update labels
    selected_labels, selected_label_index = get_selected_labels(mutation, train_cohort)
    print(selected_labels)
    print(selected_label_index)

    train_data = update_label(train_data, selected_label_index)
    val_data = update_label(val_data, selected_label_index)
    test_data = update_label(test_data, selected_label_index)
    test_data1 = update_label(test_data1, selected_label_index)
    test_data2 = update_label(test_data2, selected_label_index)
    test_data3 = update_label(test_data3, selected_label_index)

    if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
        ext_data_st0 = update_label(ext_data_st0, selected_label_index)
        ext_data_st1 = update_label(ext_data_st1, selected_label_index)
        ext_data_union = update_label(ext_data_union, selected_label_index)
    else:
        ext_data_st0 = []
        ext_data_st1 = []
        ext_data_union = []
    
    #update item for model
            
    if train_cohort == 'comb_STNandNSTN_OPX_TCGA': 
        #Keep feature1, label, tf,1 dlabel, feature2, tf2
        train_data = [(item[0], item[1], item[2], item[3], item[7], item[9]) for item in train_data]
        test_data1 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data1]
        test_data2 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data2]
        test_data3 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data3]

        test_data = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data]
        val_data = [(item[0], item[1], item[2],  item[6], item[8]) for item in val_data]
    else:
        #Exclude tile info data, sample ID, patient ID, do not needed it for training
        train_data = [item[:-3] for item in train_data]
        test_data1 = [item[:-3] for item in test_data1]   
        test_data2 = [item[:-3] for item in test_data2]   
        test_data3 = [item[:-3] for item in test_data3]   
        test_data = [item[:-3] for item in test_data]   
        val_data = [item[:-3] for item in val_data]
    
    if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
        ext_data_st0 = [item[:-3] for item in ext_data_st0] #no st norm
        ext_data_st1 = [item[:-3] for item in ext_data_st1] #st normed
        ext_data_union = [item[:-3] for item in ext_data_union]


    
    return {'train': (train_data, train_ids, "Train"),
            'val': (val_data, val_ids, "VAL"),
            'test': (test_data, test_ids,"Test"),
            'test1': (test_data1, test_ids1, train_cohort1),
            'test2': (test_data2, test_ids2, train_cohort2),
            'test3': (test_data3, test_ids3, train_cohort3),
            'ext_data_st0': (ext_data_st0, ext_ids0, ext_cohort),
            'ext_data_st1': (ext_data_st1, ext_ids1, ext_cohort),
            'ext_data_union': (ext_data_union, ext_ids, ext_cohort)
        }, selected_labels





def get_final_split_data(id_data_dir, train_cohort1, model_data1, train_cohort2, model_data2, train_cohort3, model_data3, tumor_frac, s_fold):
    

    (train_data1, train_ids1), (val_data1, val_ids1), (test_data1, test_ids1) = get_train_test_val_data_singlecohort(id_data_dir, 
                                                                                                               train_cohort1 ,
                                                                                                               model_data = model_data1, 
                                                                                                               tumor_frac = tumor_frac, 
                                                                                                               s_fold = s_fold)
    
    
    if train_cohort2 is not None:
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = get_train_test_val_data_singlecohort(id_data_dir, 
                                                                                                                   train_cohort2 ,
                                                                                                                   model_data = model_data2, 
                                                                                                                   tumor_frac = tumor_frac, 
                                                                                                                   s_fold = s_fold)
    
    else:
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = ([], []), ([], []), ([], [])
        
        
    
    if train_cohort3 is not None:
        (train_data3, train_ids3), (val_data3, val_ids3), (test_data3, test_ids3) = get_train_test_val_data_singlecohort(id_data_dir, 
                                                                                                                   train_cohort3 ,
                                                                                                                   model_data = model_data3, 
                                                                                                                   tumor_frac = tumor_frac, 
                                                                                                                   s_fold = s_fold)
    
    else:
        (train_data3, train_ids3), (val_data3, val_ids3), (test_data3, test_ids3) = ([], []), ([], []), ([], [])
    
    
    train_data = train_data1 + train_data2 + train_data3
    train_ids = train_ids1 + train_ids2 + train_ids3
    
    val_data = val_data1 + val_data2 + val_data3
    val_ids = val_ids1 + val_ids2 + val_ids3
    
    test_data = test_data1 + test_data2  + test_data3
    test_ids = test_ids1 + test_ids2 + test_ids3
    
    return {
            "train": (train_data, train_ids),
            "val": (val_data, val_ids),
            "test": (test_data, test_ids),
            "test1": (test_data1, test_ids1),
            "test2": (test_data2, test_ids2),
            "test3": (test_data3, test_ids3)
            }


def get_partial_data(indata, selected_ids):
    r'''
    Input:   Model Ready data pool, List of selected_ids
    Returns: Model ready data for selected Ids
    -------
    '''
    #Get ordered patient IDs in indata
    ids_pool  = [x[-1] for x in indata] #The 2nd to the last one is sample ID, the last one is patient ID

    #Find all index of the selected ids
    inc_idx = [i for x in selected_ids for i, val in enumerate(ids_pool) if val == x]

    #Subsets
    indata_subset = Subset(indata, inc_idx)
    
    #Get final 
    sp_ids_order =  [x[-2] for x in indata_subset] #sample IDs
    pt_ids_order =  [x[-1] for x in indata_subset] #patient IDs

    return indata_subset,sp_ids_order, pt_ids_order


def get_train_test_val_data(data_pool_train, data_pool_test, id_df, fold):

    #Get train, test IDs
    train_ids_pt = list(id_df.loc[id_df['FOLD' + str(fold)] == 'TRAIN', 'PATIENT_ID'])
    test_ids_pt  = list(id_df.loc[id_df['FOLD' + str(fold)] == 'TEST', 'PATIENT_ID'])
    val_ids_pt   = list(id_df.loc[id_df['FOLD' + str(fold)] == 'VALID', 'PATIENT_ID'])
   
    train_data_final, train_sp_ids_final, train_pt_ids_final = get_partial_data(data_pool_train, train_ids_pt)
    test_data_final, test_sp_ids_final, test_pt_ids_final = get_partial_data(data_pool_test, test_ids_pt)
    val_data_final, val_sp_ids_final, val_pt_ids_final = get_partial_data(data_pool_train, val_ids_pt)
    
    print(f'Sample N: Train: {len(set(train_sp_ids_final))}; Test: {len(set(test_sp_ids_final))}; Val: {len(set(val_sp_ids_final))}')
    print(f'Patient N: Train: {len(set(train_pt_ids_final))}; Test: {len(set(test_pt_ids_final))}; Val: {len(set(val_pt_ids_final))}')
    print(f'dataset N: Train: {len(train_data_final)}; Test: {len(test_data_final)}; Val: {len(val_data_final)}')

    return (train_data_final, train_sp_ids_final), (val_data_final, val_sp_ids_final), (test_data_final, test_sp_ids_final)



def get_train_test_val_data_singlecohort(data_dir, cohort_name , model_data, tumor_frac, s_fold):
    
    #data_dir = proj_dir + 'intermediate_data/3B_Train_TEST_IDS'
    
    #Load ID split data
    d_path =  os.path.join(data_dir, cohort_name ,'TFT' + str(tumor_frac))
    train_test_val_id_df = pd.read_csv(os.path.join(d_path, "train_test_split.csv"))
    train_test_val_id_df.rename(columns = {'TMB_HIGHorINTERMEDITATE': 'TMB'}, inplace = True)
    
    
    #Load data
    train_model_data = model_data['OL100']
    test_model_data  = model_data['OL0']

    #get train test data
    (train_data, train_ids), (val_data, val_ids), (test_data, test_ids) = get_train_test_val_data(train_model_data, test_model_data, train_test_val_id_df, s_fold)
    
    #add domain label:
    train_data = [item[:3] + (torch.tensor(1.0),) + item[3:] for item in train_data] #1 for OPX, biopsy, #0 for TCGA, surgical
    
    return (train_data, train_ids), (val_data, val_ids), (test_data, test_ids)





class ModelReadyData_diffdim_V2(Dataset):
    def __init__(self,
                 tile_info_list,
                 selected_features,
                 selected_labels,
                ):

        #Get feature
        self.x =  [torch.FloatTensor(df[selected_features].to_numpy()) for df in tile_info_list]
        
        # Get the Y labels
        self.y =  [torch.FloatTensor(df[selected_labels].drop_duplicates().to_numpy()) for df in tile_info_list]

        #Get tumor fraction
        self.tf = [torch.FloatTensor(df['TUMOR_PIXEL_PERC'].to_numpy()) for df in tile_info_list]
        
        #Get site loc
        self.site_loc = [torch.FloatTensor(df['SITE_LOCAL'].to_numpy()) for df in tile_info_list]

        #Get other info
        self.other_info = [df.drop(columns = selected_features + selected_labels + ['TUMOR_PIXEL_PERC']) for df in tile_info_list]
        
        #Train_orTest info
        self.fold0 = [df['FOLD0'].unique().item() for df in tile_info_list]
        self.fold1 = [df['FOLD1'].unique().item() for df in tile_info_list]
        self.fold2 = [df['FOLD2'].unique().item() for df in tile_info_list]
        self.fold3 = [df['FOLD3'].unique().item() for df in tile_info_list]
        self.fold4 = [df['FOLD4'].unique().item() for df in tile_info_list]
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x  = self.x[index]
        y  = self.y[index]
        tf = self.tf[index]
        site_loc = self.site_loc[index]
        of = self.other_info[index]
        sp_id = of['SAMPLE_ID'].unique().item()
        pt_id = of['PATIENT_ID'].unique().item()
        f0 = self.fold0[index]
        f1 = self.fold1[index]
        f2 = self.fold2[index]
        f3 = self.fold3[index]
        f4 = self.fold4[index]
        
        return {
        "x": x,
        "y": y,
        "tumor_fraction": tf,
        "site_location": site_loc,
        "tile_info": of,
        "sample_id": sp_id,
        "patient_id": pt_id,
        "fold0": f0,
        "fold1": f1,
        "fold2": f2,
        "fold3": f3,
        "fold4": f4
        }
                


class ModelReadyData_diffdim_withclusterinfo(Dataset):
    def __init__(self,
                 feature_list,
                 label_list,
                 tumor_info_list,
                 cluster_list,
                ):
        
        self.x =[torch.FloatTensor(feature) for feature in feature_list] 
        
        # Get the Y labels
        self.y = [torch.FloatTensor(label) for label in label_list] 

        self.tf = [torch.FloatTensor(tf) for tf in tumor_info_list] 

        self.c = [torch.FloatTensor(c) for c in cluster_list] 
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        tf= self.tf[index]
        c= self.c[index]
        
        return x, y, tf,c

class ModelReadyData_Instance_based(Dataset):
    def __init__(self,
                 feature_data,
                 label_data,
                ):
            
        self.x = feature_data
        
        # Get the Y labels
        self.y = label_data
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        
        return x, y

def modify_to_instance_based(indata):
    feature_list = []
    label_list = []
    for data_it, data in enumerate(indata):
        cur_labels = data[1]
        label_list.append(cur_labels.repeat(data[0].shape[0], 1))
        feature_list.append(data[0].squeeze())
    
    feature_data = torch.concat(feature_list, dim = 0) #torch.Size([N_TILES, 1536])
    label_data = torch.concat(label_list, dim = 0) #torch.Size([N_TILES, 7])

    indata_instance_based = ModelReadyData_Instance_based(feature_data, label_data)

    return indata_instance_based
    
class add_tile_xy(Dataset):
    def __init__(self, original_dataset, additional_component):
        self.original_dataset = original_dataset
        self.additional_component = additional_component

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_sample = self.original_dataset[idx]
        tile_xy = torch.tensor(list(self.additional_component[idx]['TILE_XY_INDEXES'].apply(self.str_to_tuple)))

        # Assuming original_sample is a tuple
        return original_sample + (tile_xy,)

    # Function to convert string coordinates to tuples
    def str_to_tuple(self,coord_str):
        coord_str = coord_str.strip('()')
        x, y = map(int, coord_str.split(','))
        return (x, y)
    

def check_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        #print(f"Missing columns: {missing}")
        return True
    return False



def get_sample_feature(folder_name, feature_path, fe_method, cancer_info_path):    
    #Input dir
    input_dir = os.path.join(feature_path, 
                             folder_name, 
                             'features', 
                             f"features_alltiles_{fe_method}.h5")
    
    #feature
    feature_df = pd.read_hdf(input_dir, key='feature')
    feature_df.columns = feature_df.columns.astype(str)
    feature_df.reset_index(drop = True, inplace = True)
    
    #Get all tile info
    id_df = pd.read_hdf(input_dir, key='tile_info') #Tile ID (Do not use the labels in this, only use the tile info, because the label has been updated in all_tile_info_df from 3_otherinfo)
    id_df.reset_index(drop = True, inplace = True)
    
    #Combine
    if id_df.shape[0] != feature_df.shape[0]:
        raise ValueError("id_df and feature_df must have the same number of rows.")
    
    
    combined_df = pd.concat([id_df, feature_df], axis=1)
    
    
    #Check if tile info contains this 
    missing = check_columns(combined_df, ['pred_map_location', 'TUMOR_PIXEL_PERC'])
    
    if missing: #if missing get it from cancer detection
        cancer_info_dir = os.path.join(cancer_info_path, 
                                         folder_name, 
                                         'ft_model')
        file_name = [f for f in os.listdir(cancer_info_dir) if f.endswith("_TILE_TUMOR_PERC.csv")][0]
        cancer_df = pd.read_csv(os.path.join(cancer_info_dir, file_name))
        
        if combined_df.shape[0] != cancer_df.shape[0]:
            raise ValueError("combined_df and cancer_df must have the same number of rows.")
        
        combined_df = combined_df.merge(cancer_df, how = 'left', on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 
                                                                       'PIXEL_OVERLAP','LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 
                                                                       'WHITE_SPACE','TISSUE_COVERAGE'])
        
  
    cols_to_keep = list(feature_df.columns) + ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 
                                         'PIXEL_OVERLAP','LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 
                                         'WHITE_SPACE','TISSUE_COVERAGE', 'pred_map_location', 'TUMOR_PIXEL_PERC']
    combined_df = combined_df[cols_to_keep] 

        
        
    return combined_df



def combine_features_label_allsamples(selected_ids, feature_path, fe_method, TUMOR_FRAC_THRES, all_label_df, id_col, cancer_info_path):
    
    all_comb = []
    ct = 0
    for pt in selected_ids:
        
        if ct % 10 == 0 : print(ct)
        #Get feature
        feature_df = get_sample_feature(pt, feature_path, fe_method, cancer_info_path)    
        
        #Select tumor fraction > X tiles
        feature_df = feature_df.loc[feature_df['TUMOR_PIXEL_PERC'] >= TUMOR_FRAC_THRES].copy()
        feature_df.reset_index(inplace = True, drop = True)
        
        #Get label df 
        label_df = all_label_df.loc[all_label_df[id_col] == pt]
        
        #Combine label
        comb_df = feature_df.merge(label_df, how = 'left', on = ["SAMPLE_ID"])
        
        
        all_comb.append(comb_df)
        ct += 1
    
    all_comb_df = pd.concat(all_comb)
    
    return all_comb_df, all_comb

def get_model_ready_data(datalist, fold_name = 'fold0', data_type = 'TRAIN', selected_label = 'HR1', concat_tf  = False):

    #Get label
    all_labels = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"]
    label_idx = all_labels.index(selected_label) 
    
    #subset
    subsets = [item for item in datalist if item[fold_name] == data_type]
    
    if concat_tf == False:
        data_tensor = [(item['x'], 
                        item['y'][:,[label_idx]],
                        item['tumor_fraction'],
                        item['site_location']) for item in subsets]
    else:
        data_tensor = [(torch.concat([item['x'], item['tumor_fraction'].unsqueeze(1)], dim = 1), 
                        item['y'][:,[label_idx]],
                        item['tumor_fraction'],
                        item['site_location']) for item in subsets]
        

    sample_ids      = [item['sample_id'] for item in subsets]
    patient_ids     = [item['patient_id'] for item in subsets]
    corhor_names     = [re.split(r"[-_]", item['patient_id'])[0] for item in subsets]
    
    coords = [np.array(item['tile_info']['TILE_XY_INDEXES'].apply(ast.literal_eval).tolist()) for item in subsets]
    
    return data_tensor, sample_ids, patient_ids, corhor_names, coords

def find_missing_groups(ds, idx_map):
    """
    Find sample_ids whose mapped group key does NOT exist in ds._f["cases"].
    idx_map is the result of your index_sid_to_idx or index_sid_to_key.
    """
    ds._ensure()
    cases = ds._f["cases"]
    existing_keys = set(cases.keys())

    missing = []
    for sid, key in idx_map.items():
        key = str(key)  # ensure string
        if key not in existing_keys:
            missing.append((sid, key))

    return missing


def load_dataset_splits(ol100, ol0, fold, label, concat_tf = False):
    
    """Load train, validation, and test splits for a dataset."""
    train, train_sp_ids, train_pt_ids, train_cohorts, train_coords = get_model_ready_data(
        ol100, f'fold{fold}', 'TRAIN', selected_label=label, concat_tf = concat_tf
    )
    val, val_sp_ids, val_pt_ids, val_cohorts, val_coords = get_model_ready_data(
        ol100, f'fold{fold}', 'VALID', selected_label=label, concat_tf = concat_tf
    )
    test, test_sp_ids, test_pt_ids, test_cohorts, test_coords = get_model_ready_data(
        ol0, f'fold{fold}', 'TEST', selected_label=label, concat_tf = concat_tf
    )
    
    return {
        "train": (train, train_sp_ids, train_pt_ids, train_cohorts, train_coords),
        "val":   (val, val_sp_ids, val_pt_ids, val_cohorts, val_coords),
        "test":  (test, test_sp_ids, test_pt_ids, test_cohorts, test_coords),
    }



# --- prep helper: unique tile key + remember original row index
def prep(df, tf, dataset_id, key_cols=('SAMPLE_ID','PATIENT_ID','TILE_XY_INDEXES', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',)):
    out = df.copy().reset_index(drop=True)
    out['original_row_idx'] = out.index
    out['dataset'] = dataset_id
    # tf: torch.Tensor or numpy/Series -> ensure Series for concat
    if isinstance(tf, torch.Tensor):
        tf = tf.detach().cpu().numpy()
    out['tumor_fraction'] = pd.Series(tf).reset_index(drop=True)
    # build a robust string key; adjust key_cols to your schema
    out['__key__'] = out[list(key_cols)].astype(str).agg('§'.join, axis=1)
    return out

def merge_data_lists(list1, list2, merge_type = 'union'):
    """
    For each tile, keep the feature from normed or not normed data, keep the one with the highest tumor fraction
    Keep, for each tile key, the row coming from d1 or d2 with the higher tumor_fraction.
    per-tile tensors (x, tumor_fraction, site_location) are aligned row-wise with tile_info.
    """

    #Convert data list into a dict keyed by sample id
    dict1 = {d['sample_id']: d for d in list1}
    dict2 = {d['sample_id']: d for d in list2}

    merged_list = []
    
    #All sample id
    all_ids = list(set(dict1.keys()) | set(dict2.keys()))
    all_ids.sort()

    for sid in all_ids:
                
        d1 = dict1.get(sid)
        d2 = dict2.get(sid)

        if d1 and d2: #if both exsit                
            
            if merge_type == 'union':
                #add tf and original row idx
                c1 = prep(d1['tile_info'], d1['tumor_fraction'], 1)
                c2 = prep(d2['tile_info'], d2['tumor_fraction'], 2)
                both = pd.concat([c1, c2], ignore_index=True)
                
                #pick the row with max tumor_fraction per tile key
                best_idx = both.groupby('__key__')['tumor_fraction'].idxmax() # idx of the max tumor_fraction within each key group
                chosen = both.loc[best_idx].reset_index(drop=True)
                
                # --- split chosen by source to slice tensors efficiently and in order
                chosen1 = chosen[chosen['dataset'] == 1]
                chosen2 = chosen[chosen['dataset'] == 2]
                idx1 = chosen1['original_row_idx'].tolist()
                idx2 = chosen2['original_row_idx'].tolist()
                
                
                # --- slice tensors (handle empty-index cases)
                parts_x = []
                parts_tf = []
                parts_sl = []
                ti_parts = []
            
                if len(idx1):
                    parts_x.append(d1['x'][idx1])
                    parts_tf.append(d1['tumor_fraction'][idx1])
                    parts_sl.append(d1['site_location'][idx1])
                    ti_parts.append(d1['tile_info'].iloc[idx1])
                    
                if len(idx2):
                    parts_x.append(d2['x'][idx2])
                    parts_tf.append(d2['tumor_fraction'][idx2])
                    parts_sl.append(d2['site_location'][idx2])
                    ti_parts.append(d2['tile_info'].iloc[idx2])
                    
                    
                X = torch.cat(parts_x, dim=0) if len(parts_x) > 1 else (parts_x[0] if parts_x else torch.empty(0))
                TF = torch.cat(parts_tf, dim=0) if len(parts_tf) > 1 else (parts_tf[0] if parts_tf else torch.empty(0))
                SL = torch.cat(parts_sl, dim=0) if len(parts_sl) > 1 else (parts_sl[0] if parts_sl else torch.empty(0))
                TI = pd.concat(ti_parts, axis=0).reset_index(drop=True)


        
            # --- final merged sample dict
            merged = {
                'x': X,
                'y': d1.get('y', d2.get('y')),  #keep whichever is available
                'tumor_fraction': TF,
                'site_location': SL,
                'tile_info': TI,                 # does NOT include helper columns
                'sample_id': d1.get('sample_id', d2.get('sample_id')),
                'patient_id': d1.get('patient_id', d2.get('patient_id')),
                'fold0': d1.get('fold0', d2.get('fold0')),
                'fold1': d1.get('fold1', d2.get('fold1')),
                'fold2': d1.get('fold2', d2.get('fold2')),
                'fold3': d1.get('fold3', d2.get('fold3')),
                'fold4': d1.get('fold4', d2.get('fold4')),
            }
            
            merged_list.append(merged)


    return merged_list


def index_sid_to_idx(ds):
    ds._ensure()                     # ensures ds._f is an open h5py.File
    idx = {}
    cases = ds._f["cases"]

    for k in cases.keys():           # keys are "0", "1", "2", ...
        g = cases[k]
        sid = g.attrs["sample_id"]   # read the attribute only (very cheap)
        idx[sid] = int(k)

    return idx


def merge_sample(d1, d2, merge_type = "union"):
    
    if merge_type == 'union':
        #add tf and original row idx
        c1 = prep(d1['tile_info'], d1['tumor_fraction'], 1)
        c2 = prep(d2['tile_info'], d2['tumor_fraction'], 2)
        both = pd.concat([c1, c2], ignore_index=True)
        
        #pick the row with max tumor_fraction per tile key
        best_idx = both.groupby('__key__')['tumor_fraction'].idxmax() # idx of the max tumor_fraction within each key group
        chosen = both.loc[best_idx].reset_index(drop=True)
        
        # --- split chosen by source to slice tensors efficiently and in order
        chosen1 = chosen[chosen['dataset'] == 1]
        chosen2 = chosen[chosen['dataset'] == 2]
        idx1 = chosen1['original_row_idx'].tolist()
        idx2 = chosen2['original_row_idx'].tolist()
        
        
        # --- slice tensors (handle empty-index cases)
        parts_x = []
        parts_tf = []
        parts_sl = []
        ti_parts = []
    
        if len(idx1):
            parts_x.append(d1['x'][idx1])
            parts_tf.append(d1['tumor_fraction'][idx1])
            parts_sl.append(d1['site_location'][idx1])
            ti_parts.append(d1['tile_info'].iloc[idx1])
            
        if len(idx2):
            parts_x.append(d2['x'][idx2])
            parts_tf.append(d2['tumor_fraction'][idx2])
            parts_sl.append(d2['site_location'][idx2])
            ti_parts.append(d2['tile_info'].iloc[idx2])
            
            
        X = torch.cat(parts_x, dim=0) if len(parts_x) > 1 else (parts_x[0] if parts_x else torch.empty(0))
        TF = torch.cat(parts_tf, dim=0) if len(parts_tf) > 1 else (parts_tf[0] if parts_tf else torch.empty(0))
        SL = torch.cat(parts_sl, dim=0) if len(parts_sl) > 1 else (parts_sl[0] if parts_sl else torch.empty(0))
        TI = pd.concat(ti_parts, axis=0).reset_index(drop=True)



    # --- final merged sample dict
    merged = {
        'x': X,
        'y': d1.get('y', d2.get('y')),  #keep whichever is available
        'tumor_fraction': TF,
        'site_location': SL,
        'tile_info': TI,                 # does NOT include helper columns
        'sample_id': d1.get('sample_id', d2.get('sample_id')),
        'patient_id': d1.get('patient_id', d2.get('patient_id')),
        'fold0': d1.get('fold0', d2.get('fold0')),
        'fold1': d1.get('fold1', d2.get('fold1')),
        'fold2': d1.get('fold2', d2.get('fold2')),
        'fold3': d1.get('fold3', d2.get('fold3')),
        'fold4': d1.get('fold4', d2.get('fold4')),
    }
    
    return merged
            
    

def merge_data_lists_h5(h5_case1, h5_case2, merge_type = 'union'):
    """
    same as merge_data_list, works for h5
    """

    #get ID to IDX map
    idx_a = index_sid_to_idx(h5_case1)
    idx_b = index_sid_to_idx(h5_case2)
    
    all_ids = sorted(set(idx_a) | set(idx_b))  #union of the two ids
    
    merged_list = []
    for sid in tqdm(all_ids, desc="Merging samples"):
        d1 = h5_case1[idx_a[sid]] if sid in idx_a else None
        d2 = h5_case2[idx_b[sid]] if sid in idx_b else None
        
        
        if d1 and d2: #if both exsits
            merged = merge_sample(d1, d2, merge_type = merge_type)
            merged_list.append(merged)

    return merged_list



def save_merged_samples_to_h5(merged_samples, out_path):
    """
    Save a list (or iterable) of merged sample dicts into an HDF5 file
    with the same structure expected by H5Cases:

        /cases/<idx>/
            x
            y
            tumor_fraction
            site_location
            other_info_csv
            attrs: sample_id, patient_id, fold0..4
    """
    def _to_numpy(arr):
        # Handle torch tensors and numpy arrays
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _write_tile_info(group, df: pd.DataFrame):
        # Store tile_info DataFrame as CSV bytes, like your loader expects
        csv_str = df.to_csv(index=False)
        csv_bytes = np.frombuffer(csv_str.encode("utf-8"), dtype='uint8')
        group.create_dataset("other_info_csv", data=csv_bytes)

    with h5py.File(out_path, "w") as f:
        cases = f.create_group("cases")

        # tqdm progress bar
        for i, sample in tqdm(
            enumerate(merged_samples),
            total=len(merged_samples),
            desc="Writing merged HDF5"
        ):
            g = cases.create_group(str(i))

            # --- main tensors/arrays
            g.create_dataset("x",              data=_to_numpy(sample["x"]))
            g.create_dataset("y",              data=_to_numpy(sample["y"]))
            g.create_dataset("tumor_fraction", data=_to_numpy(sample["tumor_fraction"]))
            g.create_dataset("site_location",  data=_to_numpy(sample["site_location"]))

            # --- tile_info as CSV bytes
            _write_tile_info(g, sample["tile_info"])

            # --- attributes
            g.attrs["sample_id"]  = sample["sample_id"]
            g.attrs["patient_id"] = sample["patient_id"]
            g.attrs["fold0"] = sample["fold0"]
            g.attrs["fold1"] = sample["fold1"]
            g.attrs["fold2"] = sample["fold2"]
            g.attrs["fold3"] = sample["fold3"]
            g.attrs["fold4"] = sample["fold4"]



def h5_to_list(ds):
    return [ds[i] for i in tqdm(range(len(ds)), desc="Loading H5 cases")]

def _merge_one_sid(sid, idx_a, idx_b, h5_case1, h5_case2, merge_type):
    
    d1 = h5_case1[idx_a[sid]] if sid in idx_a else None
    d2 = h5_case2[idx_b[sid]] if sid in idx_b else None
    
    if not (d1 and d2):
        return None

    return merge_sample(d1, d2, merge_type)  

def merge_data_lists_h5_parallel(h5_case1, h5_case2, merge_type='union', max_workers=None):
    idx_a = index_sid_to_idx(h5_case1)
    idx_b = index_sid_to_idx(h5_case2)

    # iterate only ids that yield merged results (both exist)
    all_ids = list(idx_a.keys() & idx_b.keys())

    # choose a reasonable default
    if max_workers is None:
        max_workers = max(2, (os.cpu_count() or 8))

    results = [None] * len(all_ids)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2pos = {
            ex.submit(_merge_one_sid, sid, idx_a, idx_b, h5_case1, h5_case2, merge_type): pos
            for pos, sid in enumerate(all_ids)
        }
        # for fut in tqdm(as_completed(fut2pos), total=len(fut2pos)):  # progress bar (optional)
        for fut in tqdm(as_completed(fut2pos), total=len(fut2pos), desc="Merging samples", unit="sample"):
            pos = fut2pos[fut]
            r = fut.result()
            results[pos] = r

    # keep original order, drop None
    return [r for r in results if r is not None]



def merge_data_lists_h5_parallel_v2(h5_case1, h5_case2, merge_type="union", max_workers=4):
    idx_a = index_sid_to_idx(h5_case1)
    idx_b = index_sid_to_idx(h5_case2)
    all_ids = sorted(set(idx_a) | set(idx_b))

    def worker(sid):
        if sid in idx_a and sid in idx_b:
            d1 = h5_case1[idx_a[sid]]
            d2 = h5_case2[idx_b[sid]]
            return merge_sample(d1, d2, merge_type)
        return None

    merged = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, out in enumerate(ex.map(worker, all_ids)):
            if i % 10 == 0:
                print(i)
            if out is not None:
                merged.append(out)
    return merged


def merge_data_lists_h5_serial(h5_case1, h5_case2, merge_type='union'):
    idx_a = index_sid_to_idx(h5_case1)
    idx_b = index_sid_to_idx(h5_case2)
    all_ids = list(idx_a.keys() & idx_b.keys())

    results = []
    for sid in tqdm(all_ids, desc="Merging samples", unit="sample"):
        r = _merge_one_sid(sid, idx_a, idx_b, h5_case1, h5_case2, merge_type)
        if r is not None:
            results.append(r)
    return results




def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def merge_data_lists_h5_to_file(h5_case1, h5_case2, out_path, merge_type="union", max_workers=4):
    idx_a = index_sid_to_idx(h5_case1)
    idx_b = index_sid_to_idx(h5_case2)
    all_ids = sorted(set(idx_a) | set(idx_b))

    start = time.time()
    def worker(args):
        sid, idx = args
        if sid in idx_a and sid in idx_b:
            d1 = h5_case1[idx_a[sid]]
            d2 = h5_case2[idx_b[sid]]
            merged = merge_sample(d1, d2, merge_type)
            # return what we need to write
            return sid, idx, merged, d1, d2
        return None

    with h5py.File(out_path, "w") as fout:
        cases = fout.create_group("cases")

        # optional: store how many we planned to write
        cases.attrs["planned_n"] = len(all_ids)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, result in enumerate(ex.map(worker, [(sid, i) for i, sid in enumerate(all_ids)])):
                if result is None:
                    continue
                sid, i, merged, d1, d2 = result

                g = cases.create_group(str(i))
                # --- datasets ---
                g.create_dataset("x", data=_to_np(merged["x"]), compression="gzip", compression_opts=4, shuffle=True)
                g.create_dataset("y", data=_to_np(merged["y"]), compression="gzip", compression_opts=4, shuffle=True)
                g.create_dataset("tumor_fraction", data=_to_np(merged["tumor_fraction"]))
                g.create_dataset("site_location",  data=_to_np(merged["site_location"]))

                # Save CSV info as bytes
                if hasattr(merged["tile_info"], "to_csv"):
                    csv_bytes = merged["tile_info"].to_csv(index=False).encode("utf-8")
                else:
                    # if already raw CSV string/bytes
                    csv_bytes = (merged["tile_info"].encode("utf-8")
                                 if isinstance(merged["tile_info"], str)
                                 else bytes(merged["tile_info"]))
                g.create_dataset("other_info_csv", data=np.frombuffer(csv_bytes, dtype=np.uint8))

                # --- attributes (make file compatible with H5Cases) ---
                g.attrs["sample_id"] = sid
                # prefer values on merged; else fall back to d1/d2; else default
                g.attrs["patient_id"] = (merged.get("patient_id", None)
                                         or d1.get("patient_id", None)
                                         or d2.get("patient_id", ""))

                for k in ("fold0","fold1","fold2","fold3","fold4"):
                    val = (merged.get(k, None) if isinstance(merged, dict) else None)
                    if val is None: val = d1.get(k, None)
                    if val is None: val = d2.get(k, None)
                    if val is None: val = -1
                    g.attrs[k] = val

                if i % 10 == 0:
                    print(f"{i}/{len(all_ids)} written in {time.time()-start:.2f}s")

    print(f"✅ Done writing merged HDF5 to {out_path} ({time.time()-start:.2f}s)")


    

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]





def combine_all(split, keys=('test', 'train', 'val')):
    parts = [split[k] for k in keys]
    data  = list(chain.from_iterable(p[0] for p in parts))
    sp    = list(chain.from_iterable(p[1] for p in parts))
    pt    = list(chain.from_iterable(p[2] for p in parts))
    coh   = list(chain.from_iterable(p[3] for p in parts))
    return data, sp, pt, coh

def just_test(split):
    a, b, c, d, _ = split['test']
    return a, b, c, d



def downsample(train_data, n_times=10, n_samples=100, seed=None):
    """
    Downsample positives and negatives with reproducibility option.
    
    Args:
        train_data (list): dataset [(x, label), ...]
        n_times (int): how many resamples
        n_samples (int): number of positives/negatives to select each time
        seed (int or None): random seed for reproducibility
    Returns:
        list of lists: each element is a balanced dataset
    """
    pos_data = [x for x in train_data if x[1] == 1]
    neg_data = [x for x in train_data if x[1] == 0]

    if seed is not None:
        random.seed(seed)

    all_samples = []
    for i in range(n_times):
        selected1 = random.sample(pos_data, n_samples)   # sample positives
        selected0 = random.sample(neg_data, n_samples)   # sample negatives
        selected_all = selected1 + selected0
        random.shuffle(selected_all)
        all_samples.append(selected_all)
    return all_samples


#unifomrm sampling form Harry
def _uniform_sample(coords: np.ndarray, feats: np.ndarray, max_bag: int, seed: int = None, grid: int = 32) -> tuple:
    
    rng = np.random.default_rng(seed)
    
   
    # If fewer tiles than max_bag, just return them all
    N = len(coords)
    if N  <= max_bag:
        chosen = np.arange(N, dtype=int)
        return feats[chosen], coords[chosen], chosen
    
    # --- normal uniform sampling ---

    mins, maxs = coords.min(0), coords.max(0)

    norm   = (coords - mins) / (maxs - mins + 1e-8)

    bins   = np.floor(norm * grid).astype(np.int16)

    keys   = bins[:, 0] * grid+ bins[:, 1]

    #order  = np.random.permutation(len(keys))
    order  = rng.permutation(len(keys))  # use seeded RNG

    chosen, seen = [], set()

    for idx in order:

        k = keys[idx]

        if k not in seen:

            seen.add(k); chosen.append(idx)

            if len(chosen) == max_bag:

                break

    if len(chosen) < max_bag:        # pad if needed

        rest = np.setdiff1d(np.arange(len(keys)), chosen, assume_unique=True)
        
        extra = np.random.choice(rest, max_bag-len(chosen), replace=False)

        chosen = np.concatenate([chosen, extra])

    chosen = np.asarray(chosen, dtype=int)

    return feats[chosen], coords[chosen], chosen



#Weighted spatial  unifornm sampling focus on cancer tiles
def _uniform_sample_with_cancer_prob(
    coords: np.ndarray,
    feats: np.ndarray,
    cancer_prob: np.ndarray,
    max_bag: int,
    seed: int = None,
    grid: int = 32,
    strength: float = 1.0,
    blend_uniform: float = 0.05,
) -> tuple:
    """
    Weighted spatial sampling:
      1) build GRID bins to encourage spatial diversity (<=1 pick/bin in the first pass)
      2) iterate tiles in a *probability-weighted* random order (higher prob first on average)
      3) pad remaining slots (if any) with weighted picks from the rest

    Args:
        coords: (N, 2) or (N, D) integer/float tile coordinates.
        feats:  (N, F) features aligned with coords.
        cancer_prob: (N,) or (N,1) per-tile probabilities in [0,1].
        max_bag: target number of tiles to sample.
        seed: RNG seed for reproducibility.
        grid: grid resolution used to enforce spatial diversity.
        strength: exponent on probabilities; >1 sharpens (more aggressive toward high prob),
                  <1 flattens. =0 becomes uniform (after blending).
        blend_uniform: small epsilon to ensure nonzero mass and keep some exploration.

    Returns:
        feats[chosen], coords[chosen], chosen_indices (np.int64)
    """
    rng = np.random.default_rng(seed)

    N = len(coords)
    if N == 0:
        return feats[:0], coords[:0], np.asarray([], dtype=np.int64)

    if N <= max_bag:
        chosen = np.arange(N, dtype=np.int64)
        return feats[chosen], coords[chosen], chosen

    # --- prep probabilities ---
    p = np.asarray(cancer_prob).reshape(-1)
    if p.shape[0] != N:
        raise ValueError(f"cancer_prob length {p.shape[0]} != N={N}")

    # sanitize: clip, replace NaNs, power 'strength', and blend with uniform so no zeros
    p = np.nan_to_num(p, nan=0.0)
    p = np.clip(p, 0.0, None)
    if strength != 1.0:
        # if all zeros, p**strength still zeros; blending below handles this.
        p = np.power(p, max(strength, 0.0))
    if not np.isfinite(p).all() or p.sum() == 0:
        # fallback to uniform
        p = np.ones_like(p, dtype=float)
    # blend with uniform mass
    if blend_uniform > 0:
        u = np.full_like(p, 1.0 / N)
        p = (1.0 - blend_uniform) * p + blend_uniform * u
    # normalize
    p = p / p.sum()

    # --- grid binning (same as your function, parametric grid) ---
    mins, maxs = coords.min(0), coords.max(0)
    norm = (coords - mins) / (maxs - mins + 1e-8)
    bins = np.floor(norm * grid).astype(np.int16)
    # allow coords with >2 dims: use first two for keys
    b0 = bins[:, 0]
    b1 = bins[:, 1] if bins.shape[1] > 1 else np.zeros_like(b0)
    keys = (b0 * grid + b1).astype(np.int32)

    # --- probability-weighted permutation without replacement ---
    # numpy supports p with replace=False (interpreted as successive draws proportional to p)
    order = rng.choice(N, size=N, replace=False, p=p)

    # --- first pass: one per bin, in weighted order ---
    chosen = []
    seen = set()
    for idx in order:
        k = int(keys[idx])
        if k not in seen:
            seen.add(k)
            chosen.append(idx)
            if len(chosen) == max_bag:
                break

    # --- pad if needed (still weighted, from the rest) ---
    if len(chosen) < max_bag:
        chosen = np.asarray(chosen, dtype=np.int64)
        mask = np.ones(N, dtype=bool)
        mask[chosen] = False
        rest_idx = np.nonzero(mask)[0]
        if rest_idx.size > 0:
            # renormalize probabilities over the remaining pool
            p_rest = p[rest_idx]
            p_rest = p_rest / p_rest.sum()
            extra = rng.choice(rest_idx, size=(max_bag - len(chosen)), replace=False, p=p_rest)
            chosen = np.concatenate([chosen, extra.astype(np.int64)], axis=0)
        else:
            # should be rare; just return what we have
            chosen = chosen.astype(np.int64)

    # stable dtype and order (optional shuffle to avoid deterministic bin sequence)
    # chosen = rng.permutation(chosen)  # uncomment if you want final order shuffled
    return feats[chosen], coords[chosen], chosen




#run uniform sampke for all selected samples
def uniform_sample_all_samples(indata, incoords, max_bag = 100, grid = 32, sample_by_tf = True, plot = False, tf_threshold = 0.0):
    
    new_data_list = []
    for data_item, coord_item in zip(indata,incoords):            
        
        #get feature
        feats = data_item[0] #(N_tiles, 1536)
        label = data_item[1]
        tfs  = data_item[2]
        sl  = data_item[3]
        #get coordiantes
        coords = coord_item
        
        
        # keep a copy of full coords for plotting
        coords_all = coords.copy()

        # apply threshold filtering if requested
        if tf_threshold is not None:
            mask = tfs >= tf_threshold
            feats = feats[mask]
            tfs = tfs[mask]
            sl = sl[mask]
            coords = coords[mask]
        
            # # skip if nothing remains after filtering
            # if len(feats) == 0:
            #     continue


        #uniform sampling
        if sample_by_tf == False:
            sampled_feats, sampled_coords, sampled_index = _uniform_sample(coords, feats, max_bag, grid = grid, seed = 1)
        else:
            sampled_feats, sampled_coords, sampled_index = _uniform_sample_with_cancer_prob(coords, feats, tfs, max_bag, 
                                                                                            seed = 1, 
                                                                                            grid = grid, 
                                                                                            strength = 1.0)

        if plot:
            # 3. Plot results
            plt.figure(figsize=(8, 8))
            plt.scatter(coords_all[:, 0], -coords_all[:, 1], alpha=0.3, label="All Tiles") #- for  flip Y
            plt.scatter(sampled_coords[:, 0], -sampled_coords[:, 1], color="red", label="Sampled Tiles") #- for  flip Y
            plt.legend()
            plt.title("Uniform Sampling with Grid Constraint")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
        
        
        sampled_tf = tfs[sampled_index]
        sampled_site_loc =sl[sampled_index]
        new_tuple = (sampled_feats,label, sampled_tf, sampled_site_loc)
        new_data_list.append(new_tuple)
        
        
    return new_data_list


def uniform_sample_all_samples_h5(indata, incoords, max_bag = 100, grid = 32, sample_by_tf = True, plot = False, tf_threshold = 0.0):
    
    new_data_list = []
    for data_item, coord_item in zip(indata,incoords):            
        
        #get feature
        feats = data_item['x'] #(N_tiles, 1536)
        label = data_item['y']
        tfs  = data_item['tumor_fraction']
        sl  = data_item['site_location']
        #get coordiantes
        coords = coord_item
        
        
        # keep a copy of full coords for plotting
        coords_all = coords.copy()

        # apply threshold filtering if requested
        if tf_threshold is not None:
            mask = tfs >= tf_threshold
            feats = feats[mask]
            tfs = tfs[mask]
            sl = sl[mask]
            coords = coords[mask]
        
            # # skip if nothing remains after filtering
            # if len(feats) == 0:
            #     continue


        #uniform sampling
        if sample_by_tf == False:
            sampled_feats, sampled_coords, sampled_index = _uniform_sample(coords, feats, max_bag, grid = grid, seed = 1)
        else:
            sampled_feats, sampled_coords, sampled_index = _uniform_sample_with_cancer_prob(coords, feats, tfs, max_bag, 
                                                                                            seed = 1, 
                                                                                            grid = grid, 
                                                                                            strength = 1.0)

        if plot:
            # 3. Plot results
            plt.figure(figsize=(8, 8))
            plt.scatter(coords_all[:, 0], -coords_all[:, 1], alpha=0.3, label="All Tiles") #- for  flip Y
            plt.scatter(sampled_coords[:, 0], -sampled_coords[:, 1], color="red", label="Sampled Tiles") #- for  flip Y
            plt.legend()
            plt.title("Uniform Sampling with Grid Constraint")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
        
        
        sampled_tf = tfs[sampled_index]
        sampled_site_loc =sl[sampled_index]
        new_tuple = (sampled_feats,label, sampled_tf, sampled_site_loc)
        new_data_list.append(new_tuple)
        
        
    return new_data_list


def filter_by_tumor_fraction(data, threshold=0.9):
    filtered_data = []
    masks = []
    for features, label, tumor_fraction, site_loc in data:
        mask = tumor_fraction >= threshold
        if mask.any():
            filtered_features = features[mask]
            filtered_tumor_fraction = tumor_fraction[mask]
            filtered_site_loc = site_loc[mask]
            filtered_data.append((filtered_features, label, filtered_tumor_fraction, filtered_site_loc))
            masks.append(mask)
        else:
            filtered_data.append((features, label, tumor_fraction, site_loc))
            mask_all_true = torch.ones(len(tumor_fraction), dtype=torch.bool)
            masks.append(mask_all_true)
    return filtered_data,masks





def write_h5_dataset(path, dataset):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    index = []  # rows: one per case
    with h5py.File(path, "w", libver="latest") as f:
        f.attrs["version"] = "v2"
        grp = f.create_group("cases")

        for i in range(len(dataset)):
            # pull directly from your lists to avoid __getitem__ overhead
            x  = dataset.x[i].numpy()
            y  = dataset.y[i].numpy()
            tf = dataset.tf[i].numpy()
            sl = dataset.site_loc[i].numpy()
            of = dataset.other_info[i]  # pandas DF

            case = grp.create_group(str(i))
            # chunk along the first dim to match batch access
            chunk0 = min(len(x), 1024) if x.ndim > 0 else 1

            case.create_dataset("x", data=x, chunks=(chunk0,)+x.shape[1:], compression="lzf", shuffle=True)
            case.create_dataset("y", data=y, compression="lzf", shuffle=True)
            case.create_dataset("tumor_fraction", data=tf, compression="lzf", shuffle=True)
            case.create_dataset("site_location", data=sl, compression="lzf", shuffle=True)

            # Save other_info as CSV bytes inside HDF5 (or write sidecar parquet files, see Option B)
            other_csv = of.to_csv(index=False).encode("utf-8")
            case.create_dataset("other_info_csv", data=np.frombuffer(other_csv, dtype="uint8"))

            # scalar attrs
            case.attrs["fold0"] = dataset.fold0[i]
            case.attrs["fold1"] = dataset.fold1[i]
            case.attrs["fold2"] = dataset.fold2[i]
            case.attrs["fold3"] = dataset.fold3[i]
            case.attrs["fold4"] = dataset.fold4[i]
            case.attrs["sample_id"] = str(of["SAMPLE_ID"].unique().item())
            case.attrs["patient_id"] = str(of["PATIENT_ID"].unique().item())

            index.append({"i": i})


class H5Cases(Dataset):
    def __init__(self, h5_path):
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            self.n = len(f["cases"])
        self._f = None

    def _ensure(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self): return self.n

    def __getitem__(self, i):
        self._ensure()
        g = self._f["cases"][str(i)]
        x  = torch.from_numpy(g["x"][...])         # reads only needed chunks
        y  = torch.from_numpy(g["y"][...])
        tf = torch.from_numpy(g["tumor_fraction"][...])
        sl = torch.from_numpy(g["site_location"][...])

        # reconstruct other_info from bytes
        csv_bytes = bytes(g["other_info_csv"][...].tobytes()).decode("utf-8")
        other = pd.read_csv(io.StringIO(csv_bytes))

        return {
            "x": x, "y": y,
            "tumor_fraction": tf,
            "site_location": sl,
            "tile_info": other,
            "sample_id": g.attrs["sample_id"],
            "patient_id": g.attrs["patient_id"],
            "fold0": g.attrs["fold0"],
            "fold1": g.attrs["fold1"],
            "fold2": g.attrs["fold2"],
            "fold3": g.attrs["fold3"],
            "fold4": g.attrs["fold4"],
        }
    
  
def get_fold_subset(dataset, fold_key="fold0", fold_value="TRAIN"):
    """
    Return a torch.utils.data.Subset containing only samples whose HDF5
    attribute `fold_key` equals `fold_value`.

    This function does NOT load the tensors into memory—it only inspects
    the HDF5 attributes.
    """
    dataset._ensure()  # make sure the HDF5 file handle is open
    indices = []
    for i in range(len(dataset)):
        attr = dataset._f["cases"][str(i)].attrs[fold_key]
        # normalize bytes to str if needed
        if isinstance(attr, bytes):
            attr = attr.decode("utf-8")
        if attr == fold_value:
            indices.append(i)
    return Subset(dataset, indices)


def get_train_test_valid_h5(h5_data, fold_num):
    train_data = get_fold_subset(h5_data, 'fold' + str(fold_num), "TRAIN")
    test_data  = get_fold_subset(h5_data, 'fold' + str(fold_num), "TEST")
    val_data   = get_fold_subset(h5_data, 'fold' + str(fold_num), "VALID")
    
    return  train_data, test_data, val_data
    


# ----- config -----
ATOL = 1e-6
RTOL = 1e-5

def t2np(t):
    # convert torch tensor -> numpy (CPU) without copying if possible
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t

def allclose(a, b, name):
    if isinstance(a, torch.Tensor): a = a.detach().cpu()
    if isinstance(b, torch.Tensor): b = b.detach().cpu()
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False, f"{name}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        ok = torch.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True)
        return ok, ("" if ok else f"{name}: values differ (rtol={RTOL}, atol={ATOL}, equal_nan={True})")
    else:
        # numpy path
        import numpy as np
        a = np.asarray(a); b = np.asarray(b)
        if a.shape != b.shape:
            return False, f"{name}: shape {a.shape} != {b.shape}"
        ok = np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
        return ok, ("" if ok else f"{name}: values differ (rtol={RTOL}, atol={ATOL}, equal_nan={True})")

def df_equal(df1: pd.DataFrame, df2: pd.DataFrame, name="other_info"):
    # we stored CSV in H5, so dtypes may differ slightly. Compare values ignoring dtype.
    try:
        # align columns if order differs
        if set(df1.columns) != set(df2.columns):
            return False, f"{name}: column sets differ: {set(df1.columns) ^ set(df2.columns)}"
        df2 = df2[df1.columns]
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
            check_dtype=False,
            check_like=True,       # allow column reordering
        )
        return True, ""
    except AssertionError as e:
        return False, f"{name}: {str(e).splitlines()[0]}"

def read_other_info_from_h5(group):
    csv_bytes = bytes(group["other_info_csv"][...].tobytes()).decode("utf-8")
    return pd.read_csv(io.StringIO(csv_bytes))

def compare_case(i, ds_pth, g_h5):
    mismatches = []

    # tensors/arrays
    ok, msg = allclose(ds_pth.x[i], torch.from_numpy(g_h5["x"][...]), "x")
    if not ok: mismatches.append(msg)

    ok, msg = allclose(ds_pth.y[i], torch.from_numpy(g_h5["y"][...]), "y")
    if not ok: mismatches.append(msg)

    ok, msg = allclose(ds_pth.tf[i], torch.from_numpy(g_h5["tumor_fraction"][...]), "tumor_fraction")
    if not ok: mismatches.append(msg)

    ok, msg = allclose(ds_pth.site_loc[i], torch.from_numpy(g_h5["site_location"][...]), "site_location")
    if not ok: mismatches.append(msg)

    # DataFrame
    df_pth = ds_pth.other_info[i]
    df_h5 = read_other_info_from_h5(g_h5)
    ok, msg = df_equal(df_pth, df_h5, "other_info")
    if not ok: mismatches.append(msg)

    # scalar attrs
    # sample_id / patient_id from pth side:
    sp_pth = df_pth["SAMPLE_ID"].unique().item()
    pt_pth = df_pth["PATIENT_ID"].unique().item()
    sp_h5 = g_h5.attrs["sample_id"]
    pt_h5 = g_h5.attrs["patient_id"]
    if str(sp_pth) != str(sp_h5):
        mismatches.append(f"sample_id: {sp_pth} != {sp_h5}")
    if str(pt_pth) != str(pt_h5):
        mismatches.append(f"patient_id: {pt_pth} != {pt_h5}")

    # Folds (string comparison)
    folds_pth = [
        ds_pth.fold0[i],
        ds_pth.fold1[i],
        ds_pth.fold2[i],
        ds_pth.fold3[i],
        ds_pth.fold4[i],
    ]
    folds_h5 = [
        g_h5.attrs["fold0"],
        g_h5.attrs["fold1"],
        g_h5.attrs["fold2"],
        g_h5.attrs["fold3"],
        g_h5.attrs["fold4"],
    ]
    
    # Normalize to string to avoid any subtle dtype differences (e.g., np.string_ vs str)
    folds_pth = [str(v) for v in folds_pth]
    folds_h5  = [str(v) for v in folds_h5]
    
    if folds_pth != folds_h5:
        mismatches.append(f"folds: {folds_pth} != {folds_h5}")

    return mismatches



import numpy as np

def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def _is_string_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or series.dtype == object

def _write_tile_info(group: h5py.Group, df: pd.DataFrame):
    """Store a DataFrame column-wise under group 'tile_info'."""
    tig = group.create_group('tile_info')
    # Keep column order
    tig.attrs['columns'] = np.array(df.columns.tolist(), dtype=h5py.string_dtype('utf-8'))
    dtypes = {}
    for col in df.columns:
        s = df[col]
        if _is_string_dtype(s):
            # Convert to UTF-8 varlen strings
            dt = h5py.string_dtype('utf-8')
            data = s.fillna("").astype(str).to_numpy()
            tig.create_dataset(col, data=data, dtype=dt, compression="gzip", shuffle=True)
            dtypes[col] = 'str'
        else:
            data = np.asarray(s.to_numpy())
            tig.create_dataset(col, data=data, compression="gzip", shuffle=True)
            dtypes[col] = str(s.dtype)
    # save dtype hints for reconstruction
    tig.attrs['dtypes'] = np.array([f"{k}:{v}" for k,v in dtypes.items()], dtype=h5py.string_dtype('utf-8'))

def _read_tile_info(group: h5py.Group) -> pd.DataFrame:
    """Example of how to read it back."""
    tig = group['tile_info']
    cols = [c for c in tig.attrs['columns']]
    out = {}
    for col in cols:
        out[col] = tig[col][()]  # np array
    df = pd.DataFrame(out)
    return df



def load_merged_from_h5(path: str):
    """Tiny loader showing how to get things back (per sample)."""
    out = []
    with h5py.File(path, "r") as f:
        for sid in f.keys():
            g = f[sid]
            item = {
                'sample_id': sid,
                'x': g['x'][()] if 'x' in g else None,
                'tumor_fraction': g['tumor_fraction'][()] if 'tumor_fraction' in g else None,
                'site_location': g['site_location'][()] if 'site_location' in g else None,
                'y': g['y'][()] if 'y' in g else None,
                'tile_info': _read_tile_info(g) if 'tile_info' in g else None,
                'patient_id': g.attrs.get('patient_id', None),
                'fold0': g.attrs.get('fold0', None),
                'fold1': g.attrs.get('fold1', None),
                'fold2': g.attrs.get('fold2', None),
                'fold3': g.attrs.get('fold3', None),
                'fold4': g.attrs.get('fold4', None),
            }
            out.append(item)
    return out
