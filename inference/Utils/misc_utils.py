#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 17:41:03 2025

@author: jliu6
"""

import random
import torch
import numpy as np
import os
import pandas as pd
from PIL import ImageCms, Image
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, '../Utils/RandomSplit-main/')
from RandomSplit import MakeBalancedCrossValidation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    
def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")
      
        
def get_ids(path, include=None, exclude=None):
    r'''
    Get IDs in the path

    '''
    ids = []
    for x in os.listdir(path):
        if x == ".DS_Store":
            continue
        if include and include not in x:
            continue
        if exclude and exclude in x:
            continue
        if x.endswith((".svs", ".tif",".ndpi")):   # covers both extensions
            base, _ = os.path.splitext(x)  
            ids.append(base)
        else: #for TCGA, because it is the folder name, not the slide name
            ids.append(x)
            
    return ids

        
# Min-max normalization function
def minmax_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor



def convert_img(in_img):
    srgb_profile = ImageCms.createProfile("sRGB")
    converted_img = ImageCms.profileToProfile(in_img, srgb_profile, srgb_profile)

    return converted_img



def count_mutation_perTrainTest(train_test_df, selected_label):

    # Compute N (count) per TRAIN/TEST group
    train_df = train_test_df[train_test_df['TRAIN_OR_TEST'] == 'TRAIN']
    test_df  = train_test_df[train_test_df['TRAIN_OR_TEST'] == 'TEST']
    
    n_train = len(train_df)
    n_test = len(test_df)
    
    # Initialize result storage
    results = []
    
    for col in selected_label:
        train_n = train_df[col].sum()
        test_n = test_df[col].sum()
        
        train_pct = (train_n / n_train) * 100 if n_train > 0 else 0
        test_pct = (test_n / n_test) * 100 if n_test > 0 else 0
        
        # Format as "N (%)"
        train_formatted = f"{train_n} ({train_pct:.1f}%)"
        test_formatted = f"{test_n} ({test_pct:.1f}%)"
        
        results.append({
            'Outcome': col,
            'Train N (%)': train_formatted,
            'Test N (%)': test_formatted
        })
    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(results)

    
    return results_df



def count_mutation_perTrainTestVAL(train_test_df, selected_label, n_Folds = 5):
    
    # train_test_df = train_test_valid_df
    # selected_label = SELECTED_LABEL
    
    all_res = []
    for k in range(n_Folds):
        #Update dataframe                                                 
        train_df = train_test_df.loc[train_test_df['FOLD' + str(k)] == 'TRAIN']
        test_df = train_test_df.loc[train_test_df['FOLD' + str(k)] == 'TEST']
        val_df = train_test_df.loc[train_test_df['FOLD' + str(k)] == 'VALID']
                                                                        
    
        n_train = len(train_df)
        n_test = len(test_df)
        n_val = len(val_df)
    
        # Initialize result storage
        results = []
        
        for col in selected_label:
            train_n = train_df[col].sum()
            test_n = test_df[col].sum()
            val_n = val_df[col].sum()
            
            train_pct = (train_n / n_train) * 100 if n_train > 0 else 0
            test_pct = (test_n / n_test) * 100 if n_test > 0 else 0
            val_pct = (val_n / n_val) * 100 if n_test > 0 else 0
            
            # Format as "N (%)"
            train_formatted = f"{train_n} ({train_pct:.1f}%)"
            test_formatted = f"{test_n} ({test_pct:.1f}%)"
            val_formatted = f"{val_n} ({val_pct:.1f}%)"
            
            results.append({
                'Outcome': col,
                'Train N (%)': train_formatted,
                'Val N (%)': val_formatted,
                'Test N (%)': test_formatted,
                
            })
        # Convert to DataFrame for nice display
        results_df = pd.DataFrame(results)
        results_df['FOLD'] = k
        all_res.append(results_df)
    
    all_res_df = pd.concat(all_res)

    
    return all_res_df


def get_pos_neg_ids(tile_info_pt, label_name):
    #Postive IDs
    pos_ids = list(tile_info_pt.loc[tile_info_pt[label_name] == 1 , 'PATIENT_ID'].unique()) #24
    
    #Neg Ids
    neg_ids = list(tile_info_pt.loc[tile_info_pt[label_name] == 0 , 'PATIENT_ID'].unique()) #242
    
    return pos_ids, neg_ids




def generate_balanced_cv_list(patient_label_data, selected_label, n_Folds = 5, p_test = 0.25):
    
    # patient_label_data = tile_info_df_pt.copy()
    # selected_label = SELECTED_LABEL.copy()
    # n_Folds = 5
    # p_test=0.25
        
    #Start
    label_df = patient_label_data[selected_label].copy()
    label_df.index = patient_label_data['PATIENT_ID']
    label_df['self_count'] = 1 #Add self count for the cases has zero mutations, so the randomsplit algorithm would run

    
    #Get input
    w  = label_df.T
    ids = list(w.columns)
    w = np.array(w)
    column_map = {i: ids[i] for i in range(w.shape[1])}

    training_lists, validation_lists, test_list, res, res_test = MakeBalancedCrossValidation(w, n_Folds, column_map, testing_size=p_test, tries=10)
    
    return training_lists, validation_lists, test_list



def mutation_sample_summary(df_subset, mut_cols):
    sample_total = len(df_subset)
    if sample_total == 0:
        # Return zeros if the subset is empty (avoid divide-by-zero)
        out = pd.DataFrame(index=mut_cols, columns=["Mutated_n", "Total_n", "Percent", "n(%)"])
        out["Mutated_n"] = 0
        out["Total_n"] = 0
        out["Percent"] = 0.0
        out["n(%)"] = "0 (0.0%)"
        return out

    sample_counts = df_subset[mut_cols].sum(numeric_only=True)
    sample_perc   = (sample_counts / sample_total * 100).round(1)

    summary = pd.DataFrame({
        "Mutated_n": sample_counts.astype(int),
        "Total_n": sample_total,
        "Percent": sample_perc
    })
    summary["n(%)"] = summary.apply(lambda x: f"{x.Mutated_n:.0f} ({x.Percent:.1f}%)", axis=1)
    return summary






def count_num_tiles(indata, cohort_name):
    n_tiles = [x['x'].shape[0] for x in indata]
    ids = [x['sample_id'] for x in indata]
    labels = [x['y'].squeeze().numpy() for x in indata]
    
    sample_df = pd.DataFrame({'SAMPLIE_ID':ids, 'N_TILES': n_tiles})
    label_df = pd.DataFrame(labels, columns=[f"LABEL_{i}" for i in range(len(labels[0]))])
    label_df.columns = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"]
    sample_df = pd.concat([sample_df.reset_index(drop=True),
                           label_df.reset_index(drop=True)], axis=1)

    
    df = pd.DataFrame({'cohort_name': cohort_name,
                    'AVG_N_TILES': np.mean(n_tiles).round(),
                    'Median_N_TILES': np.median(n_tiles).round(),
                  'MAX_N_TILES': max(n_tiles),
                  'MIN_N_TILES': min(n_tiles)}, index = [0])
    
    return df,sample_df


def plot_n_tiles_by_labels(df, label_cols=None, value_col="N_TILES", agg="mean", save_path=None):
    """
    Plot grouped bar charts of N_TILES by binary label columns.
    """
    if label_cols is None:
        label_cols = ["AR", "HR1", "HR2", "PTEN", "RB1", "TP53", "TMB", "MSI"]

    grouped = {}
    for col in label_cols:
        if agg == "mean":
            grouped[col] = df.groupby(col)[value_col].mean()
        elif agg == "sum":
            grouped[col] = df.groupby(col)[value_col].sum()
        else:
            raise ValueError("agg must be 'mean' or 'sum'")

    grouped_df = pd.DataFrame(grouped)

    ax = grouped_df.plot(kind="bar", figsize=(10, 6))
    plt.title(f"{agg.capitalize()} {value_col} by Label Group")
    plt.ylabel(f"{agg.capitalize()} {value_col}")
    plt.xlabel("Label value (0 or 1)")
    plt.xticks(rotation=0)
    plt.legend(title="Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()