#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:32:29 2024

@author: jliu6
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,fbeta_score,average_precision_score, precision_recall_curve
from sklearn import metrics
import torchvision
import numpy as np
from misc_utils import minmax_normalize
import torch
from torchmetrics import ROC
import os
from torcheval.metrics import BinaryAUROC, BinaryAUPRC


from sklearn.isotonic import IsotonicRegression
from ACMIL import compute_loss_singletask

def compute_performance(y_true,y_pred_prob,y_pred_class, cohort_name):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel() #CM
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
    
    # Find best threshold using Youden's J
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    
    # Average precision score = PR-AUC
    PR_AUC = average_precision_score(y_true, y_pred_prob)
    
    precision_, recall_, thresholds = precision_recall_curve(y_true, y_pred_prob)
    best_thresh_prauc = sorted(list(zip(np.abs(precision_ - recall_), thresholds)), key=lambda i: i[0], reverse=False)[0][1]

    AUC = round(metrics.auc(fpr, tpr),2)
    ACC = round(accuracy_score(y_true, y_pred_class),2)
    F1 = round(f1_score(y_true, y_pred_class),2)
    F2 = round(fbeta_score(y_true, y_pred_class,beta = 2),2)
    F3 = round(fbeta_score(y_true, y_pred_class,beta = 3),2)
    Recall = round(recall_score(y_true, y_pred_class),2)
    Specificity = round(tn / (tn + fp),2)
    Precision = round(precision_score(y_true, y_pred_class),2)
    npv = round(tn / (tn + fn),2)
    FPRate = round(fp / (fp + tn), 2)   # False Positive Rate

    
    perf_tb = pd.DataFrame({"best_thresh": best_thresh,
                            "best_thresh_prauc": best_thresh_prauc,
                            "AUC": AUC,
                            "PR_AUC":PR_AUC,
                            "Recall": Recall,
                            "Precision":Precision,
                            "NPV": npv,
                            "Specificity":Specificity,
                            "False_Positive_Rate": FPRate,
                            "ACC": ACC,
                            "F1": F1,
                            "F2": F2,
                            "F3": F3,
                            "TP": tp,
                            "TN": tn,
                            "FP": fp,
                            "FN": fn},index = [cohort_name])
    
    return perf_tb

        



def plot_LOSS (train_loss, valid_loss, outdir):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(valid_loss,label="Validation")
    plt.plot(train_loss,label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(outdir + 'LOSS.png')
    
    
def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




def compute_performance_each_label(selected_label, prediction_df, prediction_type):
    
    perf_list = []
    for mut in selected_label:
        cur_pred_df = prediction_df.loc[prediction_df['OUTCOME'] == mut]
        cur_perf_df = compute_performance(cur_pred_df['Y_True'],
                                          cur_pred_df['Pred_Prob'],
                                          cur_pred_df['Pred_Class'],
                                          prediction_type)
        cur_perf_df['OUTCOME'] = mut
        perf_list.append(cur_perf_df)
    
    comb_perf = pd.concat(perf_list)

    return comb_perf

def get_performance(y_predprob, y_true, cohort_ids, outcome, THRES):

    #Prediction df
    pred_df = pd.DataFrame({"SAMPLE_IDs":  cohort_ids, 
                            "Y_True": y_true, 
                            "Pred_Prob" :  y_predprob,
                            "OUTCOME": outcome})
        
    pred_df['Pred_Class'] = 0
    pred_df.loc[pred_df['Pred_Prob'] > THRES,'Pred_Class'] = 1


    perf_df = compute_performance_each_label([outcome], pred_df, "SAMPLE_LEVEL")

    return pred_df, perf_df


def get_performance_withlogit(y_predprob, y_true, y_logit, cohort_ids, outcome, THRES):

    #Prediction df
    pred_df = pd.DataFrame({"SAMPLE_IDs":  cohort_ids, 
                            "Y_True": y_true, 
                            "Pred_Prob" :  y_predprob,
                            "Pred_logit" :  y_logit,
                            "OUTCOME": outcome})
        
    pred_df['Pred_Class'] = 0
    pred_df.loc[pred_df['Pred_Prob'] > THRES,'Pred_Class'] = 1


    perf_df = compute_performance_each_label([outcome], pred_df, "SAMPLE_LEVEL")

    return pred_df, perf_df



def get_attention_and_tileinfo(pt_label_df, patient_att_score):    
    #Get label
    pt_label_df.reset_index(drop = True, inplace = True)

    #Get attention
    cur_att  = pd.DataFrame({'ATT':list(minmax_normalize(patient_att_score))})
    cur_att.reset_index(drop = True, inplace = True)

    #Comb
    cur_att_df = pd.concat([pt_label_df,cur_att], axis = 1)
    cur_att_df.reset_index(drop = True, inplace = True)

    return cur_att_df






def plot_roc_curve(y_pred, y_true, outdir, cohort_name, outcome_name):
    # Initialize ROC metric for binary classification
    roc = ROC(task='binary')
    
    # Compute FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc(torch.tensor(y_pred), torch.tensor(y_true,dtype=torch.int32))
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outdir, cohort_name + "_rocauc_"+ outcome_name + ".png"), dpi=300)
    plt.show()




def calibrate_probs_isotonic(y_val, y_prob_val, y_test_prob):
    """
    Perform Isotonic Regression to calibrate predicted probabilities.

    Args:
        y_val: Ground truth labels from the validation set.
        y_prob_val: Model-predicted probabilities from the validation set.
        y_test_prob: Model-predicted probabilities for the test set.

    Returns:
        Calibrated probabilities for the test set.
    """
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(y_prob_val, y_val)
    calibrated_probs = iso_reg.predict(y_test_prob)
    return calibrated_probs


def bootstrap_ci_from_df(df, y_true_col='y_true', y_pred_col=None, y_prob_col=None,
                         num_bootstrap=1000, ci=95, seed=None, cohort_name = ""):
    """
    Compute bootstrap confidence interval for a metric using predictions in a DataFrame.

    Args:
        df: DataFrame with prediction results.
        metric_fn: Metric function. Can accept (y_true, y_pred) or (y_true, y_prob).
        y_true_col: Column name for ground truth.
        y_pred_col: Column name for predicted labels (used for accuracy, etc.).
        y_prob_col: Column name for predicted probabilities (used for AUROC, etc.).
        num_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., 95 for 95% CI).
        seed: Optional random seed.

    Returns:
        (lower_bound, upper_bound) of CI.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    perf_list = []

    for _ in range(num_bootstrap):
        sample = df.sample(n=n, replace=True, random_state=rng.integers(0, 1e6))
        y_true = sample[y_true_col].values
        y_pred_class =  sample[y_pred_col].values
        y_pred_prob =   sample[y_prob_col].values
        perf_tb = compute_performance(y_true,y_pred_prob,y_pred_class, cohort_name)
        
        perf_list.append(perf_tb)
    perf_df = pd.concat(perf_list)

    mean_values = perf_df.mean()
    lower_bounds = perf_df.quantile((100 - ci) / 200)
    upper_bounds = perf_df.quantile(1 - (100 - ci) / 200)
    
    formatted_results = {
        column: f"{mean_values[column]:.2f} [{lower_bounds[column]:.2f} - {upper_bounds[column]:.2f}]"
        for column in perf_df.columns
    }
    ci_df = pd.DataFrame.from_dict(formatted_results, orient="index", columns=["Mean [CI Low, CI High]"])

    return ci_df.T


def bootstrap_ci_from_df_stratified(
    df,
    y_true_col='y_true',
    y_pred_col=None,
    y_prob_col=None,
    num_bootstrap=1000,
    ci=95,
    seed=None,
    cohort_name="",
    stratified=True,
):
    """
    Compute bootstrap mean and confidence intervals for performance metrics
    using predictions in a DataFrame. Optionally uses stratified bootstrapping
    (recommended for imbalanced data).

    Args:
        df: DataFrame with prediction results.
        y_true_col: Column name for ground truth (binary labels, e.g., {0,1}).
        y_pred_col: Column name for predicted labels (used for accuracy, etc.).
        y_prob_col: Column name for predicted probabilities (used for AUROC, etc.).
        num_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., 95 for 95% CI).
        seed: Optional random seed.
        cohort_name: Passed through to `compute_performance`.
        stratified: If True, sample within each class to preserve class presence.

    Returns:
        A 1-row DataFrame with "Mean [CI Low, CI High]" strings per metric.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        raise ValueError("Empty dataframe provided.")

    # Basic checks
    if y_true_col not in df.columns:
        raise ValueError(f"Column '{y_true_col}' not found in df.")
    if (y_pred_col is None) and (y_prob_col is None):
        raise ValueError("Provide at least one of y_pred_col or y_prob_col.")

    # Ensure we actually have two classes if we plan to compute class-based metrics
    cls_counts = df[y_true_col].value_counts()
    if stratified and len(cls_counts) < 2:
        raise ValueError(
            "Stratified bootstrapping requires at least two classes present in the data."
        )

    # Pre-split indices by class for stratified resampling
    if stratified:
        class_to_indices = {
            cls: df.index[df[y_true_col] == cls].to_numpy()
            for cls in cls_counts.index
        }
        class_to_n = {cls: int(count) for cls, count in cls_counts.items()}

    perf_list = []

    for _ in range(num_bootstrap):
        if stratified:
            # Sample WITH replacement within each class, preserving original class counts
            sampled_idxs_parts = []
            for cls, idxs in class_to_indices.items():
                k = class_to_n[cls]
                # np.random.Generator.choice is reproducible and fast
                draw = rng.choice(idxs, size=k, replace=True)
                sampled_idxs_parts.append(draw)
            sampled_idxs = np.concatenate(sampled_idxs_parts)
            # Optional shuffle to avoid any ordering artifacts
            rng.shuffle(sampled_idxs)
            sample = df.loc[sampled_idxs]
        else:
            # Vanilla bootstrap (may produce zero-positive/zero-negative samples)
            # Use numpy choice for RNG control rather than pandas .sample()
            draw = rng.choice(df.index.to_numpy(), size=n, replace=True)
            sample = df.loc[draw]

        # Pull columns (guard against None)
        y_true = sample[y_true_col].values
        y_pred_class = sample[y_pred_col].values if y_pred_col is not None else None
        y_pred_prob  = sample[y_prob_col].values if y_prob_col is not None else None

        # Your metric aggregator; expected to return a DataFrame/Series per call
        # e.g., columns like ["accuracy", "auc", "precision", "recall", ...]
        perf_tb = compute_performance(y_true, y_pred_prob, y_pred_class, cohort_name)

        # Make sure each entry is a DataFrame row
        if isinstance(perf_tb, pd.Series):
            perf_tb = perf_tb.to_frame().T
        perf_list.append(perf_tb)

    perf_df = pd.concat(perf_list, ignore_index=True)

    # Aggregate mean and percentile CI per metric
    alpha = (100 - ci) / 2.0
    lower_q = alpha / 100.0
    upper_q = 1.0 - lower_q

    mean_values = perf_df.mean(numeric_only=True)
    lower_bounds = perf_df.quantile(lower_q, numeric_only=True)
    upper_bounds = perf_df.quantile(upper_q, numeric_only=True)

    # Format results as "mean [low - high]"
    formatted_results = {
        col: f"{mean_values[col]:.2f} [{lower_bounds[col]:.2f} - {upper_bounds[col]:.2f}]"
        for col in mean_values.index
    }
    ci_df = pd.DataFrame.from_dict(
        formatted_results, orient="index", columns=["Mean [CI Low, CI High]"]
    ).T

    return ci_df




def boxplot_predprob_by_mutationclass(pred_df, cohort_name, outdir, prob_col = "Pred_Prob"):
    # Get unique outcomes
    outcomes = pred_df['OUTCOME'].unique()
    n_outcomes = len(outcomes)
    
    # Create data structure for plotting
    data_to_plot = []
    labels = []
    positions = []
    
    for i, outcome in enumerate(outcomes):
        for y_val in [0, 1]:
            group = pred_df[(pred_df['OUTCOME'] == outcome) & (pred_df['Y_True'] == y_val)]
            data_to_plot.append(group[prob_col].values)
            labels.append(f"{outcome}\nY={y_val}")
            positions.append(i * 3 + y_val)  # spread boxes for each group
    
    # Plotting
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightcoral'] * n_outcomes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xticks(positions, labels, rotation=0)
    plt.ylim(0, 1)
    plt.ylabel('Predicted Probability')
    plt.title('Predicted Probabilities by Outcome and Y_True')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, cohort_name + "_pred_prob_boxplot_byoutcome.png"), dpi=300)
    plt.show()


def get_performance_alltask(n_task, ytrue_test, ypredprob_test, ids_test, selected_label, calibration = False, ytrue_val = None, ypredprob_val = None):
    
    pred_df_list = []
    perf_df_list = []
    for i in range(n_task):

        if calibration == True:
            #Calibration
            pred_prob = calibrate_probs_isotonic(ytrue_val[i], ypredprob_val[i], ypredprob_test[i])
        else:
            pred_prob = ypredprob_test[i]
            
        pred_df, perf_df = get_performance(pred_prob, ytrue_test[i], ids_test, selected_label[i], THRES = np.quantile(ypredprob_test[i],0.5))
        pred_df_list.append(pred_df)
        perf_df_list.append(perf_df)
        
    pred_df_all = pd.concat(pred_df_list)
    perf_df_all = pd.concat(perf_df_list)
    
    return pred_df_all,perf_df_all

def get_performance_alltask_withlogit(n_task, ytrue_test, ypredprob_test, y_logit_test, ids_test, selected_label, calibration = False, ytrue_val = None, ypredprob_val = None):
    
    pred_df_list = []
    perf_df_list = []
    for i in range(n_task):

        if calibration == True:
            #Calibration
            pred_prob = calibrate_probs_isotonic(ytrue_val[i], ypredprob_val[i], ypredprob_test[i])
        else:
            pred_prob = ypredprob_test[i]
            
        pred_logit = y_logit_test[i]
            
        pred_df, perf_df = get_performance_withlogit(pred_prob, ytrue_test[i], pred_logit, ids_test, selected_label[i], THRES = np.quantile(ypredprob_test[i],0.5))
        pred_df_list.append(pred_df)
        perf_df_list.append(perf_df)
        
    pred_df_all = pd.concat(pred_df_list)
    perf_df_all = pd.concat(perf_df_list)
    
    return pred_df_all,perf_df_all


def predict_v2(net, data_loader, device, conf, criterion_da = None):    
    y_pred = []
    y_true = []
    y_pred_prob = []
    # Set the network to evaluation mode
    net.eval()
    with torch.no_grad():
        for data in data_loader:
            image_patches = data[0].to(device, dtype=torch.float32)
            label_lists = data[1][0]
            
            if criterion_da is not None:
                sub_preds_list, slide_preds_list, attn_list, _, _ = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
            else:
                sub_preds_list, slide_preds_list, attn_list, bag_feat_list = net(image_patches)
            
            #Compute loss for each task, then sum
            pred_list = []
            pred_prob_list = []
            for k in range(conf.n_task):
                sub_preds = sub_preds_list[k]
                slide_preds = slide_preds_list[k]
                attn = attn_list[k]
                labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
                pred_prob = torch.sigmoid(slide_preds)
                pred = pred_prob[0][0].round()
                pred_list.append(pred)
                pred_prob_list.append(pred_prob)
        
            y_pred.append(pred_list)
            y_true.append(label_lists)
            y_pred_prob.append(pred_prob_list)
    
        #Get prediction for each task
        y_predprob_task = []
        y_pred_tasks = []
        y_true_tasks = []
        for k in range(conf.n_task):
            y_pred_tasks.append([p[k] for p in y_pred])
            y_predprob_task.append([p[k].item() for p in y_pred_prob])
            y_true_tasks.append([t[:,k].to(device, dtype = torch.int64).item() for t in y_true])
    
    return y_pred_tasks, y_predprob_task, y_true_tasks


def predict_v2_logit(net, data_loader, device, conf, criterion_da = None):    
    
    y_pred = []
    y_true = []
    y_pred_prob = []
    y_logit = []
    # Set the network to evaluation mode
    net.eval()
    with torch.no_grad():
        for data in data_loader:
            image_patches = data[0].to(device, dtype=torch.float32)
            label_lists = data[1][0]
            
            if criterion_da is not None:
                sub_preds_list, slide_preds_list, attn_list, _, _ = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
            else:
                sub_preds_list, slide_preds_list, attn_list, bag_feat_list = net(image_patches)
            
            #Compute loss for each task, then sum
            pred_list = []
            pred_prob_list = []
            logit_list = []
            for k in range(conf.n_task):
                sub_preds = sub_preds_list[k]
                slide_preds = slide_preds_list[k]
                attn = attn_list[k]
                labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
                pred_prob = torch.sigmoid(slide_preds)
                pred = pred_prob[0][0].round()
                pred_list.append(pred)
                pred_prob_list.append(pred_prob)
                logit_list.append(slide_preds)
        
            y_pred.append(pred_list)
            y_true.append(label_lists)
            y_pred_prob.append(pred_prob_list)
            y_logit.append(logit_list)
    
        #Get prediction for each task
        y_predprob_task = []
        y_pred_tasks = []
        y_true_tasks = []
        y_logit_tasks = []
        for k in range(conf.n_task):
            y_pred_tasks.append([p[k] for p in y_pred])
            y_predprob_task.append([p[k].item() for p in y_pred_prob])
            y_logit_tasks.append([p[k].item() for p in y_logit])
            y_true_tasks.append([t[:,k].to(device, dtype = torch.int64).item() for t in y_true])
    
    return y_pred_tasks, y_predprob_task, y_true_tasks,y_logit_tasks




def predict_v2_sp_nost_andst(net, data_loader, device, conf, header, criterion_da = None):    
    y_pred = []
    y_true = []
    y_pred_prob = []
    
    rs_rate_stnorm = 1.0
    rs_rate_nostnorm = 1.0
    # Set the network to evaluation mode
    net.eval()
    with torch.no_grad():
        for data in data_loader:
            
            torch.manual_seed(0)
            
            if len(data[0]) != 0:
                image_patches1 = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
                indices = torch.randperm(image_patches1.size(1))[:round(image_patches1.size(1)*rs_rate_stnorm)] #sample
                image_patches1 = image_patches1[:, indices, :]

            if len(data[3]) != 0:
                image_patches2 = data[3].to(device, dtype=torch.float32)
                indices = torch.randperm(image_patches2.size(1))[:round(image_patches2.size(1)*rs_rate_nostnorm)] #sample
                image_patches2 = image_patches2[:, indices, :]
            
            if len(data[0]) != 0 and len(data[3]) != 0:
                image_patches = torch.concat([image_patches1,image_patches2], dim = 1)
            elif len(data[0]) == 0:
                image_patches = image_patches2
            elif len(data[3]) == 0:
                image_patches = image_patches1
                

            label_lists = data[1][0]
            
            if criterion_da is not None:
                sub_preds_list, slide_preds_list, attn_list, _, _ = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
            else:
                sub_preds_list, slide_preds_list, attn_list, bag_feat_list = net(image_patches)
            
            #Compute loss for each task, then sum
            pred_list = []
            pred_prob_list = []
            for k in range(conf.n_task):
                sub_preds = sub_preds_list[k]
                slide_preds = slide_preds_list[k]
                attn = attn_list[k]
                labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
                pred_prob = torch.sigmoid(slide_preds)
                pred = pred_prob[0][0].round()
                pred_list.append(pred)
                pred_prob_list.append(pred_prob)
        
            y_pred.append(pred_list)
            y_true.append(label_lists)
            y_pred_prob.append(pred_prob_list)
    
        #Get prediction for each task
        y_predprob_task = []
        y_pred_tasks = []
        y_true_tasks = []
        for k in range(conf.n_task):
            y_pred_tasks.append([p[k] for p in y_pred])
            y_predprob_task.append([p[k].item() for p in y_pred_prob])
            y_true_tasks.append([t[:,k].to(device, dtype = torch.int64).item() for t in y_true])
    
    return y_pred_tasks, y_predprob_task, y_true_tasks


def output_pred_perf(model, data_loader, ids, selected_label, conf, cohort_name, out_path_pred, out_path_pref, criterion_da, device, calibration = False):
    
    y_pred_tasks,  y_predprob_task, y_true_task = predict_v2(model, data_loader, device, conf, criterion_da)
    
    if calibration == False:
        ytrue_val = None
        ypredprob_val = None
        
    all_pred_df, all_perf_df = get_performance_alltask(conf.n_task, y_true_task, y_predprob_task, ids, selected_label, 
                                                       calibration = calibration, ytrue_val = ytrue_val, ypredprob_val = ypredprob_val)
    
   

    all_pred_df.to_csv(os.path.join(out_path_pred, "n_token" + str(conf.n_token) + "_" + cohort_name + "_pred_df.csv"),index = False)
    all_perf_df.to_csv(os.path.join(out_path_pref, "n_token" + str(conf.n_token) + "_" + cohort_name + "_perf.csv"),index = True)
    print(cohort_name + ":",round(all_perf_df['AUC'].mean(),2))
    
    
    
    
    #plot boxplot of pred prob by mutation class
    boxplot_predprob_by_mutationclass(all_pred_df, cohort_name, out_path_pred)

    #plot roc curve by mutation class
    for i in range(conf.n_task):
        cur_pred_df = all_pred_df.loc[all_pred_df['OUTCOME'] == selected_label[i]]
        plot_roc_curve(list(cur_pred_df['Pred_Prob']),list(cur_pred_df['Y_True']), out_path_pred, cohort_name ,selected_label[i])
        
        
    # #bootstrap perforance
    # ci_list = []
    # for i in range(conf.n_task):
    #     print(i)
    #     cur_pred_df = all_pred_df1.loc[all_pred_df1['OUTCOME'] == SELECTED_LABEL[i]]
    #     cur_ci_df = bootstrap_ci_from_df(cur_pred_df, y_true_col='Y_True', y_pred_col='Pred_Class', y_prob_col='Pred_Prob', num_bootstrap=1000, ci=95, seed=42)
    #     cur_ci_df['OUTCOME'] = SELECTED_LABEL[i]
    #     ci_list.append(cur_ci_df)
    # ci_final_df = pd.concat(ci_list)
    # print(ci_final_df)
    # ci_final_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_" + train_cohort1 + "_perf_bootstrap2.csv",index = True)
    
    
    
def output_pred_perf_with_logit(model, data_loader, ids, selected_label, conf, cohort_name, out_path_pred, out_path_pref, criterion_da, device, calibration = False):
    
    y_pred_tasks,  y_predprob_task, y_true_task, y_logit = predict_v2_logit(model, data_loader, device, conf, criterion_da)
    
    if calibration == False:
        ytrue_val = None
        ypredprob_val = None
        
    all_pred_df, all_perf_df = get_performance_alltask_withlogit(conf.n_task, y_true_task, y_predprob_task,y_logit, ids, selected_label, 
                                                       calibration = calibration, ytrue_val = ytrue_val, ypredprob_val = ypredprob_val)
    
   

    all_pred_df.to_csv(os.path.join(out_path_pred, "n_token" + str(conf.n_token) + "_" + cohort_name + "_pred_df.csv"),index = False)
    all_perf_df.to_csv(os.path.join(out_path_pref, "n_token" + str(conf.n_token) + "_" + cohort_name + "_perf.csv"),index = True)
    print(cohort_name + ":",round(all_perf_df['AUC'].mean(),2))
    
    
    
    
    #plot boxplot of pred prob by mutation class
    boxplot_predprob_by_mutationclass(all_pred_df, cohort_name, out_path_pred)

    #plot roc curve by mutation class
    for i in range(conf.n_task):
        cur_pred_df = all_pred_df.loc[all_pred_df['OUTCOME'] == selected_label[i]]
        plot_roc_curve(list(cur_pred_df['Pred_Prob']),list(cur_pred_df['Y_True']), out_path_pred, cohort_name ,selected_label[i])
        

def predict_v2_logit_singletask(net, criterion, data_loader, device, conf):    
    # Set the network to evaluation mode
    net.eval()

    y_pred_prob = []
    y_pred_logit = []
    y_true = []
    total_loss = 0
    for data_it, data in enumerate(data_loader):
        image_patches = data[0].to(device, dtype=torch.float32)
        labels = data[1][0].to(device)
        tf = data[2].to(device, dtype=torch.float32)
    
    
        #Run model
        branch_preds, slide_preds, branch_att_raw, bag_att , bag_feat = net(image_patches)
        
                
        #Get predictions
        pred_prob = torch.softmax(slide_preds, dim=1)[0,1] #torch.sigmoid(slide_preds)
        y_pred_prob.append(pred_prob.unsqueeze(0))
        y_pred_logit.append(slide_preds[0,1].unsqueeze(0))
        y_true.append(labels)
        
        
        #Compute loss
        loss, diff_loss, loss0,  loss1 = compute_loss_singletask(branch_preds, slide_preds, labels, branch_att_raw, criterion, device, conf)
        total_loss += loss
        avg_loss = total_loss/(data_it+1)
        

    #Concatenate all pred and labels
    all_pred_prob  = torch.cat(y_pred_prob, dim=0)
    all_pred_logit  = torch.cat(y_pred_logit, dim=0)
    all_labels = torch.cat(y_true, dim=0)    
    #all_pred_class = (all_pred_prob > thres).int()     
    
    
    return  all_pred_prob, all_pred_logit, all_labels

def output_pred_perf_with_logit_singletask(model, data_loader, ids, selected_label, conf, cohort_name, criterion, out_path_pred, out_path_pref, criterion_da, device, calibration = False):
   
    # data_loader = val_loader
    # ids = val_ids

    y_predprob, y_logit, y_true = predict_v2_logit_singletask(model, criterion, data_loader, device, conf)
    

    all_pred_df, all_perf_df = get_performance_withlogit(y_predprob.squeeze().detach().cpu(), 
                                                         y_true.squeeze().detach().cpu(),
                                                         y_logit.squeeze().detach().cpu(), 
                                                         ids, 
                                                         selected_label[0], 
                                                         THRES = np.quantile(y_predprob.squeeze().detach().cpu(),0.5))

    all_pred_df.to_csv(os.path.join(out_path_pred, "n_token" + str(conf.n_token) + "_" + cohort_name + "_pred_df.csv"),index = False)
    all_perf_df.to_csv(os.path.join(out_path_pref, "n_token" + str(conf.n_token) + "_" + cohort_name + "_perf.csv"),index = True)
    print(cohort_name + ":",round(all_perf_df['AUC'].mean(),2))
    
    #plot boxplot of pred prob by mutation class
    boxplot_predprob_by_mutationclass(all_pred_df, cohort_name, out_path_pred)

    #plot roc curve by mutation class
    for i in range(conf.n_task):
        cur_pred_df = all_pred_df.loc[all_pred_df['OUTCOME'] == selected_label[i]]
        plot_roc_curve(list(cur_pred_df['Pred_Prob']),list(cur_pred_df['Y_True']), out_path_pred, cohort_name ,selected_label[i])
        
                
import torch.nn.functional as F
def predict_with_logit_adjustment(model, x, class_priors, tau=1.0):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        adjustment = tau * torch.log(class_priors)
        adjusted_logits = logits - adjustment  # Eq (9) in the paper
        adjusted_probs = F.softmax(adjusted_logits, dim=1)
    return logits, probs, adjusted_logits, adjusted_probs
        
def eval_decouple(net, indata, ids, y_true, outcome, use_adjusted, class_priors, tau = 1.0, THRES = 0.5):


    
    logits, probs, adjusted_logits, adjusted_probs = predict_with_logit_adjustment(net, indata.x, class_priors, tau =tau)
    
    
    #Prediction df
    pred_df = pd.DataFrame({"SAMPLE_IDs":  ids, 
                            "Y_True": y_true, 
                            "logit": logits[:,1],
                            "Pred_Prob": probs[:,1],
                            "logit_adj" :  adjusted_logits[:,1],
                            "Pred_Prob_adj" :  adjusted_probs[:,1],
                            "OUTCOME": outcome})
    
    pred_df['Pred_Class'] = 0
    
    if use_adjusted == False:
        pred_df.loc[pred_df['Pred_Prob'] > THRES,'Pred_Class'] = 1
    else:
        pred_df.loc[pred_df['Pred_Prob_adj'] > THRES,'Pred_Class'] = 1
    
    
    perf_df = compute_performance_each_label([outcome], pred_df, "SAMPLE_LEVEL")

    return pred_df, perf_df
        


def generate_attention_csv(model,
                           test_data,
                           test_sp_ids,
                           mask_list,
                           tile_info_data,
                           outdir,
                           device="cuda"):
    model.eval()
    for (i, parts) in enumerate(zip(test_data, test_sp_ids, mask_list)):
        x, y, tf, sl = parts[0]
        x = x.unsqueeze(0).to(device)      # add batch dim
        y = y.long().view(-1).to(device)
        sp_id = parts[1]
        mask = parts[2]
        
        #Run model     
        results, log_dict = model(x,return_attention=True,
                           return_slide_feats=True)
        attention = log_dict['attention']
        s_feature = log_dict['slide_feats']
        
        #Get sample data
        sample_entry = [d for d in tile_info_data if d["sample_id"] == sp_id][0]
        tile_info = sample_entry['tile_info']            
        tile_info['tumor_fraction'] = sample_entry['tumor_fraction']
        
        #Add att
        tile_info['att'] = pd.NA
        tile_info.loc[mask.numpy(), 'att'] = attention.squeeze().detach().cpu().numpy()
        
        att_df = tile_info[['pred_map_location',"att","tumor_fraction"]]
        att_df.to_csv(os.path.join(outdir, sp_id + "_att.csv"))


def get_final_perf(pred_data, cohort, logit_adj_infer = True):

    if logit_adj_infer:
        y_true, prob, pred = pred_data['True_y'], pred_data['adj_prob_1'], pred_data['Pred_Class_adj']
    else:
        y_true, prob, pred = pred_data['True_y'], pred_data['prob_1'], pred_data['Pred_Class']
    
    perf_tb = compute_performance(y_true, prob, pred, cohort)
    perf_tb['best_thresh'] = pd.NA
    
    return perf_tb