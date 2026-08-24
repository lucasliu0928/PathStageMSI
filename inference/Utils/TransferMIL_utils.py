#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:01:42 2025

@author: jliu6
"""

from src.builder import create_model
import torch
import pandas as pd
import math
import copy


from Eval import compute_performance

#Model
from TransMIL import TransMIL



def train(train_loader, 
          model, 
          device,
          criterion, 
          optimizer, 
          model_name = 'Transfer_MIL' ,
          l2_coef = 1e-4, 
          logit_adjustments = None,
          logit_adj_train = True):
    """ Run one train epoch """
    
    total_loss = 0
    model.train()
    for data_it, data in enumerate(train_loader):
        
        #Get data
        x = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
        y = data[1][0].long().view(-1).to(device)
        tf = data[2].to(device, dtype=torch.float32)            #(1, N_Patches]
                        
        #Run model     
        if model_name == 'TransMIL':
            results = model(data = x)
        elif model_name == "Transfer_MIL":
            results, _ = model(x)
        
        #Get output   
        output = results['logits']
        
        #Logit adjustment 
        if logit_adj_train == True:
            output = output + logit_adjustments.to(device)
        
        #Get loss
        loss = criterion(output, y)
        
        #L2 reg
        loss_r = 0
        for parameter in model.parameters():
            loss_r += torch.sum(parameter ** 2)
        
        loss = loss +  l2_coef * loss_r

        total_loss += loss    #Store total loss 
        
        optimizer.zero_grad() #zero grad, avoid grad acuum
        loss.backward()       #compute new gradients
        optimizer.step()      #update paraters
        
    
    avg_loss = total_loss/(len(train_loader)) 
    
    return avg_loss



def validate(test_data, sample_ids, model, device, criterion, logit_adjustments, 
             model_name = 'Transfer_MIL', logit_adj_infer = True, logit_adj_train = False,
             l2_coef = 0, pred_thres = 0.5, dict_flag = False):
       
    model.eval()
    with torch.no_grad():
        total_loss = 0
        logits_list = []
        labels_list = []
        logits_list_adj = []
        for i in range(len(test_data)):
            if dict_flag:
                x, y, tf, sl, *_ = test_data[i].values()        
            else:
                x, y, tf, sl, *_ = test_data[i]     
            x = x.unsqueeze(0).to(device)      # add batch dim
            y = y.long().view(-1).to(device)
                            
            #Run model     
            if model_name == 'TransMIL':
                results = model(data = x)
            elif model_name == "Transfer_MIL":
                results, _ = model(x)
            
            #Get output   
            output = results['logits']
            logits_list.append(output.squeeze(0).detach().cpu()) #Get logit before posthoc adj
            loss = criterion(output, y) #get loss before post-hoc logit adust)
            
            #get adjust output post hoc
            if logit_adj_infer == True:
                adj_output = output - logit_adjustments.to(device) #get logit after posthoc adj
                logits_list_adj.append(adj_output.squeeze(0).detach().cpu()) #Get logit before posthoc adj
            else:
                logits_list_adj = None
                        
            #if logit adjustment used for training
            if logit_adj_train == True:
                loss = criterion(output + logit_adjustments.to(device), y)
                
            #L2 reg
            loss_r = 0
            for parameter in model.parameters():
                loss_r += torch.sum(parameter ** 2)
            loss = loss +  l2_coef * loss_r
                
            total_loss += loss #Store total loss 
            
            #get label
            labels_list.append(int(y.detach().cpu()))
        
        avg_loss = total_loss/(len(test_data)) 
        
    
    df = get_predict_df(logits_list, labels_list, sample_ids, logits_list_adj, thres = pred_thres)
    
    return avg_loss, df


def get_predict_df(logits_list, labels_list, sample_ids, logits_list_adj = None, thres = 0.5):
    
    
    #Get logits and prob df
    logits = torch.stack(logits_list, dim=0)              # (N, C)
    probs  = torch.softmax(logits, dim=1)             # (N, C)
    
    num_classes = logits.size(1)
    logit_cols = [f"logit_{i}" for i in range(num_classes)]
    prob_cols  = [f"prob_{i}"  for i in range(num_classes)]
    
    df_logits = pd.DataFrame(logits.numpy(), columns=logit_cols)
    df_probs  = pd.DataFrame(probs.numpy(),  columns=prob_cols)
    df = pd.concat([df_logits, df_probs], axis=1)
    df["Pred_Class"] = (df["prob_1"] > thres).astype(int)
    
    
    #if post hoc logit adj
    if logits_list_adj is not None:
        logits_adj = torch.stack(logits_list_adj, dim=0)               # (N, C)
        probs_adj  = torch.softmax(logits_adj, dim=1)           # (N, C)
        
        num_classes = logits_adj.size(1)
        logit_cols = [f"adj_logit_{i}" for i in range(num_classes)]
        prob_cols  = [f"adj_prob_{i}"  for i in range(num_classes)]
        
        df_logits_adj = pd.DataFrame(logits_adj.numpy(), columns=logit_cols)
        df_probs_adj  = pd.DataFrame(probs_adj.numpy(),  columns=prob_cols)
        df_adj = pd.concat([df_logits_adj, df_probs_adj], axis=1)
        df_adj["Pred_Class_adj"] = (df_adj["adj_prob_1"] > thres).astype(int)
        
        #combine ther before logit adj
        df = pd.concat([df, df_adj], axis = 1)
            
    #add label and ids
    df["True_y"] = labels_list
    df['SAMPLE_ID'] = sample_ids


    if logits_list_adj is not None:
        #reorder
        df = df[['SAMPLE_ID','True_y',
                 'logit_0', 'logit_1','prob_0', 'prob_1','Pred_Class',
                 'adj_logit_0', 'adj_logit_1','adj_prob_0', 'adj_prob_1','Pred_Class_adj']]
    else:
        #reorder
        df = df[['SAMPLE_ID','True_y',
                 'logit_0', 'logit_1','prob_0', 'prob_1','Pred_Class']]
    
    return df


def run_eval(data, sp_ids, cohort, criterion, model, 
             device, logit_adj_infer, logit_adj_train, logit_adjustments, 
             l2_coef=0, pred_thres = 0.5, dict_flag = False):
    
    avg_loss, pred_df = validate(
        data, sp_ids, model, device, criterion,
        logit_adjustments,
        model_name='Transfer_MIL',
        logit_adj_infer=logit_adj_infer,
        logit_adj_train=logit_adj_train,
        l2_coef = l2_coef,
        pred_thres = pred_thres,
        dict_flag = dict_flag
    )

    if logit_adj_infer:
        y_true, prob, pred = pred_df['True_y'], pred_df['adj_prob_1'], pred_df['Pred_Class_adj']
    else:
        y_true, prob, pred = pred_df['True_y'], pred_df['prob_1'], pred_df['Pred_Class']

    perf_tb = compute_performance(y_true, prob, pred, cohort)
    print(perf_tb)
    return avg_loss, pred_df, perf_tb




class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.num_bad_epochs = 0
        self.best_state = None

    def step(self, metric, model):
        """
        Check if validation metric improved.
        Returns True if training should stop.
        """
        improved = (self.best - metric) > self.min_delta
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience

    def __repr__(self):
        return (f"EarlyStopper(patience={self.patience}, min_delta={self.min_delta}, "
                f"best={self.best}, num_bad_epochs={self.num_bad_epochs})")

    
    

def build_model(model_name: str, device, num_classes=2, n_feature=None):
    """
    Create and return a model based on the model_name.

    Args:
        model_name (str): Name of the model type ('Transfer_MIL' or 'TransMIL').
        device (torch.device): Device to move the model to.
        num_classes (int): Number of output classes.
        n_feature (int): Required only for TransMIL.

    Returns:
        model (nn.Module): The constructed model on the specified device.
    """
    if model_name == "Transfer_MIL":
        model = create_model('abmil.base.uni_v2.pc108-24k', num_classes=num_classes)
        model.to(device)

        # in_dim = model.model.classifier.in_features
        # model.model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.3),
        #     nn.Linear(in_dim, 64),
        #     nn.Linear(64, 32),
        #     nn.Linear(32, num_classes),  # last layer matches num_classes
        # )
        # model.to(device)

    elif model_name == "TransMIL":
        if n_feature is None:
            raise ValueError("n_feature must be provided for TransMIL")
        model = TransMIL(n_classes=num_classes, in_dim=n_feature)
        model.to(device)
    
    elif model_name == "ABMIL":
        model = create_model('abmil',num_classes=num_classes, in_dim=n_feature, from_pretrained=False)
        model.to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model