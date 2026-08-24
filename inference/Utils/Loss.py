#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:38:47 2025

@author: jliu6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, 
                 alpha: float = 0.25, 
                 gamma: float = 2, 
                 reduction='mean'):
        r'''
        if alpha = -1.0, gamma = 0.0, then it is = CE loss
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha #alpha is for the class = 1
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits, target):
        """FocalLoss  twologits
        pred_logits: [N, C] raw logits (e.g., [N, 2])
        target: [N] or [N,1] with values in {0,...,C-1}
        """
        
        if not (0 <= self.alpha <= 1) and self.alpha != -1.0:
            raise ValueError(f"Invalid alpha value: {self.alpha}. alpha must be in the range [0,1] or -1 for ignore.")

        if target.dtype != torch.long:
            target = target.long()             
    
        ce_loss = F.cross_entropy(pred_logits, target, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = prob of the true class
        loss =  ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha != -1.0:
            alpha_t = self.alpha*target + (1.0 - self.alpha)*(1.0 - target) #if target = 1, use alpha, otherwise use 1-alpha to weight
            loss = alpha_t*loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def compute_label_freq(data_loader):
    
    label_freq = {}
    for i, data in enumerate(data_loader):
        target = data[1][0].long().view(-1)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()

    return label_freq, label_freq_array
    

def compute_logit_adjustment(label_freq_array, tau):
    """compute the base probabilities
    
    official implemenation: https://github.com/Chumsy0725/logit-adj-pytorch/blob/main/utils.py
    Output: [1, N_C]: the logit adjustment for each class
    
    """
    adjustments = np.log(label_freq_array ** tau + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    
    return adjustments
        
        
        
####################################################################################################
#TODO: Double-check
####################################################################################################
class FocalLoss_withATT(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss_withATT, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.att_reg_flag = True
        self.att_reg_loss = nn.MSELoss()

    def forward(self, inputs, targets, tumor_fractions, attention_scores):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            F_loss =  F_loss.mean()
        elif self.reduction == 'sum':
            F_loss =  F_loss.sum()

        if self.att_reg_flag == True:
            attention_scores_mean = torch.softmax(attention_scores, dim=-1).mean(dim = 1) #Take the mean across all braches
            F_loss = F_loss + self.att_reg_loss(tumor_fractions, attention_scores)

        return F_loss


class BCE_Weighted_Reg(nn.Module):
    def __init__(self, lambda_coef, reg_type, model, reduction = "mean", att_reg_flag = False):
        super(BCE_Weighted_Reg, self).__init__()
        self.lambda_coef = lambda_coef
        self.reg_type = reg_type
        self.model = model
        self.reduction = reduction
        self.att_reg_flag = att_reg_flag 

        self.att_reg_loss = nn.MSELoss()

    def forward(self, output, target, class_weight, tumor_fractions, attention_scores):

        #Compute BCE
        loss = - (target * torch.log(output) + (1-target)*torch.log(1-output))
        
        #Weight loss for each class
        pos_idex = torch.where(target == 1)[0] #index of pos
        neg_idex = torch.where(target == 0)[0] #index of neg
        
        loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * corresponding loss
        loss[pos_idex] =  loss[pos_idex]*class_weight[1]


        if self.reduction == 'mean':
            loss = loss.mean()

        if self.att_reg_flag == True:
            loss = loss + self.att_reg_loss(tumor_fractions, attention_scores)
        
        #Regularization
        l1_regularization = 0
        l2_regularization = 0
        for param in self.model.parameters():
            l1_regularization += param.abs().sum()
            l2_regularization += param.square().sum()
        if self.reg_type == "L1":
            loss = loss + self.lambda_coef*l1_regularization    
        elif self.reg_type == "L2":
            loss = loss + self.lambda_coef*l2_regularization 
        else:
            loss = loss 

        return loss

class BCE_Weighted_Reg_focal(nn.Module):
    def __init__(self, lambda_coef, reg_type, model, gamma = 2,reduction = "mean", att_reg_flag = False):
        super(BCE_Weighted_Reg_focal, self).__init__()
        self.lambda_coef = lambda_coef
        self.reg_type = reg_type
        self.model = model
        self.reduction = reduction
        self.att_reg_flag = att_reg_flag 
        self.gamma = gamma

        self.att_reg_loss = nn.MSELoss()

    def forward(self, output, target, class_weight, tumor_fractions, attention_scores):

        #Compute Focal Loss
        #https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7
        loss = - (target * ((1 - output)**self.gamma) * torch.log(output) + (1-target)* (output**self.gamma) *torch.log(1-output))
        
        #Weight loss for each class
        pos_idex = torch.where(target == 1)[0] #index of pos
        neg_idex = torch.where(target == 0)[0] #index of neg
        
        loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * corresponding loss
        loss[pos_idex] =  loss[pos_idex]*class_weight[1]


        if self.reduction == 'mean':
            loss = loss.mean()

        if self.att_reg_flag == True:
            loss = loss + self.att_reg_loss(tumor_fractions, attention_scores)
        
        #Regularization
        l1_regularization = 0
        l2_regularization = 0
        for param in self.model.parameters():
            l1_regularization += param.abs().sum()
            l2_regularization += param.square().sum()
        if self.reg_type == "L1":
            loss = loss + self.lambda_coef*l1_regularization    
        elif self.reg_type == "L2":
            loss = loss + self.lambda_coef*l2_regularization 
        else:
            loss = loss 

        return loss
    

