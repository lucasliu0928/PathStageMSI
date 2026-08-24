#!/usr/bin/env python
# coding: utf-8


import sys
import os
import numpy as np
import openslide
import torch
import pandas as pd
import warnings
import time
import PIL
import argparse
from skimage import io
import glob
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists, str2bool, get_ids
from FeatureExtractor import PretrainedModelLoader, TileEmbeddingExtractor
warnings.filterwarnings("ignore")

#source ~/.bashrc
#conda activate paimg9
#Run: python3 -u 4_get_feature.py --select_idx_start 0 --select_idx_end 29 --cuda_device 'cuda:1' --pixel_overlap 0 --save_image_size 250 --cohort_name WCDT --fine_tuned_model False --feature_extraction_method uni2


############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--cohort_name', default='CCOLA', type=str, help='Cohort name: CCF,OPX, TCGA_PRAD, Neptune, TAN_TMA_Cores,Pluvicto_TMA_Cores, Pluvicto_Pretreatment_bx, PrECOG, "CCOLA", NIH,UW_MetBx_all, Dynamo_MDAnderson_HE_84cases')
parser.add_argument('--feature_extraction_method', default='virchow2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2, hoptimus0')
parser.add_argument('--fine_tuned_model', type=str2bool, default=True, help='whether or not to use fine-tuned cancer detection model')
parser.add_argument('--stain_norm', default='norm', type=str, help='norm or no_norm')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--out_folder', default= '4_tile_feature', type=str, help='out folder name')
parser.add_argument('--select_idx_start', type=int, default = 0)
parser.add_argument('--select_idx_end', type=int, default = 1)

if __name__ == '__main__':
    
    args = parser.parse_args()
    ############################################################################################################
    #USER INPUT 
    ############################################################################################################ 
    folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)

    
    ############################################################################################################
    #DIR
    ############################################################################################################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    wsi_location = os.path.join(proj_dir, "data", args.cohort_name.replace("z_nostnorm_", ""))
    info_path  = os.path.join(proj_dir,'intermediate_data','2_cancer_detection', args.cohort_name, folder_name)
    model_path = os.path.join(proj_dir,'models','feature_extraction_models', args.feature_extraction_method)
    out_location = os.path.join(proj_dir,'intermediate_data', args.out_folder, args.cohort_name, folder_name)

    create_dir_if_not_exists(out_location)

    ############################################################################################################
    #Select GPU
    ############################################################################################################
    device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')
    print(device)

    ############################################################################################################
    #All available IDs
    #Ccola: 234
    #OPX: 360
    #TCGA: 449
    #Neptune: 350
    #TAN_TMA: 677
    #pluvicto: 606
    #PrECOG: 46
    #Pluvicto_Pretreatment_bx: 29
    #Karmanos_Pluvicto_PreTreatment: 15
    #WCDT: 78
    #UW_MetBx_all: 261
    #NIH: 117
    #Dynamo_MDAnderson_HE_84cases: 65
    #CCF : 113 
    #CCOLA: 75
    #UWTAN/WSI_HE/all_slides/: 638
    ############################################################################################################    
    # if args.cohort_name == "CCola/all_slides/":
    #     all_ids = get_ids(wsi_location, include="(2017-0133)")  # 234
    if args.cohort_name in ['WCDT', 'NIH','Dynamo_MDAnderson_HE_84cases','CCF','CCOLA']:
        #Load ID of intrest list
        metbx_df = pd.read_excel(os.path.join(proj_dir, "data", "MutationCalls_or_Clinicaldata", "met_bx_outcomes_by_folder.xlsx"))
        
        if args.cohort_name == 'Dynamo_MDAnderson_HE_84cases':
            id_of_interest = metbx_df.loc[metbx_df['cohort'] == "MDA", 'fileid'].tolist()
            id_of_interest = [x.rstrip('_') for x in id_of_interest]
        elif args.cohort_name == 'CCOLA':
            id_of_interest = metbx_df.loc[metbx_df['cohort'] == args.cohort_name, 'fileid'].tolist()
            id_of_interest = [x.replace("__", " ") for x in id_of_interest]
            wsi_location = os.path.join(wsi_location, "all_slides")
        else:
            id_of_interest = metbx_df.loc[metbx_df['cohort'] == args.cohort_name, 'fileid'].tolist()
        
        #load actual files
        svs_files = glob.glob(os.path.join(wsi_location, "**", "*.svs"), recursive=True)
        tif_files = glob.glob(os.path.join(wsi_location, "**", "*.tif"), recursive=True)
        tif_files += glob.glob(os.path.join(wsi_location, "**", "*.tiff"), recursive=True)
        ndpi_files = glob.glob(os.path.join(wsi_location, "**", "*.ndpi"), recursive=True)
        all_files = svs_files + tif_files + ndpi_files
        file_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_files]
        id_df = pd.DataFrame({"file_path": all_files,"fileid": file_ids})
        #intersect
        id_df = id_df.loc[id_df['fileid'].isin(id_of_interest)].reset_index(drop=True)
        print(args.cohort_name, ": N of IDs = " ,id_df.shape[0])
        
    elif args.cohort_name == 'UW_MetBx_all':
        
        ####
        #NOTE: SU_18_20104_A1-1_HE_MH091021 not in the our folder, 
        #      SU_09_32989_A1_HE_MH090821 not in our folder, but it should be the same asSU-09-32989_A1_HE_MH090821(2)
        #      (duplicates in the list) SU-16_15272_A1-2_HE_40X_MH040423 is the same as SU-16-15272_A1-2_HE_40X_MH040423 
        ###
        #Load ID of intrest list
        metbx_df = pd.read_excel(os.path.join(proj_dir, "data", "MutationCalls_or_Clinicaldata", "met_bx_outcomes_by_folder.xlsx"))
        #Type correction
        cond = metbx_df['fileid'] == 'SU_09_32989_A1_HE_MH090821(2)'
        metbx_df.loc[cond, 'fileid'] = 'SU-09-32989_A1_HE_MH090821(2)'
        cond = metbx_df['fileid'] == 'SU_16_01651_A1-1_HE_MH090821 (2)'
        metbx_df.loc[cond, 'fileid'] = 'SU-16-01651_A1-1_HE_MH090821(2)'
        
        id_of_interest = metbx_df.loc[metbx_df['cohort'] == "UWBX", 'fileid'].tolist()
    
        #load actual files
        svs_files = glob.glob(os.path.join(wsi_location, "**", "*.svs"), recursive=True)
        tif_files = glob.glob(os.path.join(wsi_location, "**", "*.tif"), recursive=True)
        ndpi_files = glob.glob(os.path.join(wsi_location, "**", "*.ndpi"), recursive=True)
        all_files = svs_files + tif_files + ndpi_files
        file_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_files]
        id_df = pd.DataFrame({"file_path": all_files,"fileid": file_ids})
        
        #intersect
        id_df = id_df.loc[id_df['fileid'].isin(id_of_interest)].reset_index(drop=True)
        print(args.cohort_name, ": N of IDs = " ,id_df.shape[0])

    elif args.cohort_name == 'UWTAN/WSI_HE/all_slides/':
        all_ids = get_ids(wsi_location)
        all_ids = [x for x in all_ids if x != 'frozen']
        print(args.cohort_name, ": N of IDs = " ,len(all_ids))
        
    else:
        all_ids = get_ids(wsi_location)
        print(args.cohort_name, ": N of IDs = " ,len(all_ids))
    
    
    ############################################################################################################
    #Load normalization norm target image
    ############################################################################################################
    if args.stain_norm == "norm":
        tile_norm_img_path = os.path.join(proj_dir,'intermediate_data/6A_tile_for_stain_norm/')
        norm_target_img = io.imread(os.path.join(tile_norm_img_path, 'SU21-19308_A1-2_HE_40X_MH110821_40_16500-20500_500-500.png'))
    elif args.stain_norm == "no_norm":
        norm_target_img = None
        
        
    ############################################################################################################
    # Load Pretrained representation model
    ############################################################################################################
    modelloader = PretrainedModelLoader(args.feature_extraction_method, model_path, device=device)
    model = modelloader.model
    
            
    ############################################################################################################
    #RUN
    ############################################################################################################
    if args.cohort_name in ['WCDT', 'UW_MetBx_all', 'NIH', 'Dynamo_MDAnderson_HE_84cases','CCF', 'CCOLA']:
        
        start_time_all = time.time()
        selected_range = id_df.iloc[args.select_idx_start:args.select_idx_end].shape[0]
        total = selected_range
        ct = 0
        for idx, row in id_df.iloc[args.select_idx_start:args.select_idx_end].iterrows():
            cur_id = row['fileid']
            cur_file = row['file_path']
            
            
            save_location = os.path.join(out_location, cur_id , 'features')
            create_dir_if_not_exists(save_location)
            save_name = os.path.join(save_location, 'features_alltiles_' + args.feature_extraction_method + '.h5')
            
            
            if os.path.exists(save_name) == False: #check if processed
                #Get slide file name
                cohort = args.cohort_name.replace("z_nostnorm_", "")
                slides_name = cur_id            
                    
                #Get tile info
                if args.fine_tuned_model == True:
                    cur_tile_info_df = pd.read_csv(os.path.join(info_path, cur_id ,'ft_model', slides_name + "_TILE_TUMOR_PERC.csv"))
                else:
                    cur_tile_info_df = pd.read_csv(os.path.join(info_path, cur_id ,'prior_model', slides_name + "_TILE_TUMOR_PERC.csv"))
    
                print('NOT Processed:',cur_id, "N Tiles:", str(cur_tile_info_df.shape[0]))
                
                #Load slides, and Construct embedding extractor    
                oslide = openslide.OpenSlide(cur_file) 
                embed_extractor = TileEmbeddingExtractor(cur_tile_info_df, oslide, args.feature_extraction_method, model, device, 
                                                         stain_norm_target_img = norm_target_img,
                                                         image_type = 'WSI')             
                #Get feature
                start_time = time.time()
                feature_list = [embed_extractor[i][1] for i in range(cur_tile_info_df.shape[0])]
                print(f"--- {(time.time() - start_time) / 60:.2f} minutes ---")
                feature_df = np.concatenate(feature_list)
                feature_df = pd.DataFrame(feature_df)    
                feature_df.to_hdf(save_name, key='feature', mode='w')
                cur_tile_info_df.to_hdf(save_name, key='tile_info', mode='a')
        
                ct += 1
            else:
                print('Already Processed:',cur_id)
            
            
        end_time_all = time.time()
        duration_all = end_time_all - start_time_all
        print(f"\n All {ct} IDs processed out of {total} total.")
        print(f"Started at:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time_all))}")
        print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_all))}")
        print(f"Total duration: {duration_all/60:.2f} minutes\n")
    
    
    else:
        ############################################################################################################
        #ID to Exclude
        ############################################################################################################
        #Get IDs that are in FT train or already processed to exclude 
        fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/0_cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
        ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id']) #24, 7 from OPX, 17 from ccola
        toexclude_ids = ft_train_ids + ['cca3af0c-3e0e-4cfb-bb07-459c979a0bd5'] #The latter one is TCGA issue file
        toexclude_ids = toexclude_ids + ['61887edb-c320-42bb-a0b5-fe84c17dbd21'] #This is an IHC
        
        ############################################################################################################
        #Select ID
        ############################################################################################################
        #Exclude ids in ft_train
        selected_ids = [x for x in all_ids if x not in toexclude_ids]
        selected_ids.sort()
        print("n of selected IDs:",len(selected_ids))
    

        ############################################################################################################
        #For each patient tile, get representation
        ############################################################################################################
        start_time_all = time.time()
        selected_range = selected_ids[args.select_idx_start:args.select_idx_end]
        total = len(selected_range)
        ct = 0 
        for cur_id in selected_range:
            if ct % 10 == 0: print(ct)
    
            save_location = os.path.join(out_location, cur_id , 'features')
            create_dir_if_not_exists(save_location)
            save_name = os.path.join(save_location, 'features_alltiles_' + args.feature_extraction_method + '.h5')
            
            
            if os.path.exists(save_name) == False: #check if processed
                #Get slide file name
                cohort = args.cohort_name.replace("z_nostnorm_", "")
                slides_name = cur_id            
                if cohort == "TCGA_PRAD":
                    slides_name = [f for f in os.listdir(os.path.join(wsi_location, cur_id)) if '.svs' in f][0].replace('.svs','')                
                    _file = os.path.join(wsi_location, cur_id, slides_name + ".svs")
                elif cohort in ["OPX", "TAN_TMA_Cores", "Neptune", "Pluvicto_TMA_Cores", "Pluvicto_Pretreatment_bx","Karmanos_Pluvicto_PreTreatment","UWTAN/WSI_HE/all_slides/"]:
                    _file = os.path.join(wsi_location, slides_name + ".tif")
                elif cohort in ["PrECOG"]:
                    _file = os.path.join(wsi_location, slides_name + ".svs")
                else:
                    raise ValueError(f"Unknown cohort: {cohort}")
                    
                #Get tile info
                if args.fine_tuned_model == True:
                    cur_tile_info_df = pd.read_csv(os.path.join(info_path, cur_id ,'ft_model', slides_name + "_TILE_TUMOR_PERC.csv"))
                else:
                    cur_tile_info_df = pd.read_csv(os.path.join(info_path, cur_id ,'prior_model', slides_name + "_TILE_TUMOR_PERC.csv"))
    
                print('NOT Processed:',cur_id, "N Tiles:", str(cur_tile_info_df.shape[0]))
                
                #Load slides, and Construct embedding extractor    
                if cohort in["OPX", "TCGA_PRAD", "Neptune", "PrECOG", "Pluvicto_Pretreatment_bx","Karmanos_Pluvicto_PreTreatment", "UWTAN/WSI_HE/all_slides/"]:
                    oslide = openslide.OpenSlide(_file) 
                    embed_extractor = TileEmbeddingExtractor(cur_tile_info_df, oslide, args.feature_extraction_method, model, device, 
                                                             stain_norm_target_img = norm_target_img,
                                                             image_type = 'WSI')             
    
                elif cohort in ["TAN_TMA_Cores", "Pluvicto_TMA_Cores"]:      
                    tma = PIL.Image.open(_file)
                    embed_extractor = TileEmbeddingExtractor(cur_tile_info_df, tma, args.feature_extraction_method, model, device,
                                                             stain_norm_target_img = norm_target_img,
                                                             image_type = 'TMA')
        
                #Get feature
                start_time = time.time()
                feature_list = [embed_extractor[i][1] for i in range(cur_tile_info_df.shape[0])]
                print(f"--- {(time.time() - start_time) / 60:.2f} minutes ---")
                feature_df = np.concatenate(feature_list)
                feature_df = pd.DataFrame(feature_df)    
                feature_df.to_hdf(save_name, key='feature', mode='w')
                cur_tile_info_df.to_hdf(save_name, key='tile_info', mode='a')
        
                ct += 1
            else:
                print('Already Processed:',cur_id)
            
            
        end_time_all = time.time()
        duration_all = end_time_all - start_time_all
        print(f"\n All {ct} IDs processed out of {total} total.")
        print(f"Started at:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time_all))}")
        print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_all))}")
        print(f"Total duration: {duration_all/60:.2f} minutes\n")
    
    
    
