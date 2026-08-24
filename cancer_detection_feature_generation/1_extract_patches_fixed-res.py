#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import pandas as pd
import warnings
import glob
sys.path.insert(0, '../Utils/')
from Utils import slide_ROIS
from misc_utils import create_dir_if_not_exists, get_ids
from Utils import generating_tiles, generating_tiles_tma
warnings.filterwarnings("ignore")
from pathlib import Path

#RUN
#source ~/.bashrc
#conda activate paimg9
#python3 -u 1_extract_patches_fixed-res.py  --cohort_name CCOLA --pixel_overlap 0
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Extract patches")
parser.add_argument('--mag_extract', default='20', type=int, help='specify magnification, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--mag_target_tiss', default='1.25', type=float, help='magnification for tissue detection: e.g., 1.25x')
parser.add_argument('--cohort_name', default='naresh_data', type=str, help='Cohort name: CCF,OPX, TCGA_PRAD, Neptune, TAN_TMA_Cores,Pluvicto_TMA_Cores, Pluvicto_Pretreatment_bx, PrECOG, "CCOLA", NIH,UW_MetBx_all, Dynamo_MDAnderson_HE_84cases')
parser.add_argument('--out_dir', default='1_tile_pulling', type=str, help='output directory')

parser.add_argument('--select_idx_start', default = 0,type=int)
parser.add_argument('--select_idx_end', default = 1, type=int)
if __name__ == '__main__':
    
    args = parser.parse_args()

    ############################################################################################################
    #USER INPUT 
    ############################################################################################################
    limit_bounds = True     # this is weird, dont change it
    folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)


    #DIR
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    wsi_location = os.path.join(proj_dir, "data", args.cohort_name)
    out_location = os.path.join(proj_dir,'intermediate_data', args.out_dir, args.cohort_name, folder_name)  #1_feature_extraction, cancer_prediction_results110224

    #Create output dir
    create_dir_if_not_exists(out_location)
    
    ############################################################################################################
    #All available IDs
    #Ccola: 234
    #OPX: 360
    #TCGA: 449
    #Neptune: 350
    #TAN_TMA: 677
    #pluvicto: 606
    #PrECOG: 46
    #Pluvicto_Pretreatment_bx: 27
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
    elif args.cohort_name == 'HistoSpatial/MH11-prostate/':
        all_files = sorted(
            glob.glob(os.path.join(wsi_location, "**", "*.ome.tif"), recursive=True) +
            glob.glob(os.path.join(wsi_location, "**", "*.ome.tiff"), recursive=True)
        )
    
        if len(all_files) == 0:
            raise FileNotFoundError(f"No OME-TIFF files found under {wsi_location}")
    
        file_ids = [
            Path(p).name.replace(".ome.tiff", "").replace(".ome.tif", "")
            for p in all_files
        ]
    
        id_df = pd.DataFrame({"file_path": all_files, "fileid": file_ids})
        print(args.cohort_name, ": N of IDs = ", id_df.shape[0])
    elif args.cohort_name == 'UWTAN/WSI_HE/all_slides/':
        all_ids = get_ids(wsi_location)
        all_ids = [x for x in all_ids if x not in ['frozen']]
        print(args.cohort_name, ": N of IDs = " ,len(all_ids))

    else:
        all_ids = get_ids(wsi_location)
        print(args.cohort_name, ": N of IDs = " ,len(all_ids))

    ############################################################################################################
    #RUN
    ############################################################################################################
    if args.cohort_name in ['WCDT', 'UW_MetBx_all', 'NIH', 'Dynamo_MDAnderson_HE_84cases','CCF','CCOLA','HistoSpatial/MH11-prostate/']:

        if args.cohort_name == 'CCOLA':
            rad_tissue = 2
        else:
            rad_tissue = 5
        for idx, row in id_df.iterrows():
            cur_id = row['fileid']
            cur_file = row['file_path']
            
            #create out path
            save_location = out_location + "/" + cur_id + "/"
            create_dir_if_not_exists(save_location)
        
            #check if processed:
            imgout = glob.glob(save_location + "*.png")
            if len(imgout) > 0:
                 print(cur_id + ': already processed')
            elif len(imgout) == 0:
                
                slides_name = cur_id
                #Generating tiles 
                mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles(cur_id, cur_file, args.save_image_size, args.pixel_overlap, limit_bounds, args.mag_target_tiss, rad_tissue, args.mag_extract)

                tile_info_df.to_csv(os.path.join(save_location, slides_name + "_tiles.csv"), index = False)
                lvl_img.save(os.path.join(save_location,  slides_name + '_low-res.png'))
                lvl_mask.save(os.path.join(save_location, slides_name + '_tissue.png'))
                slide_ROIS(polygons=tissue, mpp=float(mpp),savename=os.path.join(save_location, slides_name + '_tissue.json'),labels='tissue', ref=[0, 0], roi_color=-16770432)
        
    else:
        ############################################################################################################
        #ID to Exclude
        ############################################################################################################
        #Get IDs that are in FT train or already processed to exclude 
        fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/0_cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
        ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id'])
        toexclude_ids = ft_train_ids + ['cca3af0c-3e0e-4cfb-bb07-459c979a0bd5'] #The latter one is TCGA issue file
        
        ############################################################################################################
        #Select ID
        ############################################################################################################
        #Exclude ids in ft_train
        selected_ids = [x for x in all_ids if x not in toexclude_ids]
        selected_ids.sort()
        
        ############################################################################################################
        #Start 
        ############################################################################################################
        ct = 0
        for cur_id in selected_ids[args.select_idx_start:args.select_idx_end]:
            
            if (ct % 50 == 0): print(ct)
            ct += 1
            
            #create out path
            save_location = out_location + "/" + cur_id + "/"
            create_dir_if_not_exists(save_location)
        
            #check if processed:
            imgout = glob.glob(save_location + "*.png")
            if len(imgout) > 0:
                 print(cur_id + ': already processed')
            elif len(imgout) == 0:
                slides_name = cur_id
                if 'OPX' in args.cohort_name:
                    _file =  os.path.join(wsi_location, slides_name + ".tif") 
                    rad_tissue = 5
                elif 'CCOLA' in args.cohort_name:
                    _file = os.path.join(wsi_location, slides_name + ".svs") 
                    rad_tissue = 2
                elif 'TAN_TMA_Cores' in args.cohort_name:
                    _file = os.path.join(wsi_location, slides_name + ".tif") 
                    rad_tissue = 2
                elif 'Pluvicto_TMA_Cores' in args.cohort_name:
                    _file = os.path.join(wsi_location, slides_name + ".tif") 
                    rad_tissue = 2
                elif 'PrECOG' in args.cohort_name:
                    _file = os.path.join(wsi_location, slides_name + ".svs") 
                    rad_tissue = 2
                elif 'Neptune' in args.cohort_name:
                    _file = os.path.join(wsi_location, slides_name + ".tif") 
                    if cur_id == 'NEP-081PS2-1_HE_MH_03282024' or cur_id == 'NEP-123PS1-1_HE_MH06032024':
                        rad_tissue = 2
                    else:
                        rad_tissue = 5
                elif 'TCGA' in args.cohort_name:
                    slides_name = [f for f in os.listdir(os.path.join(wsi_location, cur_id)) if '.svs' in f][0].replace('.svs','')
                    _file = os.path.join(wsi_location, cur_id, slides_name + '.svs') 
                    rad_tissue = 2
                elif args.cohort_name == 'Pluvicto_Pretreatment_bx':
                    _file =  os.path.join(wsi_location, slides_name + ".tif") 
                    rad_tissue = 5
                elif args.cohort_name == 'Karmanos_Pluvicto_PreTreatment':
                    _file =  os.path.join(wsi_location, slides_name + ".tif") 
                    rad_tissue = 5
                elif args.cohort_name == 'UWTAN/WSI_HE/all_slides/':
                    _file =  os.path.join(wsi_location, slides_name + ".tif") 
                    if cur_id == '15-003FF5_H&E_MRP_09-15-21': 
                        rad_tissue = 2
                    elif cur_id == '17-081G5_H&E_MPR_08-28-2020':
                        rad_tissue = 1
                    else:
                        rad_tissue = 5
                elif 'naresh_data' in args.cohort_name:
                    _file =  os.path.join(wsi_location, slides_name + ".ndpi") 
                    rad_tissue = 5
                                
                #Generating tiles 
                if args.cohort_name in ["OPX", "CCOLA", "Neptune", "PrECOG", "Pluvicto_Pretreatment_bx", "Karmanos_Pluvicto_PreTreatment", "UWTAN/WSI_HE/all_slides/","naresh_data"]:                
                    mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles(cur_id, _file, args.save_image_size, args.pixel_overlap, limit_bounds, args.mag_target_tiss, rad_tissue, args.mag_extract)
                elif args.cohort_name in ["TAN_TMA_Cores", "Pluvicto_TMA_Cores"]:
                    mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles_tma(cur_id, _file, args.save_image_size, args.pixel_overlap, rad_tissue)
                elif  args.cohort_name in ['TCGA_PRAD']:
                    mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles(cur_id, _file, args.save_image_size, args.pixel_overlap, limit_bounds, args.mag_target_tiss, rad_tissue, args.mag_extract)
                    tile_info_df['SAMPLE_ID'] = slides_name
                    
                tile_info_df.to_csv(os.path.join(save_location, slides_name + "_tiles.csv"), index = False)
                lvl_img.save(os.path.join(save_location,  slides_name + '_low-res.png'))
                lvl_mask.save(os.path.join(save_location, slides_name + '_tissue.png'))
                slide_ROIS(polygons=tissue, mpp=float(mpp),savename=os.path.join(save_location, slides_name + '_tissue.json'),labels='tissue', ref=[0, 0], roi_color=-16770432)