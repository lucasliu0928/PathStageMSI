#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

@author: jliu6
"""

import sys
import os
import numpy as np
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import geojson
import argparse
import matplotlib.pyplot as plt
import fastai
from fastai.vision.all import *
import PIL
matplotlib.use('Agg')
import pandas as pd
import datetime
from skimage import draw, measure, morphology, filters
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
import datetime
import random
import torch
from histomicstk import preprocessing,features
from PIL import ImageCms, Image
sys.path.insert(0, '../Utils/')
from misc_utils import convert_img
warnings.filterwarnings("ignore")


def do_mask_original(img,lvl_resize, rad = 5):
    ''' create tissue mask '''
    # get he image and find tissue mask
    he = np.array(img)
    he = he[:, :, 0:3]
    heHSV = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(heHSV, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imagem = cv2.bitwise_not(thresh1)
    tissue_mask = morphology.binary_dilation(imagem, morphology.disk(radius=rad))
    tissue_mask = morphology.remove_small_objects(tissue_mask, 1000)
    tissue_mask = ndimage.binary_fill_holes(tissue_mask)

    # create polygons for faster tiling in cancer detection step
    polygons = []
    contours, hier = cv2.findContours(tissue_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cvals = contour.transpose(0, 2, 1)
        cvals = np.reshape(cvals, (cvals.shape[0], 2))
        cvals = cvals.astype('float64')
        for i in range(len(cvals)):
            cvals[i][0] = np.round(cvals[i][0]*lvl_resize,2)
            cvals[i][1] = np.round(cvals[i][1]*lvl_resize,2)
        try:
            poly = Polygon(cvals)
            if poly.length > 0: 
                polygons.append(Polygon(poly.exterior))
        except:
            pass
    tissue = unary_union(polygons)
    while not tissue.is_valid:
        print('pred_union is invalid, buffering...')
        tissue = tissue.buffer(0)

    return tissue, tissue_mask

def check_tissue(tile_starts, tile_ends, roi):
    ''' checks if tile in tissue '''
    tile_box = [tile_starts[0], tile_starts[1]], [tile_starts[0], tile_ends[1]], [tile_ends[0], tile_starts[1]], [tile_ends[0], tile_ends[1]]
    tile_box = list(tile_box)
    tile_box = MultiPoint(tile_box).convex_hull
    ov = 0  # initialize
    if tile_box.intersects(roi):
        ov_reg = tile_box.intersection(roi)
        ov += ov_reg.area / tile_box.area

    return ov

def whitespace_check(im):
    ''' checks if meets whitespace requirement'''
    bw = im.convert('L')
    bw = np.array(bw)
    bw = bw.astype('float')
    bw = bw / 255
    prop_ws = (bw > 0.8).sum() / (bw > 0).sum()
    return prop_ws

def cancer_mask(pred_image,hetissue):
    ''' smooth cancer map and find high probability areas '''
    # get pred image, mask at 50% and find regions
    preds = np.array(pred_image).astype('float64')
    preds = preds / 255
    preds[hetissue < 1] = 0
    preds = filters.gaussian(preds,sigma=10)
    preds_mask = np.zeros(preds.shape)
    preds_mask[preds > 0.8] = 1
    preds_mask = morphology.binary_dilation(preds_mask, morphology.disk(radius=2))
    preds_mask = morphology.binary_erosion(preds_mask, morphology.disk(radius=2))
    preds_mask = ndimage.binary_fill_holes(preds_mask)
    labels = measure.label(preds_mask)
    regions = measure.regionprops(labels,preds)
    for reg in regions:
        if reg.max_intensity<0.6:
            labels[labels==reg.label]=0
    labels[labels>0]=1
    return labels

def cancer_mask2(pred_image,hetissue, thres):
    ''' smooth cancer map and find high probability areas '''
    # get pred image, mask at 50% and find regions
    preds = np.array(pred_image).astype('float64')
    preds = preds / 255
    preds[hetissue < 1] = 0
    preds = filters.gaussian(preds,sigma=10)
    preds_mask = np.zeros(preds.shape)
    preds_mask[preds > thres] = 1
    preds_mask = morphology.binary_dilation(preds_mask, morphology.disk(radius=2))
    preds_mask = morphology.binary_erosion(preds_mask, morphology.disk(radius=2))
    preds_mask = ndimage.binary_fill_holes(preds_mask)
    labels = measure.label(preds_mask)
    regions = measure.regionprops(labels,preds)
    for reg in regions:
        if reg.max_intensity<0.6:
            labels[labels==reg.label]=0
    labels[labels>0]=1
    return labels

def cancer_mask_fix_res(preds,hetissue, thres):
    ''' smooth cancer map and find high probability areas '''
    preds[hetissue < 1] = 0 #If tissue map value < 1, then preds = 0
    preds_mask = np.zeros(preds.shape)
    preds_mask[preds > thres] = 1
    preds_mask = morphology.binary_dilation(preds_mask, morphology.disk(radius=2))
    preds_mask = morphology.binary_erosion(preds_mask, morphology.disk(radius=2))
    preds_mask = ndimage.binary_fill_holes(preds_mask)
    return preds_mask
    
def slide_ROIS(polygons,mpp,savename,labels,ref,roi_color):
    ''' generate geojson from polygons '''
    all_polys = unary_union(polygons)
    final_polys = []
    if all_polys.type == 'Polygon':
        poly = all_polys
        polypoints = poly.exterior.xy
        polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
        polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
        newpoly = Polygon(zip(polyx, polyy))
        if newpoly.area*mpp*mpp > 0.1:
            final_polys.append(newpoly)

    else:
        for poly in all_polys.geoms:
            # print(poly)
            if poly.type == 'Polygon':
                polypoints = poly.exterior.xy
                polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                newpoly = Polygon(zip(polyx, polyy))
                if newpoly.area*mpp*mpp > 0.1:
                    final_polys.append(newpoly)
            if poly.type == 'MultiPolygon':
                for roii in poly.geoms:
                    polypoints = roii.exterior.xy
                    polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                    polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                    newpoly = Polygon(zip(polyx, polyy))
                    if newpoly.area*mpp*mpp > 0.1:
                        final_polys.append(newpoly)

    final_shape = unary_union(final_polys)
    try:
        trythis = '['
        for i in range(0, len(final_shape)):
            trythis += json.dumps(
                {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape[i]),
                "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                                "measurements": []}}, indent=4)
            if i < len(final_shape) - 1:
                trythis += ','
        trythis += ']'
    except:
        trythis = '['
        trythis += json.dumps(
            {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape),
            "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                            "measurements": []}}, indent=4)
        trythis += ']'

    with open(savename, 'w') as outfile:
        outfile.write(trythis)
    return

def tile_ROIS(mask_arr,lvl_resize):
    ''' get cancer polygons '''
    polygons = []
    contours, hier = cv2.findContours(mask_arr.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cvals = contour.transpose(0, 2, 1)
        cvals = np.reshape(cvals, (cvals.shape[0], 2))
        cvals = cvals.astype('float64')
        for i in range(len(cvals)):
            cvals[i][0] = np.round(cvals[i][0]*lvl_resize,2)
            cvals[i][1] = np.round(cvals[i][1]*lvl_resize,2)
        try:
            poly = Polygon(cvals)
            if poly.length > 0:
                polygons.append(Polygon(poly.exterior))
        except:
            pass

    return polygons

def get_y(row): return row['label'].split(" ")


def get_mpp_x(slide):
    # case 2: OME XML metadata
    xml = slide.properties.get("openslide.comment") or slide.properties.get("tiff.ImageDescription")
    if xml:
        root = ET.fromstring(xml)
        pixels = root.find(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels")
        if pixels is not None:
            return float(pixels.attrib["PhysicalSizeX"])

    raise ValueError("MPP not found in slide metadata")


def generate_deepzoom_tiles(slide, save_image_size, pixel_overlap, limit_bounds):
    
    
    if openslide.PROPERTY_NAME_MPP_X in slide.properties:
        mpp_x = slide.properties[openslide.PROPERTY_NAME_MPP_X]
    else: #for CCF
        mpp_x = get_mpp_x(slide)
    
    # this is physical microns per pixel
    acq_mag = 10.0/float(mpp_x)


    # this is nearest multiple of 20 for base layer
    base_mag = int(20 * round(float(acq_mag) / 20))

    # this is how much we need to resample our physical patches for uniformity across studies
    physSize = round(save_image_size*acq_mag/base_mag)

    # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
    tiles = DeepZoomGenerator(slide, tile_size=physSize-round(pixel_overlap*acq_mag/base_mag), overlap=round(pixel_overlap*acq_mag/base_mag/2), 
                              limit_bounds=limit_bounds)

    # calculate the effective magnification at each level of tiles, determined from base magnification
    tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))

    return tiles, tile_lvls, physSize, base_mag, mpp_x


def extract_tile_start_end_coords(all_tile, deepzoom_lvl, x_loc, y_loc):
    r'''
    #This func returns the coordiates in the reference level 0 pixels
    '''
    #Get coords
    tile_coords = all_tile.get_tile_coordinates(deepzoom_lvl, (x_loc, y_loc)) # location - (x, y) tuple giving the top left pixel in the level 0 reference frame

    #Get top left pixel coordinates in the level 0 reference
    topleft_x = tile_coords[0][0]
    topleft_y = tile_coords[0][1]

    #Get level (original)
    o_lvl = tile_coords[1]

    #Get downsample factor
    ds_factor = all_tile._l0_l_downsamples[o_lvl] #downsample factor

    #Get region size in current level 
    rsize_x = tile_coords[2][0] 
    rsize_y = tile_coords[2][1] 

    #Get tile starts and end   
    start_loc = tile_coords[0] #start
    end_loc = (int(topleft_x + ds_factor * rsize_x), int(topleft_y + ds_factor* rsize_y)) #end

    #Get save coord name (first two is the starting loc, and the last two are the x and y size considering dsfactor)
    coord_name = str(topleft_x) + "-" + str(topleft_y) + "_" + '%.0f' % (ds_factor * rsize_x) + "-" + '%.0f' % (ds_factor * rsize_y)
    
    return start_loc, end_loc, coord_name, tile_coords


def get_map_startend(tile_start_cord, tile_end_cord, level_resize):
    r'''
    Note: The maps has x and y swaped relatived to the original slides
    #Change from floor to ceil 11/06/24
    '''
    m_xstart = int(np.ceil(tile_start_cord[1] / level_resize))
    m_xend = int(np.ceil(tile_end_cord[1] / level_resize))
    m_ystart = int(np.ceil(tile_start_cord[0] / level_resize))
    m_yend = int(np.ceil(tile_end_cord[0] / level_resize))

    return m_xstart, m_xend, m_ystart, m_yend


def get_downsample_factor(base_magnification, target_magnification):
    
    ds_factor =(base_magnification/target_magnification) #downsample factor
    
    return ds_factor


def get_image_at_target_mag(slide_obj,level0_width, level0_height,downsample_factor):
    getLvl = slide_obj.get_best_level_for_downsample(downsample_factor)
    img_pull = slide_obj.read_region((0, 0), getLvl, slide_obj.level_dimensions[getLvl]) #image at level closed to target_magnification
    img_pull = img_pull.resize(size=(int(np.ceil(level0_width/downsample_factor)),int(np.ceil(level0_height/downsample_factor))),resample=PIL.Image.LANCZOS) #resize to actual target_magnification

    return img_pull

def check_any_invalid_poly(input_polygons):
    invalid_poly = [x for x in input_polygons if not x.is_valid]
    return invalid_poly     
    
def make_valid_poly(input_polygons, buff_value = 4):
    r'''
    For invalid polygon error, add buffer to make the polygon valid:
    '''

    smooth_polygon = []
    for poly in input_polygons:
        if not poly.is_valid:
            buffered_poly = poly.buffer(buff_value)
        else:
            buffered_poly = poly
        smooth_polygon.append(buffered_poly)
    return smooth_polygon


def log_message(message, file_name):
    """Logs a message to the specified file with a timestamp."""
    with open(file_name, 'a') as file:  # 'a' mode appends to the file
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f'{timestamp} {message}\n')





def count_label(label_df, selected_label_names, cohort_name):
    
    count_list = []
    for l in selected_label_names:
        cur_count = pd.DataFrame(label_df[l].value_counts()).T
        cur_count.index = [l]
        count_list.append(cur_count)
    count_df = pd.concat(count_list)
    count_df.columns = ['N0','N1']
    count_df['Perc0'] = round(count_df['N0']/(count_df['N0'] + count_df['N1'])*100,1)
    count_df['Perc1'] = round(count_df['N1']/(count_df['N0'] + count_df['N1'])*100,1)

    count_df.columns =  cohort_name + '_' + count_df.columns

    return count_df


def simple_line_plot(x,y, x_lab, y_lab, plt_title, plt_save_dir, fig_size = (10,6)):
    plt.figure(figsize= fig_size)
    plt.plot(x, y)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(plt_title)
    plt.tight_layout()
    plt.savefig(plt_save_dir)
    plt.show()
    plt.close()



def generating_tiles(pt_id,_file, save_image_size, pixel_overlap, limit_bounds, mag_target_tiss, rad_tissue, mag_extract):
    
    #Load slides
    oslide = openslide.OpenSlide(_file)

    #Generate tiles
    tiles, tile_lvls, physSize, base_mag, mpp_x = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)

    #get level 0 size in px
    l0_w = oslide.level_dimensions[0][0]
    l0_h = oslide.level_dimensions[0][1]


    #1.25x for low res img 
    lvl_resize = get_downsample_factor(base_mag,target_magnification = mag_target_tiss) #downsample factor
    lvl_img = get_image_at_target_mag(oslide,l0_w, l0_h,lvl_resize)
    lvl_img = convert_img(lvl_img)
    
    # 1.25X for tissue detection
    print('detecting tissue')
    tissue, he_mask = do_mask_original(lvl_img, lvl_resize, rad = rad_tissue)
    lvl_mask = PIL.Image.fromarray(np.uint8(he_mask * 255))
    lvl_mask = lvl_mask.convert('L')


    print('Start pulling tiles')
    lvl =  mag_extract
    if lvl in tile_lvls:
        #get deep zoom levels
        lvl_in_deepzoom = tile_lvls.index(lvl)
        # pull tile info for level
        x_tiles, y_tiles = tiles.level_tiles[lvl_in_deepzoom] #this extract tiles at mag_extract
        print(y_tiles,x_tiles)
        tile_info = []
        for y in range(0, y_tiles):
            if y % 50 == 0: print(y)
            for x in range(0, x_tiles):
                #Grab tile coordinates
                tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords(tiles, lvl_in_deepzoom, x, y) #this returns the coors at level 0 reference original slides

                #Check tissue membership
                tile_tiss = check_tissue(tile_starts= tile_starts, tile_ends=tile_ends,roi=tissue)
                if tile_tiss > 0.9:
                    #Extract tile
                    tile_pull = tiles.get_tile(lvl_in_deepzoom, (x, y))

                    #Check white space
                    ws = whitespace_check(im=tile_pull)

                    if ws < 0.9: #If the white space is less than 90%
                        tile_info.append(pd.DataFrame({'SAMPLE_ID' : pt_id, 
                                                       'MAG_EXTRACT' : lvl,
                                                       'SAVE_IMAGE_SIZE': save_image_size,
                                                       'PIXEL_OVERLAP': pixel_overlap,
                                                       'LIMIT_BOUNDS': limit_bounds,
                                                       'TILE_XY_INDEXES' : str((x ,y)),
                                                       'TILE_COOR_ATLV0' : save_coords,
                                                       'WHITE_SPACE' : ws,
                                                       'TISSUE_COVERAGE': tile_tiss}, index = [0]))

    else:
        print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")

    tile_info_df = pd.concat(tile_info)

    return mpp_x, lvl_img, lvl_mask, tissue, tile_info_df


def calculate_num_tiles(slide_size, tile_size, overlap=0):
    """
    Calculate the number of tiles for a given image size, tile size, and overlap.
    
    Parameters:
    - image_size: Tuple (width, height) of the image.
    - tile_size: Size of each tile (assuming square tiles).
    - overlap: Number of overlapping pixels (default is 0 for non-overlapping).
    
    Returns:
    - num_tiles_width: Number of tiles along the width.
    - num_tiles_height: Number of tiles along the height.
    """
    stride = tile_size - overlap
    
    num_tiles_width = math.floor((slide_size[0] - tile_size) / stride) + 1
    num_tiles_height = math.floor((slide_size[1] - tile_size) / stride) + 1
    
    return num_tiles_width, num_tiles_height


def extract_tile_start_end_coords_tma(x_loc, y_loc, tile_size, overlap):
    r'''
    #This func returns the coordiates in the original image of TMA at original dim
    '''
    #Get stride
    stride = tile_size - overlap
    
    #Get top left pixel coordinates
    topleft_x = stride * (x_loc - 1) 
    topleft_y = stride * (y_loc - 1)
    
    #Get region size in current level 
    rsize_x = tile_size 
    rsize_y = tile_size
    
    #Get tile starts and end   
    start_loc = (topleft_x, topleft_y) #start
    end_loc = (topleft_x + rsize_x, topleft_y + rsize_y) #end
    
    #Get save coord name (first two is the starting loc, and the last two are the x and y size considering dsfactor)
    coord_name = str(topleft_x) + "-" + str(topleft_y) + "_" + '%.0f' % (rsize_x) + "-" + '%.0f' % (rsize_y)

    #Get tile_coords the same format as OPX, (start_loc, deepzzomlvl, rise)
    tile_coords = (start_loc, 'NA', (rsize_x, rsize_y))
    
    return start_loc, end_loc, coord_name, tile_coords


def generating_tiles_tma(pt_id, _file, save_image_size, pixel_overlap, rad_tissue):
    #Load slides
    tma = PIL.Image.open(_file)
    
    #get level 0 size in px
    l0_w = tma.size[0]
    l0_h = tma.size[1]
    
    #4x for low res img 500x500 (assume base_mag @40)
    mpp = 40 #for generting _tissue.json
    lvl_resize = 10  #get_downsample_factor(base_mag,target_magnification = 4) #downsample factor
    lvl_img = tma.resize(size=(int(np.ceil(l0_w/lvl_resize)),int(np.ceil(l0_h/lvl_resize))), resample=PIL.Image.LANCZOS)
    lvl_img = convert_img(lvl_img)
    
    #4x for tissue detection, asssume original tma image is at 40x
    print('detecting tissue')
    tissue, he_mask = do_mask_original(lvl_img, lvl_resize, rad = rad_tissue)
    lvl_mask = PIL.Image.fromarray(np.uint8(he_mask * 255))
    lvl_mask = lvl_mask.convert('L')
    
    print('Start pulling tiles')
    num_subs_x, num_subs_y = calculate_num_tiles(slide_size = tma.size,tile_size = save_image_size, overlap = pixel_overlap)
    print(num_subs_y,num_subs_x)
    
    tile_info = []
    for x in range(1, num_subs_x + 1):
        for y in range(1, num_subs_y + 1):
            #Grab tile coordinates
            tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords_tma(x, y, tile_size = save_image_size, overlap = pixel_overlap)
            
            #Check tissue membership
            tile_tiss = check_tissue(tile_starts= tile_starts, tile_ends=tile_ends,roi=tissue)
            if tile_tiss > 0.9:
                #Extract tile
                tile_pull = tma.crop(box=(tile_starts[0], tile_starts[1], tile_ends[0], tile_ends[1]))
            
                #Check white space
                ws = whitespace_check(im=tile_pull)
                
                if ws < 0.9: #If the white space is less than 90%
                    tile_info.append(pd.DataFrame({'SAMPLE_ID' : pt_id, 
                                                   'MAG_EXTRACT' : pd.NA,
                                                   'SAVE_IMAGE_SIZE': save_image_size,
                                                   'PIXEL_OVERLAP': pixel_overlap,
                                                   'LIMIT_BOUNDS': pd.NA,
                                                   'TILE_XY_INDEXES' : str((x ,y)),
                                                   'TILE_COOR_ATLV0' : save_coords,
                                                   'WHITE_SPACE' : ws,
                                                   'TISSUE_COVERAGE': tile_tiss}, index = [0]))
    tile_info_df = pd.concat(tile_info)
    
    return mpp, lvl_img, lvl_mask, tissue, tile_info_df


#Grab tiles and plot
def plot_tiles_with_topK_cancerprob(tile_info_data, tiles, tile_lvls, mag_extract, save_image_size, save_location, stain_norm_target_img = None, k = 5):
    
    tile_info_data_sorted= tile_info_data.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
    for i in range(0,k): #top5
        #Get tile coords
        cur_row = tile_info_data_sorted.iloc[i]
        cur_xy = cur_row['TILE_XY_INDEXES'].strip("()").split(", ")
        x ,y = int(cur_xy[0]) , int(cur_xy[1])

        #Grab tile
        tile_pull_ex = tiles.get_tile(tile_lvls.index(mag_extract), (x, y))
        tile_pull_ex = tile_pull_ex.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
        tile_pull_ex = convert_img(tile_pull_ex)
        
        if stain_norm_target_img is not None:
            tile_pull_ex = preprocessing.color_normalization.deconvolution_based_normalization(im_src=np.asarray(tile_pull_ex), im_target=stain_norm_target_img) #a color-adjusted version of your input tile 
            tile_pull_ex = Image.fromarray(tile_pull_ex)
                
        #Save tile
        cur_tf = round(cur_row['TUMOR_PIXEL_PERC'],2)
        cur_mag = cur_row['MAG_EXTRACT']
        tile_save_name = "TILE_@" + str(cur_mag) + "x" + "_X" + str(x) +  "Y" + str(y) +   "_TF" + str(cur_tf) + ".png"
        tile_pull_ex.save(os.path.join(save_location, tile_save_name))

def plot_tiles_with_topK_cancerprob_tma(tile_info_data, tma, pixel_overlap, save_image_size, save_location, stain_norm_target_img = None, k = 5):

    tile_info_data_sorted= tile_info_data.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
    for i in range(0,k): #top5
        #Get tile coords
        cur_row = tile_info_data_sorted.iloc[i]
        cur_xy = cur_row['TILE_XY_INDEXES'].strip("()").split(", ")
        x ,y = int(cur_xy[0]) , int(cur_xy[1])
    
        #Grab tile coordinates
        tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords_tma(x, y, tile_size = save_image_size, overlap = pixel_overlap)
    
        #Grab tile
        tile_pull_ex = tma.crop(box=(tile_starts[0], tile_starts[1], tile_ends[0], tile_ends[1]))
        tile_pull_ex = tile_pull_ex.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
        tile_pull_ex = convert_img(tile_pull_ex)
        
        tile_pull_ex = tile_pull_ex.convert("RGB")
        if stain_norm_target_img is not None:
            tile_pull_ex = preprocessing.color_normalization.deconvolution_based_normalization(im_src=np.asarray(tile_pull_ex), im_target=stain_norm_target_img) #a color-adjusted version of your input tile 
            tile_pull_ex = Image.fromarray(tile_pull_ex)
        
        #Save tile
        cur_tf = round(cur_row['TUMOR_PIXEL_PERC'],2)
        cur_mag = cur_row['MAG_EXTRACT']
        tile_save_name = "TILE_@" + str(cur_mag) + "x" + "_X" + str(x) +  "Y" + str(y) +   "_TF" + str(cur_tf) + ".png"
        tile_pull_ex.save(os.path.join(save_location, tile_save_name))

def get_binary_pred_tile(tile_info_data, binary_pred_matrix):
    tile_info_data['TUMOR_PIXEL_PERC'] = pd.NA
    for index, row in tile_info_data.iterrows():
        cur_map_loc = row['pred_map_location'].strip("()").split(", ")
        map_xstart, map_xend, map_ystart, map_yend = int(cur_map_loc[0]),int(cur_map_loc[1]), int(cur_map_loc[2]), int(cur_map_loc[3])
    
        #Get current prediction
        cur_pred = binary_pred_matrix[map_xstart:map_xend,map_ystart:map_yend]
        cur_count1 = np.sum(cur_pred == 1) #num pixels that has predicted prob = 1
        cur_perc1  = (cur_count1 / cur_pred.size) #fraction of pixels prob = 1
        tile_info_data.loc[index,'TUMOR_PIXEL_PERC'] = cur_perc1

    return tile_info_data


def check_tile_info_match(tile_info_data, mag_extract, save_image_size, pixel_overlap, limit_bounds):
    
    tile_mag_extract = list(set(tile_info_data['MAG_EXTRACT']))[0]
    tile_save_image_size = list(set(tile_info_data['SAVE_IMAGE_SIZE']))[0]
    tile_pixel_overlap = list(set(tile_info_data['PIXEL_OVERLAP']))[0]
    tile_limit_bounds =   list(set(tile_info_data['LIMIT_BOUNDS']))[0]

    cond1 = (tile_mag_extract == mag_extract)
    cond2 = (tile_save_image_size == save_image_size)
    cond3 = (tile_pixel_overlap == pixel_overlap)
    cond4 = (tile_limit_bounds == limit_bounds)

    if cond1 & cond2 & cond3 & cond4:
        can_proceed = True
        print(can_proceed)
    return can_proceed



def cancer_inference_wsi(_file, 
                         model, 
                         tile_info_df, 
                         mag_extract, 
                         save_image_size, 
                         pixel_overlap, 
                         limit_bounds, 
                         mag_target_prob, 
                         mag_target_tiss, 
                         rad_tissue, 
                         smooth, 
                         bi_thres, 
                         save_location, 
                         save_name,
                         stain_norm_target_img = None):
    
    #check to see if info matches, and if can proceed
    can_proceed = check_tile_info_match(tile_info_df, mag_extract, save_image_size, pixel_overlap, limit_bounds) 


    if can_proceed == True:
        #Load slides
        oslide = openslide.OpenSlide(_file)
        
        #Generate tiles
        tiles, tile_lvls, physSize, base_mag, mpp_x = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
    
        print('starting inference')
        #get level 0 size in px
        l0_w = oslide.level_dimensions[0][0]
        l0_h = oslide.level_dimensions[0][1]
    
        #2.5x for probability maps
        lvl_resize = get_downsample_factor(base_mag,target_magnification = mag_target_prob) #downsample factor
        x_map = np.zeros((int(np.ceil(l0_h/lvl_resize)),int(np.ceil(l0_w/lvl_resize))), float)
        x_count = np.zeros((int(np.ceil(l0_h/lvl_resize)),int(np.ceil(l0_w/lvl_resize))), float)
    
        
        tile_info_df['pred_map_location'] = pd.NA
        for index, row in tile_info_df.iterrows():
            if (index % 500 == 0): print(index)
            cur_xy = row['TILE_XY_INDEXES'].strip("()").split(", ")
            x ,y = int(cur_xy[0]) , int(cur_xy[1])
            #Extract tile for prediction
            lvl_in_deepzoom = tile_lvls.index(mag_extract)
            tile_pull = tiles.get_tile(lvl_in_deepzoom, (x, y))
            tile_pull = tile_pull.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
            
            if stain_norm_target_img is not None:
                try:
                    tile_pull = preprocessing.color_normalization.deconvolution_based_normalization(im_src=np.asarray(tile_pull), im_target=stain_norm_target_img) #a color-adjusted version of your input tile 
                    tile_pull = Image.fromarray(tile_pull)
                except np.linalg.LinAlgError:
                    print("Deconvolution failed on a tile – skipping") #this is due to some tiles are not actuallly not tissue, all black (Neptune), just skip the norm
                    pass
                
            tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords(tiles, lvl_in_deepzoom, x, y) #get tile coords
            map_xstart, map_xend, map_ystart, map_yend = get_map_startend(tile_starts,tile_ends,lvl_resize) #Get current tile position in map
            tile_info_df.loc[index,'pred_map_location'] = str(tuple([map_xstart, map_xend, map_ystart, map_yend]))

            #Cancer segmentation
            tile_pull = np.array(tile_pull)
            with model.no_bar():
                inp, pred_class, pred_idx, outputs = model.predict(tile_pull[:, :, 0:3], with_input=True)
            
            #Get predicted output
            #NOTe: updated 11/06, use cv2.resize
            outputs_np = outputs.numpy() #[N_CLASS, IMAGE_SIZE, IMAGE_SIZE]
            output_c1_np = cv2.resize(outputs_np[1], (map_yend - map_ystart,map_xend - map_xstart)) #class1 predicted prob, resize (width (col in np), height(row in np))
            output_c1_np = output_c1_np.round(2)
            
            #Store predicted probabily in map and count
            try: 
                x_count[map_xstart:map_xend,map_ystart:map_yend] += 1
                x_map[map_xstart:map_xend,map_ystart:map_yend] += output_c1_np
            except:
                pass
    
        print('post-processing')
        print('Cancer Prob generation')
        x_count = np.where(x_count < 1, 1, x_count)
        x_map = x_map / x_count
        x_map[x_map>1]=1

        if smooth == True:
            x_sm = filters.gaussian(x_map, sigma=2)
        if smooth == False:
            x_sm = x_map
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(x_sm)
        rgb_img = np.delete(rgba_img, 3, 2)
        colimg = PIL.Image.fromarray(np.uint8(rgb_img * 255))
        colimg.save(os.path.join(save_location, save_name + '_cancer_prob.jpeg'))


        print('Get cancer binary mask...')
        #1.25x tissue detection
        lvl_resize_tissue = get_downsample_factor(base_mag,target_magnification = mag_target_tiss) #downsample factor
        lvl_img = get_image_at_target_mag(oslide,l0_w, l0_h,lvl_resize_tissue)
        tissue, he_mask = do_mask_original(lvl_img, lvl_resize_tissue, rad = rad_tissue)

        #Binary classification
        binary_preds = cancer_mask_fix_res(x_sm,cv2.resize(np.uint8(he_mask),(x_sm.shape[1],x_sm.shape[0])), bi_thres)
        #Get binary prediction for each tile, #NOTE: previous do x_map when prediction, is not accurate, because the x_map may change as process to the next tile, so need to do this in post-processing
        tile_info_df = get_binary_pred_tile(tile_info_df, binary_preds)
        tile_info_df.to_csv(save_location + save_name + "_TILE_TUMOR_PERC.csv", index = False)
        
        #Output annotation
        print('Output json annotation...')
        polygons = tile_ROIS(mask_arr=binary_preds, lvl_resize=lvl_resize)

        #Make valid polygons (ex: OPX_022)
        invalid_polygons = check_any_invalid_poly(polygons) #check if there is any invalid polys
        if len(invalid_polygons) > 0 :
            polygons = make_valid_poly(polygons, buff_value = 4)
        
        slide_ROIS(polygons=polygons, mpp=float(mpp_x),
                        savename=os.path.join(save_location,save_name +'_cancer.json'), labels='AI_tumor', ref=[0,0], roi_color=-16711936)

        #Grab tiles and plot
        print('Plot top predicted  tiles...')
        plot_tiles_with_topK_cancerprob(tile_info_df, tiles, tile_lvls, mag_extract, save_image_size, save_location, stain_norm_target_img, k = 5)




def cancer_inference_tma(_file, 
                         model, 
                         tile_info_df, 
                         save_image_size, 
                         pixel_overlap, 
                         mag_target_prob, 
                         rad_tissue, 
                         smooth, 
                         bi_thres, 
                         save_location, 
                         save_name,
                         stain_norm_target_img = None):

    #Load slides
    tma = PIL.Image.open(_file)
    
    print('starting inference')
    #get level 0 size in px
    l0_w = tma.size[0]
    l0_h = tma.size[1]
    
    #2.5x for probability maps
    mpp = 40 ##assume @40
    lvl_resize = get_downsample_factor(mpp,target_magnification = mag_target_prob) #downsample factor
    x_map = np.zeros((int(np.ceil(l0_h/lvl_resize)),int(np.ceil(l0_w/lvl_resize))), float)
    x_count = np.zeros((int(np.ceil(l0_h/lvl_resize)),int(np.ceil(l0_w/lvl_resize))), float)
    
    
    tile_info_df['pred_map_location'] = pd.NA
    for index, row in tile_info_df.iterrows():
        if (index % 500 == 0): print(index)
        cur_xy = row['TILE_XY_INDEXES'].strip("()").split(", ")
        x ,y = int(cur_xy[0]) , int(cur_xy[1])
    
        #Grab tile coordinates
        tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords_tma(x, y, tile_size = save_image_size, overlap = pixel_overlap)
    
        # #Extract tile for prediction
        tile_pull = tma.crop(box=(tile_starts[0], tile_starts[1], tile_ends[0], tile_ends[1]))    
        map_xstart, map_xend, map_ystart, map_yend = get_map_startend(tile_starts,tile_ends,lvl_resize) #Get current tile position in map
        tile_info_df.loc[index,'pred_map_location'] = str(tuple([map_xstart, map_xend, map_ystart, map_yend]))
        
        
        #convert to RGB
        tile_pull = tile_pull.convert("RGB")
        
        if stain_norm_target_img is not None:
            try:
                tile_pull = preprocessing.color_normalization.deconvolution_based_normalization(im_src=np.asarray(tile_pull), im_target=stain_norm_target_img) #a color-adjusted version of your input tile 
                tile_pull = Image.fromarray(tile_pull)
            except np.linalg.LinAlgError:
                print("Deconvolution failed on a tile – skipping") #this is due to some tiles are not actuallly not tissue, all black (Neptune), just skip the norm
                pass
        
        
        #Cancer segmentation
        tile_pull = np.array(tile_pull)
        with model.no_bar():
            inp, pred_class, pred_idx, outputs = model.predict(tile_pull, with_input=True) #0:3 for RGB
        
        #Get predicted output
        #NOTe: updated 11/06, use cv2.resize
        outputs_np = outputs.numpy() #[N_CLASS, IMAGE_SIZE, IMAGE_SIZE]
        output_c1_np = cv2.resize(outputs_np[1], (map_yend - map_ystart,map_xend - map_xstart)) #class1 predicted prob, resize (width (col in np), height(row in np))
        output_c1_np = output_c1_np.round(2)
        
        #Store predicted probabily in map and count
        try: 
            x_count[map_xstart:map_xend,map_ystart:map_yend] += 1
            x_map[map_xstart:map_xend,map_ystart:map_yend] += output_c1_np
        except:
            pass
            
    print('post-processing')
    print('Cancer Prob generation')
    x_count = np.where(x_count < 1, 1, x_count)
    x_map = x_map / x_count
    x_map[x_map>1]=1
    
    if smooth == True:
        x_sm = filters.gaussian(x_map, sigma=2)
    if smooth == False:
        x_sm = x_map
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(x_sm)
    rgb_img = np.delete(rgba_img, 3, 2)
    colimg = PIL.Image.fromarray(np.uint8(rgb_img * 255))
    colimg.save(os.path.join(save_location, save_name + '_cancer_prob.jpeg'))
    
    
    print('Get cancer binary mask...')
    #4x tissue detection
    lvl_resize_tissue = 10  #get_downsample_factor(base_mag,target_magnification = 4) #downsample factor
    lvl_img = tma.resize(size=(int(np.ceil(l0_w/lvl_resize_tissue)),int(np.ceil(l0_h/lvl_resize_tissue))), resample=PIL.Image.LANCZOS)
    lvl_img = convert_img(lvl_img)
    tissue, he_mask = do_mask_original(lvl_img, lvl_resize_tissue, rad = rad_tissue)
    
    #Binary classification
    binary_preds = cancer_mask_fix_res(x_sm,cv2.resize(np.uint8(he_mask),(x_sm.shape[1],x_sm.shape[0])), bi_thres)
    #Get binary prediction for each tile, #NOTE: previous do x_map when prediction, is not accurate, because the x_map may change as process to the next tile, so need to do this in post-processing
    tile_info_df = get_binary_pred_tile(tile_info_df, binary_preds)
    tile_info_df.to_csv(save_location + save_name + "_TILE_TUMOR_PERC.csv", index = False)
    
    #Output annotation
    print('Output json annotation...')
    polygons = tile_ROIS(mask_arr=binary_preds, lvl_resize=lvl_resize)
    
    #Make valid polygons (ex: OPX_022)
    invalid_polygons = check_any_invalid_poly(polygons) #check if there is any invalid polys
    if len(invalid_polygons) > 0 :
        polygons = make_valid_poly(polygons, buff_value = 4)
    
    slide_ROIS(polygons=polygons, mpp=float(mpp),
                    savename=os.path.join(save_location,save_name +'_cancer.json'), labels='AI_tumor', ref=[0,0], roi_color=-16711936)
    
    #Grab tiles and plot
    print('Plot top predicted  tiles...')
    plot_tiles_with_topK_cancerprob_tma(tile_info_df,tma, pixel_overlap, save_image_size, save_location, stain_norm_target_img, k = 5)


