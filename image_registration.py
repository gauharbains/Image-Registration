#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:58:14 2019

@author: gauhar
"""

import cv2
import numpy as np
from PIL import Image
from bfio import bfio 
import bioformats     
import javabridge 

import argparse

Image.MAX_IMAGE_PIXELS = 1500000000 

def image_transformation(img1,img2):    
    '''
    Image 1= Image to be transformed-
    Image 2=  reference Image    
    '''   
    max_features=500000
    good_match_percent=0.05   
    orb = cv2.ORB_create(max_features)    
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt        
    height, width = img2.shape    
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    warped_img = cv2.warpPerspective(img1, h, (width, height))    
    return warped_img,h  

def get_scaled_down_images(image,scale_factor):
    height,width=image.shape
    new_height,new_width=[int(height/scale_factor),int(width/scale_factor)]    
    rescaled_image=cv2.resize(image,(new_width,new_height))
    rescaled_image=(rescaled_image/256).astype('uint8')
    return rescaled_image

def get_four_tiles_8bit(image):
    h,w=image.shape
    tile1=image[:int(h/2),:int(w/2)]
    tile2=image[:int(h/2),int(w/2):]
    tile3=image[int(h/2):,:int(w/2)]
    tile4=image[int(h/2):,int(w/2):] 
    return tile1,tile2,tile3,tile4

def get_four_tiles_with_buffer_8bit(image):
    h,w=image.shape
    tile1=image[:int(5*h/8),:int(5*w/8)]
    tile2=image[:int(5*h/8),int(3*w/8):]
    tile3=image[int(3*h/8):,:int(5*w/8)]
    tile4=image[int(3*h/8):,int(3*w/8):]
    tile1=(tile1/256).astype('uint8')
    tile2=(tile2/256).astype('uint8')
    tile3=(tile3/256).astype('uint8')
    tile4=(tile4/256).astype('uint8')
    return tile1,tile2,tile3,tile4

def get_four_tiles_with_buffer_16bit(image):
    h,w=image.shape
    tile1=image[:int(5*h/8),:int(5*w/8)]
    tile2=image[:int(5*h/8),int(3*w/8):]
    tile3=image[int(3*h/8):,:int(5*w/8)]
    tile4=image[int(3*h/8):,int(3*w/8):]
    return tile1,tile2,tile3,tile4

def stack_tiles(tile1,tile2,tile3,tile4):
    upper_half=np.hstack((tile1,tile2))
    lower_half=np.hstack((tile3,tile4))
    stacked_image=np.vstack((upper_half,lower_half))
    return stacked_image 

def get_homography_scale_matrix(scale_factor):
    return np.array([[1,1,scale_factor],[1,1,scale_factor],[1/scale_factor,1/scale_factor,1]])   

def stack_eight_tiles(tile1,tile2,tile3,tile4,tile5,tile6,tile7,tile8):
    upper_half=np.hstack((tile1,tile2,tile3,tile4))
    lower_half=np.hstack((tile5,tile6,tile7,tile8))
    stacked_image=np.vstack((upper_half,lower_half))
    return stacked_image   

def apply_rough_homography(image,homography_smallscale,reference_image,scale_factor):
    scale_matrix = np.array([[1,1,scale_factor],[1,1,scale_factor],[1/scale_factor,1/scale_factor,1]])
    homography_largescale= scale_matrix *homography_smallscale    
    homography_inverse=np.linalg.inv(homography_largescale)
    transformed_image=np.zeros_like(reference_image)
    height_1,width_1=reference_image.shape
    height_2,width_2=image.shape
    row_array=np.zeros((3,width_1) ,dtype='uint16')   
    for i in range(height_1):
        row_array=np.array([[x for x in range(width_1)],[i for x in range(width_1)] ,[1 for x in range(width_1)]])
        new_array=np.dot(homography_inverse,row_array)
        new_array=np.round(new_array/new_array[2,:], decimals=0)
        new_array=new_array.astype(int)
        boo = (np.all(new_array>=0, axis =0)) * (new_array[0] <width_2) * (new_array[1] <height_2)
        new_array=new_array[:,boo]
        row_array=row_array[:,boo]
        transformed_image[row_array[1,:], row_array[0,:]]= image[new_array[1,:], new_array[0,:]]    
    return transformed_image

def get_tile_by_tile_transformation(ref_tile1,ref_tile2,ref_tile3,ref_tile4,moving_tile1,moving_tile2,moving_tile3,moving_tile4):
    _,homography1_downscale=image_transformation(moving_tile1,ref_tile1)
    _,homography2_downscale=image_transformation(moving_tile2,ref_tile2)
    _,homography3_downscale=image_transformation(moving_tile3,ref_tile3)
    _,homography4_downscale=image_transformation(moving_tile4,ref_tile4)
    return homography1_downscale,homography2_downscale,homography3_downscale,homography4_downscale

def upscale_homography_matrices(Homography1,Homography2,Homography3,Homography4,scale_factor):
    scale_matrix=get_homography_scale_matrix(scale_factor)
    Homography1=scale_matrix *Homography1 
    Homography2=scale_matrix *Homography2 
    Homography3=scale_matrix *Homography3 
    Homography4=scale_matrix *Homography4 
    return Homography1,Homography2,Homography3,Homography4
    
def register_images(reference_image, moving_image):
    height,width=reference_image.shape
    reference_image_downscaled= get_scaled_down_images(reference_image,16)
    moving_image_downscaled= get_scaled_down_images(moving_image,16)
    _,Rough_Homography_Downscaled = image_transformation(moving_image_downscaled,reference_image_downscaled)
    moving_image_transformed=apply_rough_homography(moving_image,Rough_Homography_Downscaled,reference_image,16)
    del moving_image
    del reference_image     
    
    moving_image_transformed_downscaled=get_scaled_down_images(moving_image_transformed,16)
    cv2.imshow('moving_image_transformed_downscaled',moving_image_transformed_downscaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    reference_image_tile1,reference_image_tile2,reference_image_tile3,reference_image_tile4=get_four_tiles_8bit(reference_image_downscaled)
    moving_image_tile1,moving_image_tile2,moving_image_tile3,moving_image_tile4=get_four_tiles_with_buffer_16bit(moving_image_transformed_downscaled)
    homography1_downscale,homography2_downscale,homography3_downscale,homography4_downscale = get_tile_by_tile_transformation(reference_image_tile1,reference_image_tile2,reference_image_tile3,reference_image_tile4, moving_image_tile1,moving_image_tile2,moving_image_tile3,moving_image_tile4)
    
    homography1_upscale,homography2_upscale,homography3_upscale,homography4_upscale= upscale_homography_matrices(
                                                                                    homography1_downscale,homography2_downscale,homography3_downscale,homography4_downscale, 16) 
    moving_image_tile1,moving_image_tile2,moving_image_tile3,moving_image_tile4=get_four_tiles_with_buffer_16bit(moving_image_transformed)    
    moving_image_transformed_tile1=cv2.warpPerspective(moving_image_tile1,homography1_upscale,(int(width/2),int(height/2)))
    moving_image_transformed_tile2=cv2.warpPerspective(moving_image_tile2,homography2_upscale,(int(width/2),int(height/2)))
    moving_image_transformed_tile3=cv2.warpPerspective(moving_image_tile3,homography3_upscale,(int(width/2),int(height/2)))
    moving_image_transformed_tile4=cv2.warpPerspective(moving_image_tile4,homography4_upscale,(int(width/2),int(height/2)))    
    transformed_moving_image=stack_tiles(moving_image_transformed_tile1,moving_image_transformed_tile2,moving_image_transformed_tile3,moving_image_transformed_tile4)
    return transformed_moving_image
##################################################################################################################################################################

javabridge.start_vm(class_path=bioformats.JARS)
parser=argparse.ArgumentParser()
parser.add_argument('--templateDir',dest='Template_Image_Path',type=str,required=True)
parser.add_argument('--movingDir',dest='Moving_Image_Path',type=str,required=True)
args = parser.parse_args()


template_image_path = args.Template_Image_Path
print(template_image_path)
bf = bfio.BioReader(template_image_path)
template_image = bf.read_image()
template_image=template_image[:,:,0,0,0]


moving_image_path = args.Moving_Image_Path
print(moving_image_path)
bf = bfio.BioReader(moving_image_path)
moving_image = bf.read_image()
moving_image_metadata=bf.read_metadata()
moving_image=moving_image[:,:,0,0,0]

transformed_moving_image=register_images(template_image, moving_image)
transformed_moving_image_5channel=np.zeros((transformed_moving_image.shape[0],transformed_moving_image.shape[1],1,1,1),dtype='uint16') 
transformed_moving_image_5channel[:,:,0,0,0]=transformed_moving_image
print(moving_image_metadata)
output_image_name=".ome.tif"
bw = bfio.BioWriter(output_image_name,image=transformed_moving_image_5channel)

bw.write_image(transformed_moving_image_5channel)
bw.close_image()
javabridge.kill_vm()

################################################################################################################################################################




    

    
    
    
    
                                                                                    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    