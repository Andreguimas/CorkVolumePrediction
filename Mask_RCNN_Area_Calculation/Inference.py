import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from cv2 import waitKey
import skimage.io
from sklearn import metrics
from IPython import get_ipython
sys.path.append("./mrcnn")
from mrcnn import *
from mrcnn.visualize import random_colors, get_mask_contours, draw_mask, save_image
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.m_rcnn import load_image_dataset_new
import numpy as np
import cv2
import csv

import tensorflow as tf


#define class names and colors for masks
class_names = ["BG","sobreiro", "alvo"]
colors = random_colors(len(class_names))

#Config of the loaded model
class SobreirosConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sobreiros"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    
    DETECTION_MIN_CONFIDENCE = 0.7

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

#Config for inference process (overwrites previous configs)
class InferenceConfig(SobreirosConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 

#Load pre-trained model for inference
def recreate_model():
    # Directory to save logs and trained model
    MODEL_DIR = os.path.abspath(r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica\mask\Mask_RCNN\logs") #change for absolute path for the directory to save training logs and trained weights
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = os.path.abspath(r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica\mask\Mask_RCNN\logs\sobreiros20220209T1444\mask_rcnn_sobreiros_0023.h5") #change to absolute path of the model

    # Load trained weights
    model.load_weights(model_path, by_name=True)

    return model


def run_detection(model, image_name, IMAGE_DIR):

    img_original = cv2.imread(os.path.join(IMAGE_DIR, image_name))

    width = int(1024)
    height = int(1024)
    dim = (width, height)

    image = cv2.resize(img_original, dim, interpolation = cv2.INTER_AREA)
    
     # Run detection
    results = model.detect([image], verbose=1)

    r = results[0]
    
    if(len(r['class_ids']) == 0):
        return [], [], [], image

    return r['class_ids'], r['rois'], r['masks'], image

def calculate_trunk_areas(class_ids, boxes, masks, img, alvos, count_alvos):
    counter = 0
    area_m2 = 0
    sum_ratios = 0
    for ratio in alvos['Ratio_px2_to_cm2']:
        sum_ratios = sum_ratios + ratio
    
    media_ratios = sum_ratios/count_alvos

    N = boxes.shape[0]
    for i in range(N):
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        if(class_ids[i] == 1):
            counter = counter + 1
            if(counter > 1):
                print("Multiplos troncos detetados! Utilizar outra fotografia!")
                return 0
            mask = masks[:, :, i]
            mask_contours = get_mask_contours(mask)
            for contour in mask_contours:
                area_px = cv2.contourArea(contour)
                area_cm2 = round(area_px / media_ratios, 2)
                print(area_cm2)
                area_m2 = area_cm2 * 0.0001

    return area_m2
    
    
def detect_alvos_ratio(class_ids, boxes, masks, image):
    alvos = {"y1": [], "Ratio_px2_to_cm2": []}
    count_alvos = 0
    N = boxes.shape[0]
    for i in range(N):
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1 = boxes[i]
        if(class_ids[i] == 2):
            count_alvos = count_alvos + 1
            mask = masks[:, :, i]
            mask_countours = get_mask_contours(mask)
            for contour in mask_countours:
                rect = cv2.minAreaRect(contour)
                area_px = rect[1][0] * rect[1][1]          
                area_cm2 = round(area_px / 405.56, 2)
                alvos["y1"].append(y1)
                alvos["Ratio_px2_to_cm2"].append(area_cm2)
    
    return alvos, count_alvos

def calculate_area(model, image, IMAGE_DIR):
    data = []
    area_m2 = 0

    class_ids, boxes, masks, img = run_detection(model, image, IMAGE_DIR)
    
    if(len(class_ids) == 0):
        data.append(area_m2)
        data.append(image)
        return data
    
    alvos, count_alvos = detect_alvos_ratio(class_ids, boxes, masks, img)
    area_m2 = calculate_trunk_areas(class_ids, boxes, masks, img, alvos, count_alvos)
    area_m2 = round(area_m2, 2)

    data.append(area_m2)
    data.append(image)

    return data

def AddtoCSV(data):
    try:
        # open the file in the write mode
        f = open('C:/Users/andre/Desktop/Mask RCNN/ProjetoCortica/ModeloML/Areas.csv', 'a')

        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(data)

        # close the file
        f.close()

        return True
    except Exception:
            print("Falha ao escrever no ficheiro!")
            return False

if __name__== '__main__':
    config = SobreirosConfig()
    inference_config = InferenceConfig()
    model = recreate_model()

    IMAGE_DIR = sys.argv[0]

    #IMAGE_DIR = os.path.abspath(r"C:\Users\andre\Desktop\Mask RCNN\tese_janeiro") #Image directory 

    file_name = next(os.walk(IMAGE_DIR))[2]

    data = calculate_area(model, file_name, IMAGE_DIR)
    if(data[0] < 0.1):
        f = open("C:/Users/andre/Desktop/Mask RCNN/ProjetoCortica/mask/Mask_RCNN/Inference_data/inference.txt", "w")
        f.write("Error calculating cork oak trunk area!")
        f.close()
    else:
        #AddtoCSV(data)
        f = open("C:/Users/andre/Desktop/Mask RCNN/ProjetoCortica/mask/Mask_RCNN/Inference_data/inference.txt", "w")
        f.write(str(data[0]))
        f.close()


