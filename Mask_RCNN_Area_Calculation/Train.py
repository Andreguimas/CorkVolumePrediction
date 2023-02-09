import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import imgaug.augmenters as iaa


get_ipython().system('pip install --upgrade h5py==2.10.0')
import sys
sys.path.append("./mrcnn")
print("------------")
print(sys.path)
print("------------")
from m_rcnn import *
get_ipython().run_line_magic('matplotlib', 'inline')


# Extract Images
#images_path = os.path.abspath(r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica/dataset.zip")
annotations_train_path = os.path.abspath(r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica\Treino\annotations_train.json") #dataset available at https://www.kaggle.com/datasets/andreguim/cork-oak-segmentation
annotations_val_path = os.path.abspath(r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica\Treino\annotations_val.json") #dataset available at https://www.kaggle.com/datasets/andreguim/cork-oak-segmentation

#dataset_train =  = load_image_dataset(annotations_train_path, r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica\Treino\train", "train")
dataset_train = load_image_dataset_new(annotations_train_path, r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica\Treino\train") #dataset available at https://www.kaggle.com/datasets/andreguim/cork-oak-segmentation
dataset_val = load_image_dataset_new(annotations_val_path, r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica\Treino\val") #dataset available at https://www.kaggle.com/datasets/andreguim/cork-oak-segmentation
class_number = dataset_train.count_classes()
print(dataset_train)
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))

#display_image_samples(dataset_train)
#display_image_samples(dataset_val)

# Root directory of the project
ROOT_DIR = os.path.abspath(r"C:\Users\andre\Desktop\Mask RCNN\ProjetoCortica\results") #path to save the results file

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, r"logs\sobreiros20220209T1444\mask_rcnn_sobreiros_0031.h5") #use of a pre-trained model from another experiment conducted with fewer images
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class SobreirosConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
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
    TRAIN_ROIS_PER_IMAGE = 70

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
config = SobreirosConfig()
config.display()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True)
#model.load_weights(COCO_MODEL_PATH, by_name=True,
                      # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                      #          "mrcnn_bbox", "mrcnn_mask"])

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=100, 
            layers='heads',
            augmentation = iaa.Sometimes(5/6,iaa.OneOf(
                [  
                    iaa.Affine(rotate=(-10, 10)),
                    iaa.Affine(rotate=(-5, 5)),
                    iaa.Fliplr(1),
                    iaa.Affine(translate_percent=0.1),
                    iaa.WithBrightnessChannels(iaa.Add((-30, 30)))
                ]
            )))



