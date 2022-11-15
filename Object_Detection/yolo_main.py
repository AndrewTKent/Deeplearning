#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
############################

Created on Sun Jul 10 12:21:49 2022

Author: Andrew Kent

Email: andrew_kent@brown.edu

############################
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image

from preprocess import import_data, pull_labels, preprocess_input, pull_anchors, time
from preprocess import decode_netout, do_nms, draw_boxes, detect_image, detect_video


def main():
    
    # Print Start Time
    start_time = time('start')   
    
    # What you would like the model to print out
    print_original_img = False
    print_reshaped_img = False
    print_bounding_boxes = False
    print_dif_threshholds = False
    create_bounded_video = True
    
    # The path names of the different files needed for this project
    data_path = '/Users/andrewkent/Library/Mobile Documents/com~apple~CloudDocs/Projects/Deeplearning/Object_Detection/data'
    model_path = data_path + '/yolo_weights.h5'
    image_path = data_path + '/image.jpg'
    video_path = data_path + '/video1.mp4'
    video_path = data_path + '/walk_through_nyc.mp4'
    output_path = data_path + '/walk_through_nyc_detected.mp4'
    
    video_path = data_path + '/nextera.mp4'
    output_path = data_path + '/nextera_detected.mp4'
    
    # Import Data 
    import_data(True)
    anchors = pull_anchors()
    labels = pull_labels()
    
    # Parameters for Reshaping
    net_h, net_w = 416, 416
    
    # Parameters for Darknet
    obj_thresh = 0.4
    nms_thresh = 0.45

    image_pil = Image.open(image_path)
    new_image = preprocess_input(image_pil, net_h, net_w)
    image_w, image_h = image_pil.size

    if print_original_img == True:
        print("The type of the saved image is {}".format(type(image_pil)))
        plt.imshow(image_pil)
        plt.show()
    
    if print_reshaped_img == True:
        plt.imshow(new_image[0])
        plt.show()
    
    # Instantiating our YOLO model
    darknet = tf.keras.models.load_model(model_path)
    yolo_outputs = darknet.predict(new_image)
    
    # Decode the output of the network
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
    
    # Suppress non-maximal boxes
    boxes = do_nms(boxes, nms_thresh, obj_thresh)
    
    # Draw bounding boxes on the image using labels
    image_detect = draw_boxes(image_pil, boxes, labels) 
    
    # Decode
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
    
    # NMS
    boxes = do_nms(boxes, nms_thresh, obj_thresh)
    
    
    if print_bounding_boxes == True:
        plt.figure(figsize=(12,12))
        plt.imshow(image_detect)
        plt.show()
        
    if print_dif_threshholds == True:    
        # Lower objectness threshold -> more object predictions are accepted (more predictions)
        plt.figure(figsize=(12,12))  
        plt.imshow(detect_image(image_pil, 0.2, 0.45, darknet, net_h, net_w, anchors, labels))
        plt.show()
        
        # Higher nms threshold -> Allowing more overlapping bounding boxes (more predictions)
        plt.figure(figsize=(12,12))
        plt.imshow(detect_image(image_pil, 0.4, 0.8, darknet, net_h, net_w, anchors, labels))
        plt.show()
        
    if create_bounded_video == True:
        detect_video(video_path, output_path, obj_thresh, nms_thresh, darknet, net_h, net_w, anchors, labels)
  
    # Print End and Run Time
    end_time = time('end')
    time(end_time - start_time)
    
    
    
if __name__ == '__main__':
	main()
    
    





