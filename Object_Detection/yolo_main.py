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

from preprocess import import_data, pull_labels, preprocess_input
from preprocess import decode_netout, do_nms, draw_boxes, detect_image, detect_video


def main():
    
    model_path = '/Users/andrewkent/Library/Mobile Documents/com~apple~CloudDocs/Projects/Deeplearning/Object_Detection/data/yolo_weights.h5'
    image_path = '/Users/andrewkent/Library/Mobile Documents/com~apple~CloudDocs/Projects/Deeplearning/Object_Detection/data/image.jpg'
    anchors = [[[116,90], [156,198], [373,326]], [[30,61], [62,45], [59,119]], [[10,13], [16,30], [33,23]]]
    import_data()
    labels = pull_labels()

    image_pil = Image.open(image_path)
    image_w, image_h = image_pil.size
    print("The type of the saved image is {}".format(type(image_pil)))
    plt.imshow(image_pil)
    plt.show()
    
    net_h, net_w = 416, 416
    new_image = preprocess_input(image_pil, net_h, net_w)




    net_h, net_w = 416, 416
    new_image = preprocess_input(image_pil, net_h, net_w)
    plt.imshow(new_image[0])
    plt.show()
    
    darknet = tf.keras.models.load_model(model_path)
    obj_thresh = 0.4
    nms_thresh = 0.45
    
    yolo_outputs = darknet.predict(new_image)
    
    
    print(len(yolo_outputs))
    print(yolo_outputs[0].shape)
    print(yolo_outputs[1].shape)
    print(yolo_outputs[2].shape)
    
    # Decode the output of the network
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
    
    # Suppress non-maximal boxes
    boxes = do_nms(boxes, nms_thresh, obj_thresh)
    
    # Draw bounding boxes on the image using labels
    image_detect = draw_boxes(image_pil, boxes, labels) 
    
    plt.figure(figsize=(12,12))
    plt.imshow(image_detect)
    plt.show()
    
    # Decode
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
    plt.figure(figsize=(10,10))
    plt.imshow(draw_boxes(image_pil, boxes, labels))
    plt.show()
    # NMS
    boxes = do_nms(boxes, nms_thresh, obj_thresh)
    plt.figure(figsize=(10,10))
    plt.imshow(draw_boxes(image_pil, boxes, labels))
    plt.show()
    
    
    
    
    
    # Lower objectness threshold -> more object predictions are accepted (more predictions)
    plt.figure(figsize=(12,12))
    plt.imshow(detect_image(image_pil, obj_thresh = 0.2, nms_thresh = 0.45))
    plt.show()
    
    # Higher nms threshold -> Allowing more overlapping bounding boxes (more predictions)
    plt.figure(figsize=(12,12))
    plt.imshow(detect_image(image_pil, obj_thresh = 0.4, nms_thresh = 0.8))
    plt.show()
    
    
        
    video_path = '/Users/andrewkent/Library/Mobile Documents/com~apple~CloudDocs/Projects/Deeplearning/Object_Detection/data/video1.mp4'
    output_path = '/Users/andrewkent/Library/Mobile Documents/com~apple~CloudDocs/Projects/Deeplearning/Object_Detection/data/video1_detected.mp4'
    detect_video(video_path, output_path)
    
    
    
if __name__ == '__main__':
	main()
    
    





