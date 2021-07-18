import cv2
import os
import json

image_data_path = './datasets/Face/trainimage'
annotation_data_path = 'datasets/Face/annotations/instances_trainimage.json'

with open(annotation_data_path, 'r') as f:
    json_data = json.load(f)
    annotations = json_data['annotations']
    for annotation in annotations:
        image_name = annotation['id']
        bbox = annotation['bbox']
        path = os.path.join(image_data_path, str(image_name)+'.jpg')
        img = cv2.imread(path)
        cv2.rectangle(img,(bbox[1],bbox[0]), (bbox[1]+bbox[3],bbox[0]+bbox[2]), (0,0,255), 3)
        cv2.imshow("img", img)
        cv2.waitKey(1)
        print("Aspect ratio = " + str((bbox[2]*bbox[3]/(img.shape[0]*img.shape[1]))))
        print("Area = "+str(bbox[2]*bbox[3]))



