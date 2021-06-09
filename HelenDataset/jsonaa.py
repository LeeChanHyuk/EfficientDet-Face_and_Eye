import json
from collections import OrderedDict
import os
import sys
import cv2

"""This code is written by ChanHyukLee
Contact : dlcskgur3434@gmail.com
blog : https://leechanhyuk.github.io/"""

# Define the dictionarys for making json files.
file_data = OrderedDict()
file_data["info"] = {'description':'Helen Dataset annotations', 'url':'http://www.ifp.illinois.edu/~vuongle2/helen/', 'Contributor':'ChanHyukLee','Data_created':'2021/06/09'}
caterories = []
caterories.append({'id': 1, 'name':'face'})
caterories.append({'id': 2, 'name':'left_eye'})
caterories.append({'io': 3, 'name':'right_eye'})
file_data["categories"] = caterories
image_annotations = []
image_informations = []
# Define the path
annotation_path = './annotation/annotation'
train_image_path = './train_image/'

for txt in os.listdir(annotation_path):
    print(txt)
    f = open(os.path.join(annotation_path,txt))
    # Helen dataset's extension is jpg
    image_name_without_jpg = f.readline().strip('\n')
    image_name = image_name_without_jpg + '.jpg'
    img = cv2.imread(train_image_path+image_name)
    image_informations.append({'file_name':image_name, 'height':str(img.shape[0]), 'width':str(img.shape[1]), 'id':image_name_without_jpg})
    # Each list is consisted by left-top-y, left-top-x, right-down-y, right-down-x
    left_eye=[]
    right_eye=[]
    face=[]
    for i in range(4):
        left_eye.append(0)
        right_eye.append(0)
        face.append(0)

    for i in range(194):
        x, y = f.readline().split(',')
        x = x.replace(" ", "")
        y = y.replace(" ", "")
        x_num = int(float(x))
        y_num = int(float(y))
        if i == 0: # Left chin
            face[1] = x_num
        elif i == 21: # Bottom chin
            face[2] = y_num
        elif i == 40: # right chin
            face[3] = x_num
        elif i == 159 or i == 178: # Top of both eyebrow
            if y_num > face[0]: # If the eyebrow == bigger than face top coordinate
                face[0] = y_num
        elif i == 114: # left of right eye
            right_eye[1] = x_num
        elif i == 125: # right of right eye
            right_eye[3] = x_num
        elif i == 120: # top of right eye
            right_eye[0] = y_num
        elif i == 128: # bottom of right eye
            right_eye[2] = y_num
        elif i == 134: # right of left eye
            left_eye[3] = x_num
        elif i == 139: # top of left eye
            left_eye[0] = y_num
        elif i == 145: # left of left eye
            left_eye[1] = x_num
        elif i == 149: # bottom of left eye
            left_eye[2] = y_num

    # Append this annotation to json.annotation part
    # 0 is face, 1 is left eye, 2 is right eye
    face_width = face[3] - face[1] + 1
    face_height = face[0] - face[2] + 1
    left_eye_width = left_eye[3] - left_eye[1] + 1
    left_eye_height = left_eye[0] - left_eye[2] + 1
    right_eye_width = right_eye[3] - right_eye[1] + 1
    right_eye_height = right_eye[0] - right_eye[2] + 1
    image_annotations.append({'id':image_name_without_jpg, 'bbox':[face[0], face[1], face_height, face_width], "image_id":image_name_without_jpg,'area':face_width*face_height, 'category_id': 1})
    image_annotations.append({'id':image_name_without_jpg, 'bbox':[left_eye[0], left_eye[1], left_eye_height, left_eye_width], "image_id":image_name_without_jpg,'area':left_eye_width*left_eye_height, 'category_id': 2})
    image_annotations.append({'id':image_name_without_jpg, 'bbox':[right_eye[0], right_eye[1], right_eye_height, right_eye_width], "image_id":image_name_without_jpg,'area':right_eye_width*right_eye_height, 'category_id': 3})
file_data['annotations'] = image_annotations
file_data['images'] = image_informations
print(json.dumps(file_data, ensure_ascii=False, indent='\t'))
with open('instances_train2017.json', 'w', encoding='utf-8') as make_file:
    json.dump(file_data,make_file, ensure_ascii=False, indent='\t')

