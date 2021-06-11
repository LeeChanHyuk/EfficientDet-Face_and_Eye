import json
from collections import OrderedDict
import os
import sys
import cv2

"""This code is written by ChanHyukLee
Contact : dlcskgur3434@gmail.com
blog : https://leechanhyuk.github.io/"""

## Please put in this file in the project folder

# Define the dictionarys for making json files.
file_data = OrderedDict()
file_data["info"] = {'description':'Helen Dataset annotations', 'url':'http://www.ifp.illinois.edu/~vuongle2/helen/', 'Contributor':'ChanHyukLee','Data_created':'2021/06/09'}
caterories = []
caterories.append({'id': 1, 'name':'face'})
file_data["categories"] = caterories
image_annotations = []
image_informations = []
# Define the path
annotation_path = './datasets/Face/annotation_file'
train_image_path = './datasets/Face/trainimage/'

for txt in os.listdir(annotation_path):
    print(txt)
    f = open(os.path.join(annotation_path,txt))
    # Helen dataset's extension is jpg
    image_name_without_jpg = f.readline().strip('\n')
    image_name_without_jpg = image_name_without_jpg.replace("_","")
    image_name = image_name_without_jpg + '.jpg'
    img = cv2.imread(train_image_path+image_name)
    #cv2.imshow("image1",img)
    #cv2.waitKey(0)
    image_informations.append({'file_name':image_name, 'height':img.shape[0], 'width':img.shape[1], 'id':int(image_name_without_jpg)})
    # Each list is consisted by left-top-y, left-top-x, right-down-y, right-down-x
    face=[]
    for i in range(4):
        face.append(0)
    face[0] = img.shape[0] - 1
    face[1] = img.shape[1] - 1
    face[2] = 0
    face[3] = 0

    for i in range(194):
        x, y = f.readline().split(',')
        x = x.replace(" ", "")
        y = y.replace(" ", "")
        x_num = int(float(x))
        y_num = int(float(y))
        # cv2.circle(img, (x_num,y_num), 3, (0, 0, 255), 3)
        if face[0] > y_num:
            face[0] = y_num
        if face[1] > x_num:
            face[1] = x_num
        if face[2] < y_num:
            face[2] = y_num
        if face[3] < x_num:
            face[3] = x_num

    # Append this annotation to json.annotation part
    # 0 is face, 1 is left eye, 2 is right eye
    face_width = abs(face[3] - face[1]) + 1
    face_height = abs(face[0] - face[2]) + 1
    image_annotations.append({'id':int(image_name_without_jpg), 'bbox':[face[0], face[1], face_height, face_width], "image_id":int(image_name_without_jpg),'area':face_width*face_height, 'category_id': 1, 'iscrowd':0})
file_data['annotations'] = image_annotations
file_data['images'] = image_informations
print(json.dumps(file_data, ensure_ascii=False, indent='\t'))
with open('instances_Face.json', 'w', encoding='utf-8') as make_file:
    json.dump(file_data,make_file, ensure_ascii=False, indent='\t')

