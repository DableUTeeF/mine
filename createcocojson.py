import base64
import cv2
import numpy as np
from imageio import imread
import io
import os
import json
from PIL import Image
from retinanet.boxutils import create_csv_training_instances

if __name__ == '__main__':
    train_path = 'anns/ann.csv'
    classes_path = 'anns/c.csv'
    train_ints, labels, max_box_per_image = create_csv_training_instances(train_path, classes_path, False)

    csv_train_ann = open('anns/ann.csv').read().split('\n')[:-1]

    train_images = list(set([os.path.basename(element.split(',')[0]) for element in csv_train_ann]))

    ipath = '/media/palm/BiggerData/mine/new/i/'
    csv = []
    classid = []
    train_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID
    i = 0
    for instance in train_ints:
        i += 1
        print(i, end='\r')
        filename = instance['filename']
        pil_img = Image.open(filename)

        img_id = filename.replace(ipath, '')
        img_info = {
            'file_name': img_id,
            'height': pil_img.height,
            'width': pil_img.width,
            'id': img_id
        }
        train_json_dict['images'].append(img_info)
        for obj in instance['object']:
            x1 = obj['xmin']
            x2 = obj['xmax']
            y1 = obj['ymin']
            y2 = obj['ymax']
            label_id = obj['name']
            o_width = x2 - x1
            o_height = y2 - y1
            ann = {
                'area': o_width * o_height,
                'iscrowd': 0,
                'bbox': [x1, y1, o_width, o_height],
                'category_id': labels.index(label_id) + 1,
                'ignore': 0,
                'segmentation': []  # This script is not for segmentation
            }
            ann.update({'image_id': img_id, 'id': bnd_id})
            train_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
    print('bnd_id', bnd_id)
    for label_id, label in enumerate(labels):
        category_info = {'supercategory': 'none', 'id': label_id + 1, 'name': label}
        train_json_dict['categories'].append(category_info)

    with open('anns/train.json', 'w') as f:
        output_json = json.dumps(train_json_dict)
        f.write(output_json)
