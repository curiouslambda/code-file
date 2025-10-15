import json

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import os
import random
import shutil
import numpy as np
from PIL import Image

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

catego = ['Breakage', 'Crushed', 'Scratched', 'Separated']
img_origin = '../data/Dataset/1.원천데이터/damage/'
json_origin = '../data/Dataset/2.라벨링데이터/damage/'


for c in catego:
        part_json = '../data/datainfo/damage_' + c + '_train.json'
        test_json = '../data/datainfo/damage_' + c + '_test.json'
        val_json  = '../data/datainfo/damage_' + c + '_val.json'

        with open(part_json, 'r') as f:
                trpart = json.load(f)

        with open(test_json, 'r') as f:
                tepart = json.load(f)

        with open(val_json, 'r') as f:
                vapart = json.load(f)        

        coco = COCO(part_json)
        coco2 = COCO(test_json)
        coco3 = COCO(val_json)

        img_ids = coco.getImgIds()
        img_id2 = coco2.getImgIds()
        img_id3 = coco3.getImgIds()

        cate = {c : []}
        cate_color =  (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))

        for i in range(0, len(trpart['images'])):
                id = int(img_ids[i])
                image_name = trpart['images'][i]['file_name']
                if not os.path.exists('../data/Dataset/damage_type/' + c + '_img_train/') and not os.path.exists('../data/Dataset/damage_type/' + c + '_json_train/'):
                        os.makedirs('../data/Dataset/damage_type/' + c + '_img_train/')
                        os.makedirs('../data/Dataset/damage_type/' + c + '_json_train/')
                shutil.copyfile(img_origin + image_name, '../data/Dataset/damage_type/' + c + '_img_train/' + image_name)
                shutil.copyfile(json_origin + image_name.replace('.jpg', '.json'), '../data/Dataset/damage_type/' + c + '_json_train/' + image_name.replace('.jpg', '.json'))
                image_infos = coco.loadImgs(id)[0]
                width = image_infos['width']
                height = image_infos['height']

                for j in range(0, len(trpart['annotations'])):
                        if trpart['annotations'][j]['image_id'] == id:
                                if len(trpart['annotations'][j]['segmentation'][0]) > 1:
                                        for i in trpart['annotations'][j]['segmentation'][0]:
                                                if trpart['annotations'][j]['damage'] in cate.keys():
                                                        cate[trpart['annotations'][j]['damage']].append(i)
                                else:
                                        if trpart['annotations'][j]['damage'] in cate.keys():
                                                cate[trpart['annotations'][j]['damage']].append(trpart['annotations'][j]['segmentation'][0][0])

                mask = np.zeros((height,width, 3),dtype = np.uint8)
                if not os.path.exists('../data/Dataset/damage_mask/' + c + '_train_mask/'):
                        os.makedirs('../data/Dataset/damage_mask/' + c + '_train_mask/')
                for j in range(0, len(cate[c])):
                        mask = cv2.fillPoly(mask, [np.array(cate[c][j][0])], cate_color)
                cv2.imwrite(f'../data/Dataset/damage_mask/' + c + '_train_mask/' + image_name.replace('.jpg', '') + '.png', mask)
                cate[c] = []
        
        for i in range(0, len(tepart['images'])):
                id = int(img_id2[i])
                image_name = tepart['images'][i]['file_name']
                if not os.path.exists('../data/Dataset/damage_type/' + c + '_img_test/') and not os.path.exists('../data/Dataset/damage_type/' + c + '_json_test/'):
                        os.makedirs('../data/Dataset/damage_type/' + c + '_img_test/')
                        os.makedirs('../data/Dataset/damage_type/' + c + '_json_test/')
                shutil.copyfile(img_origin + image_name, '../data/Dataset/damage_type/' + c + '_img_test/' + image_name)
                shutil.copyfile(json_origin + image_name.replace('.jpg', '.json'), '../data/Dataset/damage_type/' + c + '_json_test/' + image_name.replace('.jpg', '.json'))
                image_infos = coco2.loadImgs(id)[0]
                width = image_infos['width']
                height = image_infos['height']

                for j in range(0, len(tepart['annotations'])):
                        if tepart['annotations'][j]['image_id'] == id:
                                if len(tepart['annotations'][j]['segmentation'][0]) > 1:
                                        for i in tepart['annotations'][j]['segmentation'][0]:
                                                if tepart['annotations'][j]['damage'] in cate.keys():
                                                        cate[tepart['annotations'][j]['damage']].append(i)
                                else:
                                        if tepart['annotations'][j]['damage'] in cate.keys():
                                                cate[tepart['annotations'][j]['damage']].append(tepart['annotations'][j]['segmentation'][0][0])

                mask = np.zeros((height,width, 3),dtype = np.uint8)
                if not os.path.exists('../data/Dataset/damage_mask/' + c + '_test_mask/'):
                        os.makedirs('../data/Dataset/damage_mask/' + c + '_test_mask/')
                for j in range(0, len(cate[c])):
                        mask = cv2.fillPoly(mask, [np.array(cate[c][j][0])], cate_color)
                cv2.imwrite(f'../data/Dataset/damage_mask/' + c + '_test_mask/' + image_name.replace('.jpg', '') + '.png', mask)
                cate[c] = []

        for i in range(0, len(vapart['images'])):
                id = int(img_id3[i])
                image_name = vapart['images'][i]['file_name']
                if not os.path.exists('../data/Dataset/damage_type/' + c + '_img_val/') and not os.path.exists('../data/Dataset/damage_type/' + c + '_json_val/'):
                        os.makedirs('../data/Dataset/damage_type/' + c + '_img_val/')
                        os.makedirs('../data/Dataset/damage_type/' + c + '_json_val/')
                shutil.copyfile(img_origin + image_name, '../data/Dataset/damage_type/' + c + '_img_val/' + image_name)
                shutil.copyfile(json_origin + image_name.replace('.jpg', '.json'), '../data/Dataset/damage_type/' + c + '_json_val/' + image_name.replace('.jpg', '.json'))
                image_infos = coco3.loadImgs(id)[0]
                width = image_infos['width']
                height = image_infos['height']

                for j in range(0, len(vapart['annotations'])):
                        if vapart['annotations'][j]['image_id'] == id:
                                if len(vapart['annotations'][j]['segmentation'][0]) > 1:
                                        for i in vapart['annotations'][j]['segmentation'][0]:
                                                if vapart['annotations'][j]['damage'] in cate.keys():
                                                        cate[vapart['annotations'][j]['damage']].append(i)
                                else:
                                        if vapart['annotations'][j]['damage'] in cate.keys():
                                                cate[vapart['annotations'][j]['damage']].append(vapart['annotations'][j]['segmentation'][0][0])

                mask = np.zeros((height,width, 3),dtype = np.uint8)
                if not os.path.exists('../data/Dataset/damage_mask/' + c + '_val_mask/'):
                        os.makedirs('../data/Dataset/damage_mask/' + c + '_val_mask/')
                for j in range(0, len(cate[c])):
                        mask = cv2.fillPoly(mask, [np.array(cate[c][j][0])], cate_color)
                cv2.imwrite(f'../data/Dataset/damage_mask/' + c + '_val_mask/' + image_name.replace('.jpg', '') + '.png', mask)
                cate[c] = []