import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch
import webcolors
import skimage.measure as measure
import math
def valid_json(dataset):
    print('modify data dict...')
    ### int64자료형 제거하기
    for i in range(len(dataset['annotations'])):
        dataset['annotations'][i]['bbox'] = list(map(int, dataset['annotations'][i]['bbox']))
        dataset['annotations'][i]['image_id'] = int(dataset['annotations'][i]['image_id'])
        dataset['annotations'][i]['category_id'] = int(dataset['annotations'][i]['category_id'])
        dataset['annotations'][i]['area'] = int(dataset['annotations'][i]['area'])
        dataset['annotations'][i]['iscrowd'] = int(dataset['annotations'][i]['iscrowd'])
        dataset['annotations'][i]['id'] = int(dataset['annotations'][i]['id'])
    ### segmentation 잘못 나온거 제거하기
    data_df = pd.json_normalize(dataset['annotations'])
    image_index = set([])
    anno_index = set([])
    for i, v in enumerate(dataset['annotations']):
        if len(v['segmentation']) == 0:
            idx = data_df.iloc[i]['image_id']
            image_index.add(int(idx))
            anno_index.update(list(map(int, data_df.loc[data_df['image_id'] == data_df.iloc[i]['image_id']].index)))
    anno_index = list(anno_index)
    image_index = list(image_index)
    anno_index.sort()
    image_index.sort()

    new_dataset = {}
    new_dataset['info'] =  copy.deepcopy(dataset['info'])
    new_dataset['licenses'] = copy.deepcopy(dataset['licenses'])
    new_dataset['categories'] = copy.deepcopy(dataset['categories'])
    new_dataset['annotations'] = []
    new_dataset['images'] = []
    for i, v in tqdm(enumerate(dataset['annotations'])):
        if i in anno_index:
            continue
        else:
            new_dataset['annotations'].append(copy.deepcopy(v))

    for i, v in tqdm(enumerate(dataset['images'])):
        if i in image_index:
            continue
        else:
            new_dataset['images'].append(copy.deepcopy(v))
    print('done!')
    return new_dataset


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance = 0):
    polygons = [] 
    padded_binary_mask = np.pad(binary_mask, pad_width = 1, mode = 'constant', constant_values = 0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        contour = np.flip(contour, axis = 1)
        segmentation = contour.ravel().tolist()
        segmentation = [int(0) if i<0 else int(i) for i in segmentation]
        polygons.append(segmentation)
    return polygons

def merge_image(insert_image, insert_mask, class_id, base_image=None, base_mask=None):
    """
    Args:
        insert_image: 
        insert_mask:
        class_id:
        base_image:
        base_mask:
    """
    tmp_img = np.ones((512,512,3), dtype=np.uint8) * 255
    if type(base_image) is type(None):
        base_image = tmp_img.copy()
    tmp_img[:,:,0] = np.where(insert_mask == class_id, insert_image[:,:,0], base_image[:,:,0])  # R
    tmp_img[:,:,1] = np.where(insert_mask == class_id, insert_image[:,:,1], base_image[:,:,1])  # G
    tmp_img[:,:,2] = np.where(insert_mask == class_id, insert_image[:,:,2], base_image[:,:,2])  # B
    if type(base_mask) is type(None):
        base_mask = np.zeros((512,512), dtype=np.uint8)
    tmp_mask = np.where(insert_mask == class_id, insert_mask, base_mask)  # mask
    return tmp_img, tmp_mask

def make_imagedf(name, data, cnt):
    data['images'].append({})
    data['images'][-1]['width'] = 512
    data['images'][-1]['height'] = 512
    data['images'][-1]['file_name'] = name
    data['images'][-1]['license'] = 0
    data['images'][-1]['flickr_url'] = None
    data['images'][-1]['coco_url'] = None
    data['images'][-1]['data_captured'] = None
    data['images'][-1]['id'] = int(cnt)
    
    return data


def make_annodf(imgId,
                data,
                category_id, 
                bbox,
                segmentation,
                cnt):
    data['annotations'].append({})
    data['annotations'][-1]['image_id'] = imgId
    data['annotations'][-1]['category_id'] = category_id
    data['annotations'][-1]['area'] = int(bbox[2] * bbox[3])
    data['annotations'][-1]['bbox'] = bbox
    data['annotations'][-1]['segmentation'] = segmentation
    data['annotations'][-1]['iscrowd'] = 0
    data['annotations'][-1]['id'] = int(cnt)
    return data

def make_index(arr):
    rnd_idx = np.random.permutation([_ for _ in range(len(arr))])
    rnd_list = rnd_idx.tolist()
    return rnd_idx, rnd_list

def mediate_bbox(bbox, c, pseudo_masks):
    min_x = 512
    max_x = 0
    min_y = 512
    max_y = 0
    xmin, ymin, w, h = bbox
    xmin = math.floor(xmin)
    ymin = math.floor(ymin)
    xmax = min(511, math.ceil(xmin + w))
    ymax = min(511, math.ceil(ymin + h))
    #base_image = cv2.rectangle(base_image, (math.floor(xmin), math.floor(ymin)), (math.floor(xmax) , math.floor(ymax)), (255, 255, 255), 3)
    for jx in range(math.floor(ymin), math.ceil(ymax)+1):
        for ix in range(math.floor(xmin), math.ceil(xmax)+1):
            if str(pseudo_masks[jx][ix]) == str(c):
                if min_x > ix:
                    min_x = ix
                break
    for jx in range(math.floor(ymin), math.ceil(ymax)+1):
        for ix in range(math.ceil(xmax), math.floor(xmin)-1, -1):
            if str(pseudo_masks[jx][ix]) == str(c):
                if max_x < ix:
                    max_x = ix
                break
    for ix in range(math.floor(xmin), math.ceil(xmax)+1):
        for jx in range(math.floor(ymin), math.ceil(ymax)+1):
            if str(pseudo_masks[jx][ix]) == str(c):
                if min_y > jx:
                    min_y = jx
                break
    for ix in range(math.floor(xmin), math.ceil(xmax)+1):
        for jx in range(math.ceil(ymax), math.floor(ymin)-1, -1):
            if str(pseudo_masks[jx][ix]) == str(c):
                if max_y < jx:
                    max_y = jx
                break
    return min_x, min_y, max_x, max_y

def make_miximage():
    # merge original image and cropimage 
    # use original train data, use crop image by category
    # make_miximage need train image folder(train image, mask, .json)
    #                    crop image by category folder
    #                       crop image by category folder(10 category = 10 folder) = (train image, mask)
    #                       .json file
    #
    # output : a folder
    #           - mixed image
    #           - mask(mixed image)
    #           - labeling_data.json for mixed image
    class_colormap = pd.read_csv("/opt/ml/segmentation/baseline_code/class_dict.csv")
    color_map_array = class_colormap.iloc[:, 1:].values.astype(np.uint8)

    category_names = ["Backgroud", "General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    coco = COCO('/opt/ml/segmentation/input/train_all/labeling_data.json')

    imgIds = coco.getImgIds()
    ImgDir = '/opt/ml/segmentation/input/train_all/'

    with open('/opt/ml/segmentation/input/train_all/labeling_data.json', 'r') as f:
        json_data = json.load(f)
    data = copy.deepcopy(json_data)

    print('complete get original train data!')
    masks = []
    imgNames = []

    for imgId in tqdm(imgIds):
        imgInfo = coco.loadImgs(imgId)[0]

        annIds = coco.getAnnIds(imgIds=imgInfo['id'])
        anns = coco.loadAnns(annIds)

        mask = np.zeros((imgInfo["height"], imgInfo["width"]), dtype=np.uint8)

        anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=True)
        for ann in anns:
            annCatId = ann['category_id']
            assert annCatId != 0, 'category_id must be not 0.'
            mask[coco.annToMask(ann) == 1] = annCatId

        if np.any(np.isin([1, 2, 8, 6], mask)):
            imgNames.append(imgInfo['file_name'])
            masks.append(mask)

    data_df = pd.json_normalize(data['annotations'])

    with open('/opt/ml/segmentation/extract_image/labeling_data.json', 'r') as f:
        json_data = json.load(f)
    data_Plus = copy.deepcopy(json_data) #합성할 crop된 데이터 가져오기
    print('complete get crop train data!')
    new_data_df_img = pd.json_normalize(data_Plus['images'])
    new_data_df_anno = pd.json_normalize(data_Plus['annotations'])

    arr = [[] for i in range(11)]
    for idx, value in enumerate(data_Plus['annotations']):
        arridx = value['category_id']
        arr[arridx].append(idx)

    extractImgPath = '/opt/ml/segmentation/extract_image/'
    cat1ImgNames = [_.stem for _ in Path(f"{extractImgPath}1/image").iterdir()]
    cat4ImgNames = [_.stem for _ in Path(f"{extractImgPath}4/image").iterdir()]
    cat5ImgNames = [_.stem for _ in Path(f"{extractImgPath}5/image").iterdir()]
    cat6ImgNames = [_.stem for _ in Path(f"{extractImgPath}6/image").iterdir()]
    cat10ImgNames = [_.stem for _ in Path(f"{extractImgPath}10/image").iterdir()]


    targetCatId = 1
    targetcatid_arr = [1, 4, 5, 6, 10]
    print(f'start synthesize image, crop image_category = {targetcatid_arr}')
    imgcnt = 0
    annocnt = 0
    for targetCatId in targetcatid_arr:
        print(f'synthesize image {category_names[int(targetCatId)]}')
        targetImgNames = eval(f"cat{targetCatId}ImgNames")
        extractImgPath = '/opt/ml/segmentation/extract_image/'
        mergeImgPath = '/opt/ml/segmentation/merge_image/'
        cnt = 0
        rnd_idx, rnd_list = make_index(masks)
        save_dir = f"{mergeImgPath}"
        Path(f"{save_dir}/image/").mkdir(exist_ok=True, parents=True)
        Path(f"{save_dir}/mask/").mkdir(exist_ok=True, parents=True)

        for _ in range(10):  # 해당 category에 대해 반복을 몇번 할지 정합니다.
            for idx in tqdm(range(len(targetImgNames))):
                cnt += 1
                _file_name = targetImgNames[idx]
                if _file_name.startswith('.ipynb_checkpoints'):
                    cnt -= 1
                    continue
                image1 = cv2.imread(f"{extractImgPath}{targetCatId}/image/{_file_name}.png")
                mask1 = cv2.imread(f"{extractImgPath}{targetCatId}/mask/{_file_name}.mask.png", cv2.IMREAD_GRAYSCALE)
                newimgidx = int(new_data_df_img.loc[new_data_df_img['file_name'] == f'{targetCatId}/image/{_file_name}.png']['id']) #이미지에 해당하는 id를 갖고오기
                newimganno= new_data_df_anno.iloc[newimgidx]
                newimgbbox = new_data_df_anno.iloc[newimgidx]['bbox']

                rnd_idx = rnd_list.pop()
                base_image = cv2.imread(f"{ImgDir}{imgNames[rnd_idx]}")
                base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                base_mask = cv2.imread(f"{ImgDir}/mask/{imgNames[rnd_idx]}", cv2.IMREAD_GRAYSCALE)
                anno_idx = int(imgNames[rnd_idx].split('/')[1].split('.')[0])
                tmp_img, tmp_mask = merge_image(image1, mask1, targetCatId, base_image = base_image, base_mask = base_mask)

                pseudo_masks = np.where(mask1 != 0, 100, 0) #가짜 mask, 합성할 사진을 배경으로 변환할 mask를 생성한다.
                t_img, t_mask = merge_image(image1, pseudo_masks, 100, base_image = base_image, base_mask = base_mask)
                pseudo_masks = copy.deepcopy(tmp_mask) #합성하려는 이미지를 가져와서
                pseudo_masks[np.where(t_mask == 100)] = 0 #tmask가 100인 부분을 0으로 만들어 배경으로 만든다.

                # bbox for json
                new_bbox = []
                for bbox, c in zip(list(data_df.loc[data_df['image_id'] == anno_idx]['bbox']), list(data_df.loc[data_df['image_id'] == anno_idx]['category_id'])):
                    min_x, min_y, max_x, max_y = mediate_bbox(bbox, c, pseudo_masks)
                    #tmp_mask = cv2.rectangle(tmp_mask, (min_x, min_y), (max_x, max_y), (255), 1)
                    if min_x == 512 and min_y == 512 and max_x == 0 and max_y == 0: # bbox가 잡히지 않을 때
                        pass
                    else:
                        new_bbox.append([list(map(int, [min_x, min_y, max_x - min_x, max_y - min_y])), int(c)])
                    #tmp_mask = cv2.rectangle(tmp_mask, (x, y), (x2, y2), (255), 1)

                #mask polygon for json
                new_mask = []
                idx_seg = list(data_df.loc[data_df['image_id'] == anno_idx]['image_id'])
                imgInfo = coco.loadImgs(idx_seg)[0]
                annIds = coco.getAnnIds(imgIds=imgInfo['id'])
                anns = coco.loadAnns(annIds)
                for ann, c in zip(anns, list(data_df.loc[data_df['image_id'] == anno_idx]['category_id'])):
                    mask = np.zeros((imgInfo["height"], imgInfo["width"]), dtype=np.uint8)
                    mask[coco.annToMask(ann) == 1] = c
                    smask = np.where(mask == pseudo_masks, mask, 0)
                    smask_pg = binary_mask_to_polygon(smask)
                    new_mask.append(smask_pg)      
                    
                if len(new_mask) != len(new_bbox):
                    continue
                #bbox가 가려지는 image는 학습 데이터로 사용하지 않는다.

                cv2.imwrite(f"{save_dir}/image/synth{targetCatId}_{cnt:04d}.png", tmp_img)
                cv2.imwrite(f"{save_dir}/mask/synth{targetCatId}_{cnt:04d}.png", tmp_mask.astype(np.uint8))
                x, y, x2, y2 = newimgbbox
                newimgbbox = [x, y, x2 - x, y2 - y]
                data = make_imagedf(f"image/synth{targetCatId}_{cnt:04d}.png", data, imgcnt)
                for i in range(len(new_bbox)): 
                    data = make_annodf(imgcnt,
                                        data,
                                        new_bbox[i][1], 
                                        new_bbox[i][0],
                                        new_mask[i],
                                        annocnt)
                    annocnt += 1
                data = make_annodf(imgcnt,
                            data,
                            newimganno['category_id'], 
                            newimgbbox,
                            newimganno['segmentation'],
                            annocnt)
                annocnt += 1
                imgcnt += 1

    data = valid_json(data)
    with open('/opt/ml/segmentation/merge_image/labeling_data.json', 'w', encoding = 'utf-8') as make_file:
        json.dump(data, make_file, ensure_ascii = False, indent = "\t")