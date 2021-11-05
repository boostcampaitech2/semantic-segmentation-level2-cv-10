from pycocotools.coco import COCO
import cv2
import numpy as np
import os
from tqdm import tqdm

root_data_path = '/opt/ml/segmentation/input/data'
test_json_dir = '/opt/ml/segmentation/input/data/test.json'
train_json_dir = '/opt/ml/segmentation/input/data/train.json'
train_all_json_dir = '/opt/ml/segmentation/input/data/train_all.json'
val_json_dir = '/opt/ml/segmentation/input/data/val.json'
dir_list = [root_data_path + '/test', root_data_path + '/test/img',
            root_data_path + '/train', root_data_path + '/train/img', root_data_path + '/train/mask',
            root_data_path + '/val', root_data_path + '/val/img', root_data_path + '/val/mask']
for dir in dir_list:
    os.makedirs(dir, exist_ok=True)

test_img_dir = dir_list[1]
train_img_dir = dir_list[3]
train_mask_dir = dir_list[4]
val_img_dir = dir_list[6]
val_mask_dir = dir_list[7]

def make_image_and_mask(json_dir:str) -> None:
    coco = COCO(json_dir)

    image_ids = coco.getImgIds()

    for i in tqdm(range(len(image_ids))):
        image_infos = coco.loadImgs(image_ids[i])[0]   
        images = cv2.imread(os.path.join(root_data_path, image_infos['file_name']))
        ann_ids = coco.getAnnIds(imgIds=image_infos['id'])
        anns = coco.loadAnns(ann_ids)    
        masks = np.zeros((image_infos["height"], image_infos["width"]))
        anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=True)
        if 'test' not in json_dir:
            for i in range(len(anns)):
                pixel_value = anns[i]['category_id']
                masks[coco.annToMask(anns[i]) == 1] = pixel_value
        gray_masks = masks.astype(np.int8) 
        
        img_path = None
        mask_path = None
        if 'test' in json_dir:
            img_path = test_img_dir
        elif 'train' in json_dir:
            img_path = train_img_dir
            mask_path = train_mask_dir
        elif 'val' in json_dir:
            img_path = val_img_dir
            mask_path = val_mask_dir
        num = '%04d'%(image_infos['id']) + '.png'
        img_path = os.path.join(img_path, num)    
        cv2.imwrite(img_path, images)
        if 'test' not in json_dir:
            mask_path = os.path.join(mask_path, num)
            cv2.imwrite(mask_path, gray_masks)


make_image_and_mask(test_json_dir)
make_image_and_mask(train_json_dir)
#make_image_and_mask(train_all_json_dir)
make_image_and_mask(val_json_dir)