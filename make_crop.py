import cv2
import os
from PIL import Image
import numpy as np
import skimage.measure as measure
import json
import cv2
import numpy as np
import pandas as pd
import pycocotools
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from pathlib import Path
import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from pandas import DataFrame
import copy
import tqdm
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


def make_json(json_name, make_dir):
    with open(f'/opt/ml/segmentation/input/data/{json_name}', 'r') as f:
        json_data = json.load(f)
    cnt = 0  
    make_dir = 'stratified_train'
    for idx, j in enumerate(json_data['images']):
        name = ''.join(['0' for i in range(4-len(str(cnt)))])+ str(cnt) + '.jpg'
        json_data['images'][idx]['file_name'] = 'img/' + name
        print('img/' + name)
        cnt += 1
    with open(f'/opt/ml/segmentation/input/{make_dir}/labeling_data.json', 'w', encoding = 'utf-8') as make_file:
        json.dump(json_data, make_file, ensure_ascii = False, indent = "\t")
    cnt = 0
    for idx, j in enumerate(json_data['images']):
        mask_image_path = 'mask/' + str(json_data['images'][idx]['file_name'].split('/')[1])
        name = ''.join(['0' for i in range(4-len(str(cnt)))])+ str(cnt) + '.png'
        if not os.path.exists(f'/opt/ml/segmentation/input/{make_dir}/mask/img'):
                os.makedirs(f'/opt/ml/segmentation/input/{make_dir}/mask/img')
        image = Image.open(f'/opt/ml/segmentation/input/{make_dir}/' + mask_image_path)
        image.save(f'/opt/ml/segmentation/input/{make_dir}/mask/img/' + name, 'png')
        cnt += 1


def make_imagedf(name, data, category_id, cnt):
    data['images'].append({})
    data['images'][-1]['width'] = 512
    data['images'][-1]['height'] = 512
    data['images'][-1]['file_name'] = str(category_id) + '/image/' + str(name) + '.png'
    data['images'][-1]['license'] = 0
    data['images'][-1]['flickr_url'] = None
    data['images'][-1]['coco_url'] = None
    data['images'][-1]['data_captured'] = None
    data['images'][-1]['id'] = int(cnt)
    
    return data


def make_annodf(name,
                data,
                category_id, 
                bbox,
                cnt):
    data['annotations'].append({})
    data['annotations'][-1]['image_id'] = int(cnt)
    data['annotations'][-1]['category_id'] = category_id
    data['annotations'][-1]['area'] = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    data['annotations'][-1]['bbox'] = list(map(int, bbox))
    image = Image.open("/opt/ml/segmentation/extract_image/" + str(category_id) + '/mask/' + str(name) + '.png')
    data['annotations'][-1]['segmentation'] = binary_mask_to_polygon(image)
    data['annotations'][-1]['iscrowd'] = 0
    data['annotations'][-1]['id'] = int(cnt)
    return data


def split_img_file(json_file, make_dir):
    class_colormap = pd.read_csv("/opt/ml/segmentation/baseline_code/class_dict.csv")
    color_map_array = class_colormap.iloc[:, 1:].values.astype(np.uint8)

    category_names = ["Backgroud", "General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    coco = COCO(f'/opt/ml/segmentation/input/data/{json_file}')

    imgIds = coco.getImgIds()
    annImgDir = '/opt/ml/segmentation/input/' + make_dir
    cnt = 0
    for imgId in imgIds:
        imgInfo = coco.loadImgs(imgId)[0]
        _file_name = f"{annImgDir}/{'img'}/"
        image = cv2.imread('/opt/ml/segmentation/input/data/' + imgInfo['file_name'], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        name = ''.join(['0' for i in range(4-len(str(cnt)))])+ str(cnt) + '.jpg'
        cv2.imwrite(_file_name + name, image)
        annIds = coco.getAnnIds(imgIds=imgInfo['id'])
        anns = coco.loadAnns(annIds)
        mask = np.zeros((imgInfo["height"], imgInfo["width"]), dtype=np.uint8)

        anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
        for ann in anns:
            annCatId = ann['category_id']
            assert annCatId != 0, 'category_id must be not 0.'
            mask[coco.annToMask(ann) == 1] = annCatId
        _file_name = f"{annImgDir}/{'mask'}/"
        _parent_path = Path(_file_name).parent
        if not _parent_path.exists():
            _parent_path.mkdir(parents=True)

        cv2.imwrite(_file_name+ name, mask)
        cnt += 1

def instanceMask2semtsegMask(output, n_class: int=10, img_size: list=[512, 512], score_thrs: float=0.9):
    dummy_image = np.zeros(img_size)
    Temp_mask_arr = np.zeros(img_size)
    seperate_img = []
    tmp_mask_arr = np.zeros((n_class+1, *img_size))
    for class_idx in range(n_class):
        _len = output[1][class_idx].__len__()
        if _len == 0:
            continue
        mask_arr = pycocotools.mask.decode(output[1][class_idx])
        for i in range(mask_arr.shape[-1]):
            cf_score = output[0][class_idx][i, -1]
            if cf_score >= score_thrs:
                maskBitmap = mask_arr[:, :, i]
                tmp_mask_arr[class_idx+1] += cf_score * maskBitmap
                seperate_img.append([np.argmax(np.subtract(tmp_mask_arr, Temp_mask_arr), axis=0), class_idx+1])
                Temp_mask_arr = copy.deepcopy(tmp_mask_arr)
    tmp_mask_arr = np.argmax(tmp_mask_arr, axis=0)
    #resized_tmp_mask_arr = A.Resize(256, 256)(image=dummy_image, mask=tmp_mask_arr)['mask']
    return tmp_mask_arr.flatten()

def instanceMask2semtsegMaskV2(output, n_class: int=10, img_size: list=[512, 512], score_thrs: float=0.9):
    dummy_image = np.zeros(img_size)
    Temp_mask_arr = np.zeros(img_size)
    seperate_img = []
    tmp_mask_arr = np.zeros((n_class+1, *img_size))
    for class_idx in range(n_class):
        _len = output[1][class_idx].__len__()
        if _len == 0:
            continue
        mask_arr = pycocotools.mask.decode(output[1][class_idx])
        for i in range(mask_arr.shape[-1]):
            cf_score = output[0][class_idx][i, -1]
            if cf_score >= score_thrs:
                maskBitmap = mask_arr[:, :, i]
                tmp_mask_arr[class_idx+1] += cf_score * maskBitmap
                seperate_img.append([np.argmax(np.subtract(tmp_mask_arr, Temp_mask_arr), axis=0), class_idx+1])
                Temp_mask_arr = copy.deepcopy(tmp_mask_arr)
    tmp_mask_arr = np.argmax(tmp_mask_arr, axis=0)
    #resized_tmp_mask_arr = A.Resize(256, 256)(image=dummy_image, mask=tmp_mask_arr)['mask']
    return tmp_mask_arr, seperate_img

def merge_image(insert_image, insert_mask, class_id, base_image=None, base_mask=None):
    """
    Args:
        insert_image: 
        insert_mask:
        class_id:
        base_image:
        base_mask:
    """
    tmp_img = np.zeros((512,512,3), dtype=np.uint8)
    if type(base_image) is type(None):
        base_image = tmp_img.copy()
    tmp_img[:,:,0] = np.where(insert_mask == class_id, insert_image[:,:,0], base_image[:,:,0])  # R
    tmp_img[:,:,1] = np.where(insert_mask == class_id, insert_image[:,:,1], base_image[:,:,1])  # G
    tmp_img[:,:,2] = np.where(insert_mask == class_id, insert_image[:,:,2], base_image[:,:,2])  # B
    if type(base_mask) is type(None):
        base_mask = np.zeros((512,512), dtype=np.uint8)
    tmp_mask = np.where(insert_mask == class_id, insert_mask, base_mask)  # mask
    return tmp_img, tmp_mask
    

def make_crop_image(name):
    # ------------------------------------------- #
    dataset_path = '/opt/ml/segmentation/input/data/'
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    target_category_arr = [1, 2, 3, 4, 5, 6, 7, 9, 10]
    thr = 0.8
    class_num = 10
    # ------------------------------------------- #
    # config file 들고오기
    cfg = Config.fromfile(f'./z_custom/models/{name}.py')

    epoch = 'latest'


    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.seed=2021
    cfg.gpu_ids = [1]
    cfg.work_dir = './work_dirs/' + name

    #cfg.model.roi_head.bbox_head.num_classes = 10
    print('inference setting finish')
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    print('dataloader finish')

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    # inference

    output = single_gpu_test(model, data_loader,show = True, out_dir = './z_patch', show_score_thr=thr) # output 계산
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    all_pred = []
    for i, out in enumerate(output):
        prediction = []
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[0][j]:
                if o[4] > thr:
                    prediction.append(np.concatenate(([j], list(map(round, o))), axis = 0))
        all_pred.append([image_info['file_name'], prediction])
    print('output_information collect finish')
    seg_output = []
    for i, out in enumerate(output):
        image_info = coco.loadImgs(coco.getImgIds(imgIds=img_ids[i]))[0]
        _, tmp_mask_arr = instanceMask2semtsegMaskV2(out, score_thrs=thr)
        seg_output.append(tmp_mask_arr)

    with open('/opt/ml/segmentation/input/train_all/labeling_data.json', 'r') as f:
        json_data = json.load(f)
    data = copy.deepcopy(json_data)
    data['images'] = []
    data['annotations'] = []
    account_cnt = 0
    passcnt = 0
    for target_category in target_category_arr:
        save_dir = f"/opt/ml/segmentation/extract_image/{target_category}"
        Path(f"{save_dir}/image/").mkdir(exist_ok=True, parents=True)
        Path(f"{save_dir}/mask/").mkdir(exist_ok=True, parents=True)
        for i, seg_mask in enumerate(seg_output):
            cnt = 0
            for mask_seperate, bbox_seperate in zip(seg_mask, all_pred[i][1]):
                if not np.isin(target_category, seg_mask):
                    continue
                c, xmin, ymin, xmax, ymax, score = bbox_seperate
                image_info = coco.loadImgs(coco.getImgIds(imgIds=img_ids[i]))[0]
                image = cv2.imread(dataset_path + all_pred[i][0], cv2.IMREAD_COLOR)
                catImage, catSegMask = merge_image(image, mask_seperate[0], target_category)
                _path = Path(image_info['file_name'])
                if np.where(catImage > 0, 1, 0).sum() < 10000:
                    passcnt += 1
                    continue
                image_name = f"{save_dir}/image/{_path.parent}_{_path.stem}_{cnt}.png"
                mask_name = f"{save_dir}/mask/{_path.parent}_{_path.stem}_{cnt}.mask.png"
                cv2.imwrite(image_name, catImage)
                cv2.imwrite(mask_name, catSegMask)
                data = make_imagedf(f"{_path.parent}_{_path.stem}_{cnt}", data, target_category, account_cnt + cnt)
                data = make_annodf(f"{_path.parent}_{_path.stem}_{cnt}.mask",
                                        data,
                                        target_category, 
                                        [xmin, ymin, xmax, ymax],
                                        account_cnt+ cnt)
                cnt += 1
            account_cnt += cnt
    with open('/opt/ml/segmentation/extract_image/labeling_data.json', 'w', encoding = 'utf-8') as make_file:
        json.dump(data, make_file, ensure_ascii = False, indent = "\t")
    print('finish.\nerror image = ' + str(passcnt))


if __name__ == '__main__':
    make_crop_image('swin_L384htc')