import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import pycocotools
import numpy as np
import copy
from tqdm import tqdm
import albumentations as A
def instanceMask2semtsegMask(output, n_class: int=10, img_size: list=[512, 512], score_thrs: float=0.2):
    dummy_image = np.zeros(img_size)
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
        
    tmp_mask_arr = np.argmax(tmp_mask_arr, axis=0)
    resized_tmp_mask_arr = A.Resize(256, 256)(image=dummy_image, mask=tmp_mask_arr)['mask']
    return resized_tmp_mask_arr.flatten()

if __name__ == '__main__':
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg = Config.fromfile('./z_custom/models/swin_L384htc.py')

    epoch = 'latest'


    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.seed=2021
    cfg.gpu_ids = [1]
    cfg.work_dir = './work_dirs/swin_L384htc'

    #cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)


    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    #output = single_gpu_test(model, data_loader,show = True, out_dir = './z_patch', show_score_thr=0.8) # output 계산

    output = single_gpu_test(model, data_loader, show_score_thr=0.05) 
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in tqdm(enumerate(output)):
        image_info = coco.loadImgs(coco.getImgIds(imgIds=img_ids[i]))[0]
        prediction_strings.append(' '.join(str(e) for e in instanceMask2semtsegMask(out, score_thrs=0.5).tolist()))
        file_names.append(image_info['file_name'])
        

    submission = pd.DataFrame()
    submission['image_id'] = file_names
    submission['PredictionString'] = prediction_strings
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)
    submission.head()