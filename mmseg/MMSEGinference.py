#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import mmcv

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import pandas as pd
import numpy as np
import json


# In[10]:


# config file 들고오기
##cfg = Config.fromfile('./configs/_custom_/models/hr48_512x512.py')
cfg = Config.fromfile('/opt/ml/mmsegmentation/work_dirs/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K.py')

root='/opt/ml/segmentation/input/mmseg/test/'

epoch = 'latest'

# dataset config 수정
cfg.data.test.img_dir = root
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 4

cfg.work_dir = '/opt/ml/mmsegmentation/work_dirs/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K'

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')


# In[11]:


dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)


# In[12]:


model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])


# In[13]:


output = single_gpu_test(model, data_loader)


# In[15]:


# sample_submisson.csv 열기
submission = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)
json_dir = os.path.join("/opt/ml/segmentation/input/data/test.json")
with open(json_dir, "r", encoding="utf8") as outfile:
    datas = json.load(outfile)

input_size = 512
output_size = 256
bin_size = input_size // output_size
		
# PredictionString 대입
for image_id, predict in enumerate(output):
    image_id = datas["images"][image_id]
    file_name = image_id["file_name"]
    
    temp_mask = []
    predict = predict.reshape(1, 512, 512)
    mask = predict.reshape((1, output_size, bin_size, output_size, bin_size)).max(4).max(2) # resize to 256*256
    temp_mask.append(mask)
    oms = np.array(temp_mask)
    oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)

    string = oms.flatten()

    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=False)


# In[16]:


# reference : https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
input_size = 512
output_size = 256
bin_size = input_size // output_size

for image_id, predict in enumerate(output):
  temp_mask = []
  predict = predict.reshape(1, 512, 512)
  print(predict.shape)
  mask = predict.reshape((1, output_size, bin_size, output_size, bin_size)).max(4).max(2)
  print(mask.shape)
  temp_mask.append(mask)
  
  oms = np.array(temp_mask)
  oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)
  print(oms.shape)
  predict = oms
  print(predict.shape)


# In[ ]:




