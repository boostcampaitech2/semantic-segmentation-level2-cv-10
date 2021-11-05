# MMDetection
## Config
mmdetection download
```
git clone https://github.com/open-mmlab/mmdetection.git
```

- 처음 train dataset을 이용해 학습할 때
```
python ./tools/train.py swin_L384.py
```
## pseudo labeling
필수조건 : 처음 train dataset을 이용해 학습한 후, .pth file이 work_dir/swin_L384 디렉토리 내에 있음을 확인합니다.
### 실행 방법
- pseudo_labeling.py 를 실행합니다.
- merge_image 디렉토리에 mixed image와 이에 대한 json 파일이 있음을 확인합니다.
<br>

- train dataset과 mixed dataset을 이용해 학습할 때
```
python ./tools/train.py swin_L384V2.py
```

## TTA를 위한 코드 수정
일부 Aug가 `samples_per_gpu`가 1인 경우에만 작동을 하여 해당 코드를 수정합니다.
- mmdet/apis/train.py#L144
  ```diff
    val_dataloader = build_dataloader(
    val_dataset,
  - samples_per_gpu=val_samples_per_gpu,
  + samples_per_gpu= 1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)
  ```


## Model
- swin_L HTC
  - pretrained weight imageNet 22K, [swin_L weight github](https://github.com/microsoft/Swin-Transformer)
  - window_size, image size 384X384

## Backbone - Swin Transformer
backbone 모델로 swin transformer를 기반으로 하여 학습을 진행하였습니다.

## Neck and Head
  - 12 epoch
  - neck(in_channel = 192, 384, 768, 1536)
  - Loss
    - RPN
      - classification = CELoss
      - bbox regression = SmoothL1 Loss
    - RoI Head
      - classification = CELoss
      - bbox regression = SmoothL1 Loss
    - Mask Head : HTC에서 사용하는 Loss, semantic head에서 사용하는 Loss로 두개를 사용합니다
      - Mask_head = CELoss
      - semantic_head = CELoss
  - Optimizer 
    - AdamW
      -  lr=0.00015,
      -  weight_decay=0.05
     - scheduler
       - stepLR (8 epoch, 11 epoch)
       - warmup (ratio = 0.001)