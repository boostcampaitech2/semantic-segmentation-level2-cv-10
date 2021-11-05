# MMSegmentatnion

## Model

## UperNet
backbone 모델로 swin-b 기반하여 학습을 진행하였습니다.

실행 방법
```bash
python mmseg_train.py -c mmdet_config/models/swin/swin-t_img-768_AdamW-24e_pseudo_labeling.py
```