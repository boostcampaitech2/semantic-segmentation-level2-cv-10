# dataset settings
dataset_type = 'RecycleDataset'

data_root = "/opt/ml/segmentation/input/"

classes = ("Background", "General trash", "Paper", "Paper pack", "Metal",
           "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery",
           "Clothing")

palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
           [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
           [64, 64, 128], [128, 0, 192]]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         reduce_zero_label=True,),
         # imdecode_backend='cv2'),
    # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='Resize', img_scale=(512, 512)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(2048, 512),
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(2048, 512),
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(samples_per_gpu=4,
            workers_per_gpu=4,
            train=dict(type=dataset_type,
                       classes=classes,
                       palette=palette,
                       data_root=data_root,
                       reduce_zero_label=True,
                       img_dir='train/img',
                       ann_dir='train/mask',
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     classes=classes,
                     palette=palette,
                     data_root=data_root,
                     reduce_zero_label=True,
                     img_dir='val/img',
                     ann_dir='val/mask',
                     pipeline=val_pipeline),
            test=dict(type=dataset_type,
                      classes=classes,
                      palette=palette,
                      data_root=data_root,
                      reduce_zero_label=True,
                      img_dir='test/img',
                      ann_dir='test/mask',
                      pipeline=test_pipeline))
