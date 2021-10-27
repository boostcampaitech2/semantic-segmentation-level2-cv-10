dataset_type = 'CocoDataset'
data_root = '/opt/ml/segmentation/input/data/'
train_root = '/opt/ml/segmentation/input/stratified_train/'
val_root = '/opt/ml/segmentation/input/stratified_val/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic",
           "Styrofoam", "Plastic bag", "Battery", "Clothing")


albu_train_transforms = [
    dict(type='ShiftScaleRotate',
         shift_limit=0.0625,
         scale_limit=0.0,
         rotate_limit=0,
         interpolation=1,
         p=0.7),
    dict(type='RandomBrightnessContrast',
         brightness_limit=[0.1, 0.3],
         contrast_limit=[0.1, 0.3],
         p=0.5),
    dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='OneOf',
         transforms=[
             dict(type='Blur', blur_limit=3, p=1.0),
             dict(type='MedianBlur', blur_limit=3, p=1.0)
         ],
         p=0.5),
]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= train_root + 'labeling_data.json',
        img_prefix= train_root,
        classes = classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=val_root + 'labeling_data.json',
        img_prefix=val_root,
        classes = classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'], classwise = True)
