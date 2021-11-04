_base_ = './htc_without_semantic_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))))
albu_train_transforms = [
    dict(type='Sharpen', p=0.8),
    dict(type='CLAHE', p=0.8),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.8),
    dict(type='RandomRotate90', always_apply=False, p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=1.0),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[-0.3, -0.1],
                contrast_limit=[-0.3, -0.1],
                p=1.0)
        ],
        p=0.5),
    dict(
        type='RGBShift',
        r_shift_limit=10,
        g_shift_limit=10,
        b_shift_limit=10,
        p=0.5),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.5),
    dict(type='ChannelShuffle', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.5)
]
albu_mask_transforms = [
    dict(
        type='GridDropout',
        always_apply=False,
        p=0.5),
]

train_root = '/opt/ml/segmentation/input/train_all/'
pseudo_root = '/opt/ml/segmentation/merge_image/'
val_root = '/opt/ml/segmentation/input/stratified_val/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.5,
        hflag=False,
        aug_ratio=0.5),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
'''
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=True,
        flip_direction = ['horizontal', 'vertical' ,'diagonal'],
        transforms=[
            dict(type='Resize', img_scale=[(384, 384), (512, 512), (768, 768), (1024, 1024)],
                 multiscale_mode="value",
                 keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
'''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(384, 384), (512, 512), (768, 768), (1024, 1024)],
        flip=[True, True, True, True],
        flip_direction = ['horizontal', 'vertical' ,'diagonal'],
        transforms=[
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(
        seg_prefix= train_root + 'mask',
        pipeline=train_pipeline),
    val=dict(
        seg_prefix= val_root + 'mask',   
        pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))


classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic",
           "Styrofoam", "Plastic bag", "Battery", "Clothing")
'''data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        seg_prefix= train_root + 'mask',   
        pipeline=train_pipeline),
    val=dict(
        seg_prefix= val_root + 'mask',   
        pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)'''
ClassBaldataset = dict(
        type = 'ClassBalancedDataset',
        oversample_thr=0.54,
        dataset= dict(
                type='CocoDataset',
                ann_file= train_root + 'labeling_data.json',
                img_prefix= train_root,
                classes = classes,
                seg_prefix= train_root + 'mask',
                pipeline=train_pipeline),
    )

datasetpseudo = dict(
    type='CocoDataset',
    ann_file= pseudo_root + 'labeling_data.json',
    img_prefix= pseudo_root,
    classes = classes,
    seg_prefix= pseudo_root + 'mask',
    pipeline=train_pipeline
    )

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=[
        ClassBaldataset,
        datasetpseudo
    ],
    val=dict(
        seg_prefix= val_root + 'mask',   
        pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)
