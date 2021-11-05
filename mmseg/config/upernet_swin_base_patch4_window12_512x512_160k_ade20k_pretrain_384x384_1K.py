_base_ = [
    'upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_'
    'pretrain_224x224_1K.py'
]
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth'
model = dict(
    #pretrained='pretrain/swin_base_patch4_window12_384.pth',
    
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained) #
        ),
        
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=11),
    auxiliary_head=dict(in_channels=512, num_classes=11))
