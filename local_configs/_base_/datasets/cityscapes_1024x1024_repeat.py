# dataset settings
dataset_type = 'CustomDataset'
# data_root = '/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/dataset'
# data_root = '/data/2_data_server/cv-07/challenge/semantic_sementation/FeedFormer/FeedFormer-master/dataset/real_raw'
data_root = '/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/dataset'
# data_root = '/data/2_data_server/cv-07/challenge/semantic_sementation/open_earth_map/data'

img_norm_cfg = dict(
    # mean=[118.99, 121.39,106.77], std=[44.79, 40.69, 41.17], to_rgb=True)
    mean=[49,49,49], std=[34,34,34], to_rgb=False)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    # dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RepeatChannels'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.8),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    # dict(type='LoadImageFromFile'),
    dict(type='RepeatChannels'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='only_sar/train/images',
            ann_dir='only_sar/train/labels',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='only_sar/val/images',
        ann_dir='only_sar/val/labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))