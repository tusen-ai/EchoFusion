_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/kitti-3d-car.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

pi = 3.141592653589793
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -103.0, -5.0, 103.0, 103.0, 3.0]
polar_range = [0, -pi*30/180, -5.0, 103.0, pi*30/180, 3.0]
polar_voxel_size = [103/512, pi/3/256, 0.2]
radius_range = [0., 103., 103/128]  # [start, end, interval]
grid_res = 0.8
voxel_size = [grid_res, grid_res, grid_res]

output_size = [256, 64, 10]  # [azimuth, radius, height] TODO: choose better values
img_norm_cfg = dict(  # TODO: choose better values
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = ['Car', ]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True)

model = dict(
    type='EchoFusion_PCD_IMG',
    use_grid_mask=True,
    pretrained=dict(img='ckpts/resnet50_bev.pth'),
    pts_voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=polar_range,
        voxel_size=polar_voxel_size,
        max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=polar_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=polar_range,
        ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[256, 512]),
    pts_backbone=dict(
        type='RadarResNet',
        depth=50,
        in_channels=64,
        num_stages=4,
        strides=(1, (2, 1), 2, 2),
        out_indices=(1, 2, 3),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    pts_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3,
        norm_cfg=dict(type='SyncBN', eps=0.001, momentum=0.01),
    ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='FPN_RADIAL',
        num_encoder=0,  # encoder is not used here
        num_decoder=3,
        num_levels=3,
        polar_range=polar_range,
        radius_range=radius_range,
        use_different_res=True,
        use_bev_aug=False,
        output_multi_scale=True,
        grid_res=grid_res,
        pc_range=point_cloud_range,  # only z-dimension is used
        output_size=output_size,
        fpn_cfg=dict(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=3,
            relu_before_extra_convs=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
    ),
    pts_bbox_head=dict(
        type='EchoFusionHead',
        num_query=30,
        num_classes=1,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        polar_range=polar_range,
        radius_range=radius_range,
        code_size=8,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PolarTransformer',
            num_feature_levels=3,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256, num_levels=3),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
             decoder=dict(
                type='PolarTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                pc_range=point_cloud_range,  # not used
                radius_range=radius_range,  # not used
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=3)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,  # for pred filtering
            pc_range=point_cloud_range,  # not used
            max_num=10,
            voxel_size=voxel_size,  # not used
            num_classes=1),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,  # not used
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))  # not used

dataset_type = 'RADIalXDataset'
data_root = 'data/radial_kitti_format/'

file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadPolarPointsFromFile',
         coord_type='LIDAR',
         load_dim=11,
         use_dim=4,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True,
         with_label_3d=True, with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=polar_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),  # filter out boxes outside this range
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'points'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'cam2lidar', 'cam_intrinsic',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPolarPointsFromFile',
         coord_type='LIDAR',
         load_dim=11,
         use_dim=4,
         file_client_args=file_client_args),
    dict(type='PointsRangeFilter', point_cloud_range=polar_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'points'],
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'cam2lidar', 'cam_intrinsic',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow'))
        ])
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'radialx_infos_train.pkl',
            split='',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            pts_prefix='lidars',
            radar_prefix='radars_pcd',
            box_type_3d='LiDAR'),
    ),
    val=dict(type=dataset_type, 
             data_root=data_root,
             ann_file=data_root + 'radialx_infos_val.pkl',
             split='',
             pipeline=test_pipeline,
             classes=class_names, 
             modality=input_modality,
             pts_prefix='lidars',
             radar_prefix='radars_pcd',),
    test=dict(type=dataset_type, 
              data_root=data_root,
              ann_file=data_root + 'radialx_infos_test.pkl',
              split='',
              pipeline=test_pipeline,
              classes=class_names, 
              modality=input_modality,
              pts_prefix='lidars',
              radar_prefix='radars_pcd',),)

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        }),
    weight_decay=0.075)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline, save_best='auto')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
