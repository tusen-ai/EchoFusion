# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import spconv.pytorch as spconvpy
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmcv.cnn import ConvModule
from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from mmdet3d.models import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class KRadarSparseEncoder(nn.Module):
    r"""Sparse encoder for KRadar.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=64,
                 output_channels=256,
                 encoder_channels=((64, 64, 64), (128, 128, 128), (256, 256, 256)),
                 encoder_strides=(1, 2, 2),
                 encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1)),
                 bev_out_channels=(256, 256, 256),
                 bev_kernel_sizes=(3, 6, 12),
                 bev_strides=(1, 2, 4),
                 bev_paddings=(1, 2, 4),
                 mlvl_output=True,
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_strides = encoder_strides
        self.encoder_paddings = encoder_paddings
        self.bev_out_channels = bev_out_channels
        self.bev_kernel_sizes = bev_kernel_sizes
        self.bev_strides = bev_strides
        self.bev_paddings = bev_paddings
        self.mlvl_output = mlvl_output
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}    

        # conv_input for channel extension
        self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                1,
                norm_cfg=norm_cfg,
                indice_key='subm1',
                conv_type='SparseConv3d',
                order=('conv', ))  # no bias

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.s2b_encoder = nn.ModuleList()
        self.bev_encoder = nn.ModuleList()
        self.out_encoder = nn.ModuleList()
        merged_channels = sum(self.bev_out_channels)

        for i, blocks in enumerate(self.encoder_channels):
            in_channel = tuple(blocks)[-1]
            z_kernel_size = math.ceil(self.sparse_shape[0] / 2**i)
            self.s2b_encoder.append(
                                make_sparse_convmodule(
                                    in_channel,
                                    in_channel,
                                    (z_kernel_size, 1, 1),
                                    norm_cfg=norm_cfg,
                                    indice_key=f's2b{i + 1}',
                                    conv_type='SparseConv3d'))
            self.bev_encoder.append(
                                ConvModule(
                                    in_channel,
                                    self.bev_out_channels[i],
                                    self.bev_kernel_sizes[i],
                                    stride=self.bev_strides[i],
                                    padding=self.bev_paddings[i],
                                    conv_cfg=dict(type='ConvTranspose2d'),
                                ))
            if self.mlvl_output:
                self.out_encoder.append(
                                ConvModule(
                                    merged_channels,
                                    self.bev_out_channels[i],
                                    self.bev_kernel_sizes[i],
                                    stride=self.bev_strides[i],
                                    padding=self.bev_paddings[i],
                                    conv_cfg=dict(type='Conv2d'),
                                ))

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        out_bev_features = []
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)
            encode_features.append(x)
            y = self.s2b_encoder[i](x)
            bev_in = y.dense()

            N, C, D, H, W = bev_in.shape
            bev_in = bev_in.view(N, C * D, H, W)
            out_bev_features.append(self.bev_encoder[i](bev_in))

        spatial_features = torch.cat(out_bev_features, dim=1)
        if self.mlvl_output:
            out_features = []
            for i, out_layer in enumerate(self.out_encoder):
                out_features.append(out_layer(spatial_features))
            return out_features

        return spatial_features

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=self.encoder_strides[i],
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=1,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = spconv.SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

@MIDDLE_ENCODERS.register_module()
class KRadarSparseEncoderV2(nn.Module):
    r"""Sparse encoder for KRadar. Added weight loading.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=64,
                 output_channels=256,
                 encoder_channels=((64, 64, 64), (128, 128, 128), (256, 256, 256)),
                 encoder_strides=(1, 2, 2),
                 encoder_paddings=(1, 1, 1),
                 bev_out_channels=(256, 256, 256),
                 bev_kernel_sizes=(3, 6, 12),
                 bev_strides=(1, 2, 4),
                 bev_paddings=(1, 2, 4),
                 mlvl_output=True,
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_strides = encoder_strides
        self.encoder_paddings = encoder_paddings
        self.bev_out_channels = bev_out_channels
        self.bev_kernel_sizes = bev_kernel_sizes
        self.bev_strides = bev_strides
        self.bev_paddings = bev_paddings
        self.mlvl_output = mlvl_output
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}    

        # conv_input for channel extension
        self.input_conv = spconv.SparseConv3d(
            in_channels=self.in_channels, out_channels=self.base_channels,
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key='sp0') 
        
        in_channels = self.base_channels
        for idx_enc, blocks in enumerate(self.encoder_channels):
            setattr(self, f'spconv{idx_enc}', \
                spconv.SparseConv3d(in_channels=in_channels, out_channels=blocks[0], kernel_size=3, \
                    stride=self.encoder_strides[idx_enc], padding=self.encoder_paddings[idx_enc], dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}', nn.BatchNorm1d(blocks[0]))
            setattr(self, f'subm{idx_enc}a', \
                spconv.SubMConv3d(in_channels=blocks[0], out_channels=blocks[1], kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}a', nn.BatchNorm1d(blocks[1]))
            setattr(self, f'subm{idx_enc}b', \
                spconv.SubMConv3d(in_channels=blocks[1], out_channels=blocks[2], kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}b', nn.BatchNorm1d(blocks[2]))
            in_channels = blocks[2]

            # sparse to dense
            z_kernel_size = math.ceil(self.sparse_shape[0] / 2**idx_enc)
            setattr(self, f'toBEV{idx_enc}', \
                spconv.SparseConv3d(in_channels=in_channels, \
                    out_channels=in_channels, kernel_size=(z_kernel_size, 1, 1)))
            setattr(self, f'bnBEV{idx_enc}', \
                nn.BatchNorm1d(in_channels))
            setattr(self, f'convtrans2d{idx_enc}', \
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.bev_out_channels[idx_enc], \
                    kernel_size=self.bev_kernel_sizes[idx_enc], stride=self.bev_strides[idx_enc],  padding=self.bev_paddings[idx_enc]))
            setattr(self, f'bnt{idx_enc}', nn.BatchNorm2d(self.bev_out_channels[idx_enc]))

        self.relu = nn.ReLU()
        self.out_encoder = nn.ModuleList()
        merged_channels = sum(self.bev_out_channels)

        for i in range(self.stage_num):
            if self.mlvl_output:
                self.out_encoder.append(
                                ConvModule(
                                    merged_channels,
                                    self.bev_out_channels[i],
                                    self.bev_kernel_sizes[i],
                                    stride=self.bev_strides[i],
                                    padding=self.bev_paddings[i],
                                    conv_cfg=dict(type='Conv2d'),
                                ))

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.input_conv(input_sp_tensor)

        encode_features = []
        out_bev_features = []
        for idx_layer in range(self.stage_num):
            x = getattr(self, f'spconv{idx_layer}')(x)
            x.features = getattr(self, f'bn{idx_layer}')(x.features)
            x.features = self.relu(x.features)
            x = getattr(self, f'subm{idx_layer}a')(x)
            x.features = getattr(self, f'bn{idx_layer}a')(x.features)
            x.features = self.relu(x.features)
            x = getattr(self, f'subm{idx_layer}b')(x)
            x.features = getattr(self, f'bn{idx_layer}b')(x.features)
            x.features = self.relu(x.features)

            bev_sp = getattr(self, f'toBEV{idx_layer}')(x)
            bev_sp.features = getattr(self, f'bnBEV{idx_layer}')(bev_sp.features)
            bev_sp.features = self.relu(bev_sp.features)
            bev_in = bev_sp.dense()
            N, C, D, H, W = bev_in.shape
            bev_in = bev_in.view(N, C * D, H, W)
            bev_dense = getattr(self, f'convtrans2d{idx_layer}')(bev_in)
            bev_dense = getattr(self, f'bnt{idx_layer}')(bev_dense)
            out_bev_features.append(bev_dense)

        spatial_features = torch.cat(out_bev_features, dim=1)
        if self.mlvl_output:
            out_features = []
            for i, out_layer in enumerate(self.out_encoder):
                out_features.append(out_layer(spatial_features))
            return out_features

        return spatial_features


@MIDDLE_ENCODERS.register_module()
class KRadarSparseEncoderV3(nn.Module):
    r"""Sparse encoder for KRadar. Added weight loading.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=64,
                 output_channels=256,
                 encoder_channels=((64, 64, 64), (128, 128, 128), (256, 256, 256)),
                 encoder_strides=(1, 2, 2),
                 encoder_paddings=(1, 1, 1),
                 bev_out_channels=(256, 256, 256),
                 bev_kernel_sizes=(3, 6, 12),
                 bev_strides=(1, 2, 4),
                 bev_paddings=(1, 2, 4),
                 mlvl_output=True,
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_strides = encoder_strides
        self.encoder_paddings = encoder_paddings
        self.bev_out_channels = bev_out_channels
        self.bev_kernel_sizes = bev_kernel_sizes
        self.bev_strides = bev_strides
        self.bev_paddings = bev_paddings
        self.mlvl_output = mlvl_output
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}    

        # conv_input for channel extension
        self.input_conv = spconvpy.SparseConv3d(
            in_channels=self.in_channels, out_channels=self.base_channels,
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key='sp0') 
        
        in_channels = self.base_channels
        for idx_enc, blocks in enumerate(self.encoder_channels):
            setattr(self, f'spconv{idx_enc}', \
                spconvpy.SparseConv3d(in_channels=in_channels, out_channels=blocks[0], kernel_size=3, \
                    stride=self.encoder_strides[idx_enc], padding=self.encoder_paddings[idx_enc], dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}', nn.BatchNorm1d(blocks[0]))
            setattr(self, f'subm{idx_enc}a', \
                spconvpy.SubMConv3d(in_channels=blocks[0], out_channels=blocks[1], kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}a', nn.BatchNorm1d(blocks[1]))
            setattr(self, f'subm{idx_enc}b', \
                spconvpy.SubMConv3d(in_channels=blocks[1], out_channels=blocks[2], kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}b', nn.BatchNorm1d(blocks[2]))
            in_channels = blocks[2]

            # sparse to dense
            z_kernel_size = math.ceil(self.sparse_shape[0] / 2**idx_enc)
            setattr(self, f'toBEV{idx_enc}', \
                spconvpy.SparseConv3d(in_channels=in_channels, \
                    out_channels=in_channels, kernel_size=(z_kernel_size, 1, 1)))
            setattr(self, f'bnBEV{idx_enc}', \
                nn.BatchNorm1d(in_channels))
            setattr(self, f'convtrans2d{idx_enc}', \
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.bev_out_channels[idx_enc], \
                    kernel_size=self.bev_kernel_sizes[idx_enc], stride=self.bev_strides[idx_enc],  padding=self.bev_paddings[idx_enc]))
            setattr(self, f'bnt{idx_enc}', nn.BatchNorm2d(self.bev_out_channels[idx_enc]))

        self.relu = nn.ReLU()
        self.out_encoder = nn.ModuleList()
        merged_channels = sum(self.bev_out_channels)

        for i in range(self.stage_num):
            if self.mlvl_output:
                self.out_encoder.append(
                                ConvModule(
                                    merged_channels,
                                    self.bev_out_channels[i],
                                    self.bev_kernel_sizes[i],
                                    stride=self.bev_strides[i],
                                    padding=self.bev_paddings[i],
                                    conv_cfg=dict(type='Conv2d'),
                                ))

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconvpy.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.input_conv(input_sp_tensor)

        encode_features = []
        out_bev_features = []
        for idx_layer in range(self.stage_num):
            x = getattr(self, f'spconv{idx_layer}')(x)
            x = x.replace_feature(getattr(self, f'bn{idx_layer}')(x.features))
            x = x.replace_feature(self.relu(x.features))
            x = getattr(self, f'subm{idx_layer}a')(x)
            x = x.replace_feature(getattr(self, f'bn{idx_layer}a')(x.features))
            x = x.replace_feature(self.relu(x.features))
            x = getattr(self, f'subm{idx_layer}b')(x)
            x = x.replace_feature(getattr(self, f'bn{idx_layer}b')(x.features))
            x = x.replace_feature(self.relu(x.features))

            bev_sp = getattr(self, f'toBEV{idx_layer}')(x)
            bev_sp = bev_sp.replace_feature(getattr(self, f'bnBEV{idx_layer}')(bev_sp.features))
            bev_sp = bev_sp.replace_feature(self.relu(bev_sp.features))
            bev_in = bev_sp.dense()
            N, C, D, H, W = bev_in.shape
            bev_in = bev_in.view(N, C * D, H, W)
            bev_dense = getattr(self, f'convtrans2d{idx_layer}')(bev_in)
            bev_dense = getattr(self, f'bnt{idx_layer}')(bev_dense)
            out_bev_features.append(bev_dense)

        spatial_features = torch.cat(out_bev_features, dim=1)
        if self.mlvl_output:
            out_features = []
            for i, out_layer in enumerate(self.out_encoder):
                out_features.append(out_layer(spatial_features))
            return out_features

        return spatial_features