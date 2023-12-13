from mmdet.models.necks import FPN
from ..utils.positional_encoding import TransSinePositionalEncoding
import torch
import torch.nn as nn
import cv2
from torch.nn.init import normal_
from mmdet.models import NECKS
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
import torch.nn.functional as F
import math
from ..utils.bev_aug import BEVRandomRotateScale, BEVRandomFlip
from torch.nn.modules.normalization import LayerNorm
from .fpn_trans import FPN_TRANS
import numpy as np
from matplotlib import pyplot as plt
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         build_positional_encoding,
                                         build_transformer_layer_sequence)


@NECKS.register_module()
class FPN_RADIAL(FPN_TRANS):
    """FPN_RADAR as image neck designed for single view RADIal Dataset.

    Args:
        topdown_layers (int): num of BasicBlock in topdown module. 
            Topdown module is used to transform 3D voxels to BEV map.
        grid_res (float): voxel size. Default: 0.5.
        pc_range (list): Point Cloud Range [x1, y1, z1, x2, y2, z2]. 
            Default: [-50, -50, -5, 50, 50, 3]. Value used in 
            nuscenes dataset in mmdet3d.
        output_size (list): output voxel map size, [Y, Z, X].
            Deafult: [128, 128, 10].
        cam_types: (list): A list of str. Represent camera order.
            Default: ['FRONT','FRONT_RIGHT','FRONT_LEFT', \
                      'BACK','BACK_LEFT','BACK_RIGHT']
            Camera order used in detr3d.
        fpn_cfg: (dict): FPN config. Please refer to mmdet.
            Default: None
    """

    def __init__(self,
                 grid_res=0.5,
                 pc_range=[-50, -50, -5, 50, 50, 3],
                 output_size=[128, 128, 10],
                 nhead=8,
                 num_encoder=4,
                 num_decoder=4,
                 num_radar_decoder=2,
                 num_levels=3,
                 polar_range=None,
                 radius_range=[1., 65., 1.],
                 scales=[1/16., 1/32., 1/64.],
                 use_different_res=False,
                 use_bev_aug=False,
                 output_multi_scale=False,
                 bev_aug_cfg=dict(rot_scale=dict(prob=0.5),
                        flip=dict(prob=0.5)),
                 cam_types=['FRONT','FRONT_RIGHT','FRONT_LEFT',
                                'BACK','BACK_LEFT','BACK_RIGHT'],
                 fpn_cfg=None):
        super(FPN_RADIAL, self).__init__(
            grid_res=grid_res,
            pc_range=pc_range,
            output_size=output_size,
            nhead=nhead,
            num_encoder=num_encoder,
            num_decoder=num_decoder,
            num_levels=num_levels,
            radius_range=radius_range,
            scales=scales,
            use_different_res=use_different_res,
            use_bev_aug=use_bev_aug,
            output_multi_scale=output_multi_scale,
            bev_aug_cfg=bev_aug_cfg,
            cam_types=cam_types,
            fpn_cfg=fpn_cfg
        )
        self.polar_range = polar_range
        self.radar_transformer_layers = nn.ModuleList([nn.Transformer(d_model=self.out_channels, nhead=nhead, 
                                        num_encoder_layers=num_encoder, num_decoder_layers=num_radar_decoder) for i in range(self.num_levels)])
    
    
    def _forward_radar(self, mlvl_feats, mlvl_radars_ra):
        """Forward function for single_camera.

        Args:
            mlvl_feats (list): List of torch.Tensor. 
                Contains feature maps of multiple levels.
                Each element with the shape of [B, C, R, W].
            
            mlvl_radars_ra (list): List of torch.Tensor. 
                Contains radar feature maps of multiple levels.
                Each element with the shape of [B, C, R, A].

        Returns:
            ortho (torch.Tensor): single camera BEV feature map with  \
                                    the shape of [B, C, H, W].
            mask (torch.Tensor): Mask indicating which region of BEV map \
                                    could be seen in the image, with the \
                                        shape of [B, 1, H, W]. Possitive value \
                                            indicates visibility.
        """

        ret_list = []
        for i in range(self.num_levels):
            scale = self.scales[i]
            feat = mlvl_feats[i]
            radar_feat = mlvl_radars_ra[i]

            B, C, R, W = feat.shape
            _, _, R_, A = radar_feat.shape
            assert R == R_, "radar feature should have same channel with image feature."

            polar_x_range = torch.arange(0., float(W), 1., device=feat.device)
            polar_x_range = polar_x_range.unsqueeze(0).repeat(B, 1)
            polar_y_range = torch.arange(0., float(R), 1., device=feat.device)
            polar_y_range = polar_y_range.unsqueeze(0).repeat(B, 1)
            polar_pos = self.pos_encoding(polar_x_range, polar_y_range)
            polar_rows = feat + polar_pos
            polar_rows = polar_rows.permute(3, 0, 2, 1).flatten(1, 2) # [W, B*R, C]
            
           
            radar_x_range = torch.arange(0., float(A), 1., device=feat.device)
            radar_x_range = radar_x_range.unsqueeze(0).repeat(B, 1)
            radar_y_range = torch.arange(0., float(R), 1., device=feat.device)
            radar_y_range = radar_y_range.unsqueeze(0).repeat(B, 1)
            radar_pos = self.pos_encoding(radar_x_range, radar_y_range)
            radar_rows = radar_feat + radar_pos
            radar_rows = radar_rows.permute(3, 0, 2, 1).flatten(1, 2) # [A, B*R, C]

            bev_out = self.radar_transformer_layers[i](radar_rows, polar_rows)
            bev_out = bev_out.view(W, B, R, C).permute(1, 3, 2, 0) # [B, C, R, W]
            ret_list.append(bev_out)
        
        return ret_list

    def _forward_single_camera(self, feature_list):
        """Forward function for single_camera.

        Args:
            feature_list (list): List of torch.Tensor. 
                Contains feature maps of multiple levels.
                Each element with the shape of [B, C, H, W].

        Returns:
            ortho (torch.Tensor): single camera BEV feature map with  \
                                    the shape of [B, C, H, W].
            mask (torch.Tensor): Mask indicating which region of BEV map \
                                    could be seen in the image, with the \
                                        shape of [B, 1, H, W]. Possitive value \
                                            indicates visibility.
        """

        ret_list = []
        for i in range(self.num_levels):
            scale = self.scales[i]
            feat = feature_list[i]
            B, C, H, W = feat.shape
            if self.use_different_res:
                R = int(self.radius / 2**i)
            else:
                R = self.radius

            img_x_range = torch.arange(0., float(W), 1., device=feat.device)
            img_x_range = img_x_range.unsqueeze(0).repeat(B, 1)
            img_y_range = torch.arange(0., float(H), 1., device=feat.device)
            img_y_range = img_y_range.unsqueeze(0).repeat(B, 1)
            img_pos = self.pos_encoding(img_x_range, img_y_range)
            
           
            polar_x_range = torch.arange(0., float(W), 1., device=feat.device)
            polar_x_range = polar_x_range.unsqueeze(0).repeat(B, 1)
            polar_y_range = torch.arange(0., float(R), 1., device=feat.device)
            polar_y_range = polar_y_range.unsqueeze(0).repeat(B, 1)
            polar_rays_pos = self.pos_encoding(polar_x_range, polar_y_range) # [B, C*2, R, W]

            img_columns = feat + img_pos
            img_columns = img_columns.permute(2, 0, 3, 1).flatten(1, 2) # [H, B*W, C]
            polar_rays = polar_rays_pos.permute(2, 0, 3, 1).flatten(1, 2) # [R, B*W, C*2]

            bev_out = self.transformer_layers[i](img_columns, polar_rays)
            bev_out = bev_out.view(R, B, W, C).permute(1, 3, 0, 2) # [B, C, R, W]
            ret_list.append(bev_out)

        return ret_list

    def forward(self, feature_list, img_metas, radar_list, gt_bboxes_3d, gt_labels_3d):
        """Forward function for FPN.

        Args:
            feature_list (list): List of torch.Tensor. 
                Contains feature maps of multiple levels.
                Each element with the shape of [N, C, H, W].
                N=B*num_cam.
            img_metas (List): List of dict. Each dict contains  \
                information such as gt, filename and camera matrix. 
                The lenth of equals to batch size.

        Returns:
            topdown (torch.Tensor): multi-camera BEV feature map with \
                                    the shape of [B, C, H, W].
        """
        feature_list = self.fpn(feature_list)

        B = len(img_metas)
        N, C, _, _ = feature_list[0].shape  # N = B*num_cam
        num_cam = int(N/B)

        feature_list = [feat.view(B, num_cam, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                        for feat in feature_list]

        cam_id = 0
        cam_type = self.cam_types[cam_id]  # Only use front view image
        feature_single_cam = [feat[:, cam_id] for feat in feature_list]

        lidar2img = [img_meta['lidar2img'] for img_meta in img_metas]

        res_batch = []
        mask_batch = []
        ret_polar_ray_list = self._forward_single_camera(feature_single_cam)

        for lvl in range(self.num_levels):
            feature_size = feature_list[lvl].shape[3:]
            scale = self.scales[lvl]
            bev_feat, mask = self._interpolate_by_image_column(
                feature_size, scale, cam_type, ret_polar_ray_list[lvl], lidar2img, lvl)
            res_batch.append(bev_feat.clone())
            mask_batch.append(mask.unsqueeze(1).clone())

        if not self.output_multi_scale:
            mask_batch[mask_batch == 0] = 1  # avoid zero divisor
            res_batch /= mask_batch
            res_batch = [res_batch]
        else:
            for lvl in range(self.num_levels):
                mask_batch[lvl][mask_batch[lvl] == 0] = 1
                res_batch[lvl] /= mask_batch[lvl]

        # res_batch is list
        if self.use_bev_aug:
            res_batch = self.bev_aug[0](res_batch, gt_bboxes_3d, gt_labels_3d)
            res_batch = self.bev_aug[1](res_batch, gt_bboxes_3d)

        # query radar features
        res_batch = self._forward_radar(res_batch, radar_list)

        return res_batch

    def _interpolate_by_image_column(self, feature_size, scale, cam_type, polar_ray, lidar2img, lvl):
        B, device = polar_ray.shape[0], polar_ray.device
        # Generate voxel corners-10-10
        if self.output_multi_scale and self.use_different_res:
            x = torch.arange(
                0., 1, 1/self.output_size[0]*2**lvl) * (self.polar_range[4] - self.polar_range[1]) + self.polar_range[1] + 1/self.output_size[0]/2
            y = torch.arange(self.radius_range[0], self.radius_range[1],
                            self.radius_range[2]*2**lvl) + self.radius_range[2]/2
            z = torch.arange(
                self.pc_range[2], self.pc_range[5], self.grid_res*2**lvl) + self.grid_res/2
        else:
            x = torch.arange(
                0., 1, 1/self.output_size[0]) * (self.polar_range[4] - self.polar_range[1]) + self.polar_range[1] + 1/self.output_size[0]/2
            y = torch.arange(
                self.radius_range[0], self.radius_range[1], self.radius_range[2]) + self.radius_range[2]/2
            z = torch.arange(
                self.pc_range[2], self.pc_range[5], self.grid_res) + self.grid_res/2
        xx, yy, zz = torch.meshgrid(x, y, z)
        corners = torch.cat([xx.unsqueeze(-1).sin()*yy.unsqueeze(-1),
                            xx.unsqueeze(-1).cos()*yy.unsqueeze(-1), zz.unsqueeze(-1)], dim=-1)
        
        # Project to image plane, calib is same for all frames
        dimx, dimy, dimz = corners.shape[:3]
        corners_array = corners.reshape(-1, 3).numpy()
        img_corners, _ = cv2.projectPoints(corners_array, 
                              lidar2img[0]['rvec'], 
                              lidar2img[0]['tvec'],
                              lidar2img[0]['cam_mat'],
                              lidar2img[0]['dist_coeffs'])
        img_corners = torch.from_numpy(img_corners).reshape(dimx, dimy, dimz, 2)
        img_corners = img_corners.unsqueeze(0).repeat(
            B, 1, 1, 1, 1).to(device)  # [B, X, Y, Z, 2]

        # normalize to [-1, 1]
        img_height, img_width = feature_size
        img_size = img_corners.new([img_width, img_height]) / scale
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)

        sampling_pixel = norm_corners[..., 0]

        batch, width, depth, height, _ = norm_corners.size()

        rot_mat = cv2.Rodrigues(lidar2img[0]['rvec'])[0]
        cam_coords = -np.linalg.inv(rot_mat) @ lidar2img[0]['tvec']
        cam_coords = torch.from_numpy(cam_coords[None, None, None, :2]).repeat(B, 1, 1, 1).to(device)  # [B, 1, 1, 2]

        xx, yy = torch.meshgrid(x, y)
        grid_map = torch.stack([xx.sin()*yy, xx.cos()*yy], dim=-1).to(device)
        grid_map = grid_map.unsqueeze(0).repeat(B, 1, 1, 1)
        grid_map -= cam_coords
        radius_map = torch.norm(grid_map, dim=-1)
        norm_radius_map = (
            2*(radius_map-self.radius_range[0])/self.radius - 1).clamp(-1, 1)

        sample_loc = torch.stack([sampling_pixel,
                                norm_radius_map.unsqueeze(-1).repeat(1, 1, 1, sampling_pixel.shape[-1])], dim=-1)
        sample_loc = sample_loc.reshape(batch, width, depth*height, 2)

        sampling = F.grid_sample(polar_ray, sample_loc, padding_mode="border", align_corners=True)
        sampling = sampling.reshape(
            batch, sampling.shape[1], width, depth, height)
        sampling = sampling.mean(-1)

        visible = ((abs(sampling_pixel) != 1).all(-1)
                ) & (abs(norm_radius_map) != 1)

        assert cam_type == 'FRONT', "FPN_RADAR is used for front view image."
        # visible[:, int(width/4):int(width*3/4), :] = False

        interpolated_feat = sampling * visible[:, None, :, :]
        interpolated_feat = interpolated_feat.transpose(2, 3)
        visible = visible.transpose(1, 2)


        return interpolated_feat, visible
