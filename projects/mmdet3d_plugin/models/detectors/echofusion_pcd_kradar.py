# ------------------------------------------------------------------------
# Copyright (c) 2022 Fudan Zhang Vision Group. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from .polarformer import PolarFormer
from ..utils.positional_encoding import TransSinePositionalEncoding

@DETECTORS.register_module()
class EchoFusion_PCD_KRadar(PolarFormer):
    """PolarFormer."""

    def __init__(self,
                 out_channels=256,
                 polar_width=128,
                 num_levels=3,
                 nhead=8,
                 num_encoder=0,
                 num_decoder=3,
                 num_lidar_decoder=2,
                 use_different_res=True,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        
        super(EchoFusion_PCD_KRadar,
              self).__init__(use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        
        self.polar_width = polar_width
        self.num_levels = num_levels
        self.out_channels = out_channels
        self.use_different_res = use_different_res
        self.pos_encoding = TransSinePositionalEncoding(int(self.out_channels/2))
        self.lidar_transformer_layers = nn.ModuleList([nn.Transformer(d_model=self.out_channels, nhead=nhead, 
                                        num_encoder_layers=num_encoder, num_decoder_layers=num_lidar_decoder) for i in range(self.num_levels)])
    
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

        
    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        voxels, coors = self.voxelize(pts)
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)  # [N, 4]
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        if isinstance(x, list):
            outs = [x[i].permute(0, 1, 3, 2).contiguous() for i in range(len(x))]
        else:
            outs = x.permute(0, 1, 3, 2).contiguous()

        return outs
    
    def extract_img_feat(self, img, img_metas, pts_feats, gt_bboxes_3d, gt_labels_3d):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)  # Use a safer squeeze
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats, img_metas, pts_feats, gt_bboxes_3d, gt_labels_3d)
        
        return img_feats

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas, points, gt_bboxes_3d, gt_labels_3d):
        """Extract features from images and points."""
        pts_feats = self.extract_pts_feat(points) # use only one radar
        # img_feats = self.extract_img_feat(img, img_metas, pts_feats, gt_bboxes_3d, gt_labels_3d)

        ret_list = []
        for i in range(self.num_levels):
            feat = pts_feats[i]
            B, C, R, A = feat.shape
            if self.use_different_res:
                W = int(self.polar_width / 2**i)
            else:
                W = self.polar_width

            lidar_x_range = torch.arange(0., float(A), 1., device=feat.device)
            lidar_x_range = lidar_x_range.unsqueeze(0).repeat(B, 1)
            lidar_y_range = torch.arange(0., float(R), 1., device=feat.device)
            lidar_y_range = lidar_y_range.unsqueeze(0).repeat(B, 1)
            lidar_pos = self.pos_encoding(lidar_x_range, lidar_y_range) # [B, C, R, W]
            
            polar_x_range = torch.arange(0., float(W), 1., device=feat.device)
            polar_x_range = polar_x_range.unsqueeze(0).repeat(B, 1)
            polar_y_range = torch.arange(0., float(R), 1., device=feat.device)
            polar_y_range = polar_y_range.unsqueeze(0).repeat(B, 1)
            polar_rays_pos = self.pos_encoding(polar_x_range, polar_y_range) # [B, C, R, W]
            polar_rows = polar_rays_pos.permute(3, 0, 2, 1).flatten(1, 2) # [W, B*R, C]

            lidar_rows = feat + lidar_pos
            lidar_rows = lidar_rows.permute(3, 0, 2, 1).flatten(1, 2) # [A, B*R, C]

            bev_out = self.lidar_transformer_layers[i](lidar_rows, polar_rows)
            bev_out = bev_out.view(W, B, R, C).permute(1, 3, 2, 0) # [B, C, R, W]
            ret_list.append(bev_out)

        return ret_list

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      points=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (B, N, C, H, W). Defaults to None. N: default=1, 1 camera.
            points (torch.Tensor optional): Radar data of each sample with shape
                (B, N, C, H, W). Defaults to None. N: default=1, 1 radar.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # import pdb
        # pdb.set_trace()
        img_feats = self.extract_feat(img=img, img_metas=img_metas, points=points, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
    
    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        return self.simple_test(img_metas[0], img[0], points[0], **kwargs)
        # if num_augs == 1:
        #     img = [img] if img is None else img
        #     return self.simple_test(None, img_metas[0], img[0], **kwargs)
        # else:
        #     return self.aug_test(None, img_metas, img, **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
      
        return bbox_results
    
    def simple_test(self, img_metas, img=None, points=None, rescale=False,gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas, points=points, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
