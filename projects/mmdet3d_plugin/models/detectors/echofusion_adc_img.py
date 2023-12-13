# ------------------------------------------------------------------------
# Copyright (c) 2022 Fudan Zhang Vision Group. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np
from cplxmodule.nn import CplxConv2d, CplxLinear, CplxDropout
from cplxmodule.nn import CplxModReLU, CplxParameter, CplxModulus, CplxToCplx
from cplxmodule.nn.modules.casting import TensorToCplx
from cplxmodule.nn import RealToCplx, CplxToReal

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from .polarformer import PolarFormer

class Range_Fourier_Net(nn.Module):
    def __init__(self, channels=512, convert_cplx=True):
        super(Range_Fourier_Net, self).__init__()
        self.convert_cplx = convert_cplx
        self.range_nn = CplxConv2d(channels, channels, 1, bias = False)
        range_weights = np.zeros((channels, channels, 1, 1), dtype = np.complex64)
        for j in range(0, channels):
            for h in range(0, channels):
                range_weights[h][j][0][0] = np.exp(-1j * 2 * np.pi *(j*h/channels))
        range_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(range_weights)))
        self.range_nn.weight = CplxParameter(range_weights)

    def forward(self, x):
        cplx = RealToCplx()(x)
        cplx = self.range_nn(cplx)
        if self.convert_cplx:
            mag = abs(cplx)
            angle = cplx.angle
            out = torch.cat((angle, mag), dim=-1)
        else:
            out = CplxToReal()(cplx)
        return out

@DETECTORS.register_module()
class EchoFusion_ADC_IMG(PolarFormer):
    """PolarFormer."""

    def __init__(self,
                 use_range_embedder=True,
                 convert_cplx=True,
                 use_log=False,
                 use_layer_norm=False,
                 use_batch_norm=False,
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
                 radar_input_dim=384,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EchoFusion_ADC_IMG,
              self).__init__(use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.use_log = use_log
        self.use_range_embedder = use_range_embedder
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.epsillon = 1e-3
        self.numSamplePerChirp = 512
        self.numChirps = 256
        self.numReducedDoppler = 16
        self.numChirpsPerLoop = 16
        self.radar_input_dim = radar_input_dim
        if self.with_pts_backbone:
            # with_pts_backbone is used for radar
            layers = []
            if self.use_layer_norm:
                layers.append(nn.LayerNorm([self.numSamplePerChirp, self.numChirps]))
            elif self.use_batch_norm:
                layers.append(nn.BatchNorm2d(self.radar_input_dim))
            layers.append(nn.Conv2d(self.radar_input_dim, 64, 1))
            self.radar_embedder = nn.Sequential(*layers)
            if self.use_range_embedder:
                self.range_embedder = Range_Fourier_Net(self.numSamplePerChirp, convert_cplx)

    def extract_radar_feat(self, radars_adc):
        
        # Unwrap TX information
        B, R, D, _ = radars_adc.shape
        if self.use_log:
            radars_adc[:, :, :, ::2] = torch.log(radars_adc[:, :, :, ::2] + self.epsillon)
        if self.use_range_embedder:
            radars_adc = self.range_embedder(radars_adc)  # [B, R, D, A*2]

        doppler_indexes = []
        dividend_constant_arr = torch.arange(0, self.numReducedDoppler*self.numChirpsPerLoop, self.numReducedDoppler)
        for doppler_bin in range(self.numChirps):
            DopplerBinSeq = torch.remainder(doppler_bin + dividend_constant_arr, self.numChirps)
            DopplerBinSeq = torch.cat((DopplerBinSeq[0].unsqueeze(0),DopplerBinSeq[5:]))
            doppler_indexes.append(DopplerBinSeq.tolist())
        decoded_adc = radars_adc[:,:,doppler_indexes,:].reshape(B, R, D, -1)  # [B, R, D, A*2]

        radars_adc = decoded_adc.permute(0, 3, 1, 2).contiguous()  # [B, A*2, R, D]
        radars_adc = self.radar_embedder(radars_adc)  # [B, C, R, D]

        # Feature extraction
        radar_feats = self.pts_backbone(radars_adc)
        if self.with_pts_neck:
            radar_feats = self.pts_neck(radar_feats)
        
        return radar_feats
        
    def extract_img_feat(self, img, img_metas, radar_feats, gt_bboxes_3d, gt_labels_3d):
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
            img_feats = self.img_neck(img_feats, img_metas, radar_feats, gt_bboxes_3d, gt_labels_3d)
        
        return img_feats

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas, radars_adc, gt_bboxes_3d, gt_labels_3d):
        """Extract features from images and points."""
        radar_feats = self.extract_radar_feat(radars_adc[0]) # use only one radar
        img_feats = self.extract_img_feat(img, img_metas, radar_feats, gt_bboxes_3d, gt_labels_3d)
        return img_feats

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
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      radars_adc=None,
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
            radars_adc (torch.Tensor optional): Radar data of each sample with shape
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
        img_feats = self.extract_feat(img=img, img_metas=img_metas, radars_adc=radars_adc, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
    
    def forward_test(self, img_metas, img=None, radars_adc=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        radars_adc = [radars_adc] if radars_adc is None else radars_adc
        return self.simple_test(img_metas[0], img[0], radars_adc[0], **kwargs)
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
    
    def simple_test(self, img_metas, img=None, radars_adc=None, rescale=False,gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas, radars_adc=radars_adc, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

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
