import copy
import torch
import numpy as np
import os.path as osp
import tempfile
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet3d.datasets import KittiDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from .utils import GetFullMetrics

@DATASETS.register_module()
class RADIalXDataset(KittiDataset):
    r"""RADIal Dataset.

    This class serves as the API for experiments on the `RADIal Dataset
    <https://github.com/valeoai/RADIal>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'lidars'.
        radar_prefix (str, optional): Prefix of radar files.
            Defaults to 'radars_rt'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'RADAR': Box in RADAR coordinates.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ('Car', )

    def __init__(self,
                 data_root,
                 ann_file,
                 split='',
                 pts_prefix='lidars',
                 radar_prefix='radars_rt',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 noise_lvl=0,
                 pcd_limit_range=[-103, 0, -10, 103, 103, 10]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            split=split,
            pcd_limit_range=pcd_limit_range,
            pts_prefix=pts_prefix)

        self.radar_prefix = radar_prefix
        self.noise_lvl = noise_lvl

    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:06d}.bin')
        return pts_filename
    
    def _get_radar_filename(self, idx):
        """Get radar filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        radar_filename = osp.join(self.root_split, self.radar_prefix,
                                f'{idx:06d}.bin')
        return radar_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str | None): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = osp.join(self.data_root,
                                    info['image']['image_path'])

        # record lidar2img as dict since dist_coeffs can't be merged in intrinsics
        if self.noise_lvl > 0:
            # generate noise of normal distribution with 0 mean
            tvec_noise = np.random.normal(0, self.noise_lvl * 5, size=(3,)) * 0.01
            rvec_noise = np.random.normal(0, self.noise_lvl) * np.pi / 180
            rot_angle = np.linalg.norm(info['calib']['rvec'])
            noise_angle = rot_angle + rvec_noise
            rvec_noise_scalar = noise_angle / rot_angle
        else:
            tvec_noise = np.zeros((3,))
            rvec_noise_scalar = 1.0

        lidar2img = dict(
            dist_coeffs = info['calib']['dist_coeffs'].astype(np.float32),
            cam_mat = info['calib']['cam_mat'].astype(np.float32),
            rvec = info['calib']['rvec'].astype(np.float32) * rvec_noise_scalar,
            tvec = info['calib']['tvec'].astype(np.float32) + tvec_noise
        )

        pts_filename = self._get_pts_filename(sample_idx)
        radar_filename = self._get_radar_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            radar_filename=radar_filename,
            img_prefix=None,
            img_filename=[img_filename],
            lidar2img=lidar2img)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # TODO: delete commented redundant code
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        annos = info['annos']
        # we need other objects to avoid collision when sample
        # annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=7)

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        
        predictions = [torch.cat([pred['pts_bbox']['boxes_3d'].tensor, 
               pred['pts_bbox']['scores_3d'][:, None]], dim=-1).numpy() for pred in results]
        object_labels = [np.concatenate([info['annos']['location'], info['annos']['dimensions'], 
                    info['annos']['rotation_y'][:, None]], axis=-1) for info in self.data_infos]

        assert len(predictions) == len(object_labels), "prediction and gt lengths are not equal."
        
        # TODO: delete results saving part
        pred_to_save = [torch.cat([pred['pts_bbox']['boxes_3d'].tensor, 
               pred['pts_bbox']['scores_3d'][:, None]], dim=-1).numpy() for pred in results]
        np.save('results_3d.npy', np.array(pred_to_save))  # save results

        ap_result_str, ap_dict= GetFullMetrics(predictions, object_labels)

        print_log(f'Results of RADIal:\n' + ap_result_str, logger=logger)

        return ap_dict
    
    def test_GetFullMetrics(self, predictions, object_labels):
        # TODO: Delete this function.
        ap_result_str, ap_dict= GetFullMetrics(predictions, object_labels)
        return ap_result_str
