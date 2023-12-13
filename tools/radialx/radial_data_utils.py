# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import math
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from skimage import io


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)


def get_radial_info_path(idx,
                        prefix,
                        info_type='images',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    file_path = Path(info_type) / img_idx_str
    # if training:
    #     file_path = Path('training') / info_type / img_idx_str
    # else:
    #     file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='images',
                   use_prefix_id=False):
    return get_radial_info_path(idx, prefix, info_type, '.png', training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='labels_x',
                   use_prefix_id=False):
    return get_radial_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_lidars_info_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_radial_info_path(idx, prefix, 'lidars', '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def get_radars_info_path(idx,
                      prefix,
                      modality='radars_rt',
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    assert modality in ['radars_rt', 'radars_pcd', 'radars_ra'], "Modality not supported."
    return get_radial_info_path(idx, prefix, modality, '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):

    img_idx_str = 'camera_calib'
    img_idx_str += '.npy'
    info_type = 'calibs'
    prefix = Path(prefix)
    file_path = Path(info_type) / img_idx_str

    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'doppler': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()

    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[1] != '-1'])

    if num_objects == 0: # if there is no object
        content = []
    annotations['name'] = np.array(['Car' for x in content])
    
    annotations['dimensions'] = np.array([[float(info) for info in x[3:6]]
                                    for x in content]).reshape(-1, 3)

    annotations['location'] = np.array([[float(info) for info in x[0:3]]
                                    for x in content]).reshape(-1, 3)

    annotations['location'][:, 2] = annotations['location'][:, 2] - annotations['dimensions'][:, -1]/ 2

    annotations['rotation_y'] = np.array([float(x[6]) for x in content]).reshape(-1)

    # deal with truncated, occluded and alpha
    annotations['alpha'] = np.array([-10 for x in content])
    annotations['truncated'] = np.array([0.0 for x in content])
    annotations['occluded'] = np.array([0 for x in content])

    index = list(range(num_objects))
    annotations['index'] = np.array(index, dtype=np.int32)
    
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_radial_image_info(path,
                         training=True,
                         label_info=True,
                         lidars=False,
                         radars=True,
                         calib=True,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    RADIal annotation format version 1:
    {
        image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        [optional]point_cloud: {
            num_features: 11
            lidars_path: ...
        }
        [optional]radar: {
            radars_path: ...
        }
        [optional]calib: {
            rvec: ...
            tvec: ...
            cam_mat: ...
            dist_coeffs: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            doppler: [num_gt] doppler array
            power: [num_gt] reflected power array
            difficulty: kitti difficulty
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 11}
        radar_info = {'radar_shape': [512, 256, 16]}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if lidars:
            pc_info['lidars_path'] = get_lidars_info_path(
                idx, path, training, relative_path)
        if radars:
            radar_info['radars_rt_path'] = get_radars_info_path(
                idx, path, 'radars_rt', training, relative_path)
            radar_info['radars_pcd_path'] = get_radars_info_path(
                idx, path, 'radars_pcd', training, relative_path)
            radar_info['radars_ra_path'] = get_radars_info_path(
                idx, path, 'radars_ra', training, relative_path)
            
        image_info['image_path'] = get_image_path(idx, path, training,
                                                  relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            try:
                annotations = get_label_anno(label_path)
            except:
                print(f'{idx}-th file has error.')
        info['image'] = image_info
        info['radar'] = radar_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            calibration = np.load(calib_path,allow_pickle=True).item()

            calib_info['rvec'] = calibration['extrinsic']['rotation_vector']
            calib_info['tvec'] = calibration['extrinsic']['translation_vector']
            calib_info['cam_mat'] = calibration['intrinsic']['camera_matrix']
            calib_info['dist_coeffs'] = calibration['intrinsic']['distortion_coefficients']

            info['calib'] = calib_info

        if annotations is not None:
            info['annos'] = annotations

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)

