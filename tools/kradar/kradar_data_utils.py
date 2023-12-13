# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import math
import cv2
import os
from scipy.io import loadmat
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from skimage import io
from scipy.spatial.transform import Rotation as R


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)


def get_kradar_info_path(idx,
                        prefix,
                        info_type='images',
                        file_tail='.png',
                        training=True,
                        relative_path=False,
                        exist_check=True,
                        use_prefix_id=False):
    if not isinstance(idx, str):
        img_idx_str = get_image_index_str(idx, use_prefix_id)
    else:
        img_idx_str = idx
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
                   info_type='cam-front',
                   use_prefix_id=False):
    return get_kradar_info_path(idx, prefix, info_type, '.png', training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='info_label',
                   use_prefix_id=False):
    return get_kradar_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_lidars_info_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_kradar_info_path(idx, prefix, 'os2-64', '.pcd', training,
                               relative_path, exist_check, use_prefix_id)


def get_radars_info_path(idx,
                      prefix,
                      modality='radar_zyx_cube',
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    assert modality in ['radar_zyx_cube', 'sp_rdr_cube'], "Modality not supported."
    if modality == 'radar_zyx_cube':
        file_tail = '.mat'
    elif modality == 'sp_rdr_cube':
        file_tail = '.npy'
    return get_kradar_info_path(idx, prefix, modality, file_tail, training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(idx,
                   prefix,
                   info_type='info_calib',
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):

    if not isinstance(idx, str):
        img_idx_str = get_image_index_str(idx, use_prefix_id)
    else:
        img_idx_str = idx
    file_tail = '.txt'
    img_idx_str += file_tail
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
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()

    rdr_idx, ldr_idx, camf_idx, _, _ = lines[0].split(',')[0].split('=')[1].split('_')
    content = [line.strip().split(', ') for line in lines[1:]]
    num_objects = len([x[0] for x in content if x[0]=='*'])

    if num_objects == 0: # if there is no object
        content = []
    annotations['name'] = np.array([x[3] for x in content if x[0]=='*'])
    annotations['truncated'] = np.array([0.00 for x in content if x[0]=='*'])
    annotations['occluded'] = np.array([0 for x in content if x[0]=='*'])
    annotations['alpha'] = np.array([float(0) for x in content if x[0]=='*'])
    annotations['bbox'] = np.array(
        [[50, 50, 150, 150] for x in content if x[0]=='*']).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content if x[0]=='*']).reshape(
            -1, 3)[:, [1, 2, 0]] * 2  # xcaml, ycaml, zcaml
    annotations['location'] = np.array(
        [[float(info) for info in x[4:7]] for x in content if x[0]=='*']).reshape(
            -1, 3) @ np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).T # xcam, ycam, zcam
    annotations['rotation_y'] = np.array(
        [float(x[7]) for x in content if x[0]=='*']).reshape(-1) * np.pi / 180
    annotations['score'] = np.zeros([len(annotations['bbox'])])

    return rdr_idx, ldr_idx, camf_idx, annotations

def get_calib_info(path_calib, is_z_offset_from_cfg=True):
    '''
    * return: [X, Y, Z]
    * if you want to get frame difference, get list_calib[0]
    '''
    with open(path_calib) as f:
        lines = f.readlines()
        f.close()
    list_calib = list(map(lambda x: float(x), lines[1].split(',')))
    # list_calib[0] # frame difference
    list_values = [list_calib[1], list_calib[2], 0.7] # X, Y, Z

    return np.array(list_values)

def get_description(path_desc):
    try:
        with open(path_desc) as f:
            line = f.readline()
        road_type, capture_time, climate = line.split(',')
        dict_desc = {
            'capture_time': capture_time,
            'road_type': road_type,
            'climate': climate,
        }
    except:
        raise FileNotFoundError(f'* Exception error (Dataset): check description {path_desc}')
    
    return dict_desc


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_kradar_image_info(path,
                         training=True,
                         label_info=True,
                         lidars=False,
                         radars=True,
                         calib=True,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True,
                         create_mono=True,
                         create_dense_pcd=True,
                         create_ra=True,):
    """
    kradar annotation format version 1:
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
    num_ids = list(range(len(image_ids)))
    list_roi_idx_cb = [70, 89, 184, 215, 0, 179]
    norm_val = 1e+13
    
    img_size = (1280,720)
    dict_values = {
        'fx':557.720776478944,
        'fy':567.2136917114258,
        'px':636.720776478944,
        'py':369.3068656921387,
        'k1':-0.028873818023371287,
        'k2':0.0006023302214797655,
        'k3':0.0039573086622276855,
        'k4':-0.005047176298643093,
        'k5':0.0,
        'roll_c':0.0,
        'pitch_c':0.0,
        'yaw_c':0.0,
        'roll_l':0.0,
        'pitch_l':0.7,
        'yaw_l':-0.5,
        'x_l':0.1,
        'y_l':0.0,
        'z_l':-0.7
    }
    tr_rotation_default = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    intrinsics = np.array([
        [dict_values['fx'], 0.0, dict_values['px']],
        [0.0, dict_values['fy'], dict_values['py']],
        [0.0, 0.0, 1.0]
    ])
    distortion = np.array([
        dict_values['k1'], dict_values['k2'], dict_values['k3'], \
        dict_values['k4'], dict_values['k5']
    ]).reshape((-1,1))
    yaw_c = dict_values['yaw_c']
    pitch_c = dict_values['pitch_c']
    roll_c = dict_values['roll_c']
    r_cam = (R.from_euler('zyx', [yaw_c, pitch_c, roll_c], degrees=True)).as_matrix()
    yaw_l = dict_values['yaw_l']
    pitch_l = dict_values['pitch_l']
    roll_l = dict_values['roll_l']
    r_l = (R.from_euler('zyx', [yaw_l, pitch_l, roll_l], degrees=True)).as_matrix()
    r_l = np.matmul(r_l, tr_rotation_default)
    x_l = dict_values['x_l']
    y_l = dict_values['y_l']
    z_l = dict_values['z_l']
    tr_lid_cam = np.concatenate([r_l, np.array([x_l,y_l,z_l]).reshape(-1,1)], axis=1)
    ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
    for j in range(3):
        for i in range(3):
            intrinsics[j,i] = ncm[j, i]
    map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, r_cam, ncm, img_size, cv2.CV_32FC1)
    org_P2 = np.insert(intrinsics, 3, values=[0,0,0], axis=1) # [3, 4]

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 9}
        radar_info = {}
        calib_info = {}
        annotations = None

        seq = image_ids[idx].split(',')[0]
        label = image_ids[idx].split(',')[1].split('.')[0]
        seq_path = osp.join(path, seq)

        desc_path = osp.join(seq_path, 'description.txt')
        desc = get_description(desc_path)
        for key in desc:
            info[key] = desc[key]

        if label_info:
            label_path = get_label_path(label, seq_path, training, relative_path)
            if relative_path:
                label_path = str(root_path / seq / label_path)
            try:
                rdr_idx, ldr_idx, camf_idx, annotations = get_label_anno(label_path)
            except:
                print(f'{idx}-th file has error.')
        else:
            print(f'label-info is required.')

        image_info = {'image_idx': idx}
        
        if lidars:
            ldr_idx = 'os2-64_' + ldr_idx
            pc_info['lidars_path'] = get_lidars_info_path(
                ldr_idx, seq_path, training, relative_path)
        if radars:
            ds_rdr_idx = 'cube_' + rdr_idx
            radar_info['ds_rdr_path'] = get_radars_info_path(
                ds_rdr_idx, seq_path, 'radar_zyx_cube', training, relative_path)
            sp_rdr_idx = 'spcube_' + rdr_idx
            radar_info['sp_rdr_path'] = get_radars_info_path(
                sp_rdr_idx, seq_path, 'sp_rdr_cube', training, relative_path)

        camf_idx = 'cam-front_' + camf_idx
        image_info['image_path'] = get_image_path(camf_idx, seq_path, training,
                                                  relative_path)
                                                  
        if create_mono:
            img_path = image_info['image_path']
            if relative_path:
                img_path = osp.join(seq_path, img_path)
            mono_img = cv2.imread(img_path)[:,:1280,[2,1,0]].copy()
            undist_img = cv2.remap(mono_img, map_x, map_y, cv2.INTER_LINEAR)
            if not osp.exists(osp.join(seq_path, 'images')):
                os.makedirs(osp.join(seq_path, 'images'))
            if not osp.exists(osp.join(seq_path, 'images', camf_idx + '.png')):
                cv2.imwrite(osp.join(seq_path, 'images', camf_idx + '.png'), undist_img)
            image_info['image_path'] = get_image_path(camf_idx, seq_path, training,
                                                relative_path, info_type='images')
        
        if create_dense_pcd:
            ds_rdr_path = radar_info['ds_rdr_path']
            ds_pcd_path = osp.join('radar_dense_pcd', 'dspcd_' + rdr_idx + '.npy')
            ds_pcd_parent_path = 'radar_dense_pcd'
            if relative_path:
                ds_rdr_path = osp.join(seq_path, ds_rdr_path)
                ds_pcd_path = osp.join(seq_path, ds_pcd_path)
                ds_pcd_parent_path = osp.join(seq_path, ds_pcd_parent_path)
            if not osp.exists(ds_pcd_parent_path):
                os.makedirs(ds_pcd_parent_path)

            zyx_cube = np.flip(loadmat(ds_rdr_path)['arr_zyx'], axis=0) # z-axis is flipped
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = list_roi_idx_cb
            zyx_cube = zyx_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
            zyx_cube = np.maximum(zyx_cube, 0.) / norm_val  # [20, 32, 180]

            Z, Y, X = zyx_cube.shape
            z_inds, y_inds, x_inds = np.where(zyx_cube > 0)
            pw = zyx_cube[z_inds, y_inds, x_inds]
            radar_pcd = np.stack([x_inds, y_inds, z_inds, pw], axis=1)
            radar_pcd[:, 0] = (radar_pcd[:, 0] - 0.5) / X * 71.6
            radar_pcd[:, 1] = (radar_pcd[:, 1] - 0.5) / Y * 12.4 - 6.4
            radar_pcd[:, 2] = (radar_pcd[:, 2] - 0.5) / Z * 7.6 - 2.0
            if not osp.exists(ds_pcd_path):
                np.save(ds_pcd_path, radar_pcd)
            radar_info['ds_pcd_path'] = osp.join('radar_dense_pcd', 'dspcd_' + rdr_idx + '.npy')
            
        if create_ra:
            ds_rdr_path = radar_info['ds_rdr_path']
            ra_path = osp.join('radar_ra', 'ra_' + rdr_idx + '.npy')
            ra_parent_path = 'radar_ra'
            if relative_path:
                ds_rdr_path = osp.join(seq_path, ds_rdr_path)
                ra_path = osp.join(seq_path, ra_path)
                ra_parent_path = osp.join(seq_path, ra_parent_path)
            if not osp.exists(ra_parent_path):
                os.makedirs(ra_parent_path)

            zyx_cube = np.flip(loadmat(ds_rdr_path)['arr_zyx'], axis=0) # z-axis is flipped
            zyx_cube = np.maximum(zyx_cube, 0.) / norm_val  # [20, 32, 180]

            Z, Y, X = zyx_cube.shape
            z_inds, y_inds, x_inds = np.where(zyx_cube > 0)
            pw = zyx_cube[z_inds, y_inds, x_inds]
            radar_pcd = np.stack([x_inds, y_inds, z_inds, pw], axis=1)
            radar_pcd[:, 0] = (radar_pcd[:, 0] - 0.5) / X * 100.0
            radar_pcd[:, 1] = (radar_pcd[:, 1] - 0.5) / Y * 160.0 - 80.0
            radar_pcd[:, 2] = (radar_pcd[:, 2] - 0.5) / Z * 60 - 30.0
            r = np.sqrt(radar_pcd[:, 0]**2 + radar_pcd[:, 1]**2)
            az = np.arctan2(radar_pcd[:, 1], radar_pcd[:, 0])
            z = radar_pcd[:, 2]

            polar_range = [0, -np.pi*20/180, -2.0, 72.0, np.pi*20/180, 6.0]
            polar_shape = [192, 64, 20]
            polar_tensor = np.zeros(polar_shape)
            mask = (r < polar_range[3]) & (r > polar_range[0]) & \
                (az < polar_range[4]) & (az > polar_range[1]) & \
                (z < polar_range[5]) & (z > polar_range[2])
            r = r[mask]
            az = az[mask]
            z = z[mask]
            pw = pw[mask]
            # plt.scatter(r, az, s=0.1, c=10*np.log10(pw), cmap='rainbow')
            r_idx = np.floor((r - polar_range[0]) / (polar_range[3] - polar_range[0]) * polar_shape[0]).astype(np.int32)
            az_idx = np.floor((az - polar_range[1]) / (polar_range[4] - polar_range[1]) * polar_shape[1]).astype(np.int32)
            z_idx = np.floor((z - polar_range[2]) / (polar_range[5] - polar_range[2]) * polar_shape[2]).astype(np.int32)
            polar_tensor[r_idx, az_idx, z_idx] = pw
            if not osp.exists(ra_path):
                np.save(ra_path, polar_tensor)

            radar_info['ra_path'] = osp.join('radar_ra', 'ra_' + rdr_idx + '.npy')
            

        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = osp.join(seq_path, img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
            if not create_mono:
                image_info['image_shape'][1] /= 2  # stereo image
        
        info['image'] = image_info
        info['radar'] = radar_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                'calib_radar_lidar', seq_path, training=training, relative_path=False)
            T_velo_to_radar = get_calib_info(calib_path)
            
            if annotations['location'].shape[0] > 0:
                # calib annotation to radar frame
                delta = T_velo_to_radar[None, :] @ np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).T
                annotations['location'] += delta

            P0 = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ])
            P1 = P0.copy()
            P2 = org_P2.copy()
            P3 = P0.copy()
            if extend_matrix:
                P0 = _extend_matrix(P0)  # [4, 4]
                P1 = _extend_matrix(P1)  # [4, 4]
                P2 = _extend_matrix(P2)  # [4, 4]
                P3 = _extend_matrix(P3)  # [4, 4]
            R0_rect = r_cam
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect  # [4, 4]
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = tr_lid_cam
            Tr_velo_to_radar = np.identity(3)
            Tr_velo_to_radar = np.insert(Tr_velo_to_radar, 3, values=T_velo_to_radar, axis=1)

            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_velo_to_radar = _extend_matrix(Tr_velo_to_radar)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_velo_to_radar'] = Tr_velo_to_radar
            info['calib'] = calib_info

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['label_file'] = image_ids[idx]

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, num_ids)

    return list(image_infos)

