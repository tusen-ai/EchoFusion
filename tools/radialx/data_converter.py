# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from collections import OrderedDict
from pathlib import Path

from mmdet3d.core.bbox import box_np_ops
from radial_data_utils import get_radial_image_info

kitti_categories = ('Pedestrian', 'Cyclist', 'Car')


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def create_radial_info_file(data_path,
                           pkl_prefix='radialx',
                           save_path=None,
                           relative_path=True,
                           workers=8):
    """Create info file of RADIal dataset.

    Given the kitti format data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    radial_infos_train = get_radial_image_info(
        data_path,
        training=True,
        lidars=True,
        radars=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
        num_worker=workers)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'RADIal info train file is saved to {filename}')
    mmcv.dump(radial_infos_train, filename)
    radial_infos_val = get_radial_image_info(
        data_path,
        training=True,
        lidars=True,
        radars=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path,
        num_worker=workers)

    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'RADIal info val file is saved to {filename}')
    mmcv.dump(radial_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'RADIal info trainval file is saved to {filename}')
    mmcv.dump(radial_infos_train + radial_infos_val, filename)

    radial_infos_test = get_radial_image_info(
        data_path,
        training=False,
        label_info=True,
        lidars=True,
        radars=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path,
        num_worker=workers)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'RADIal info test file is saved to {filename}')
    mmcv.dump(radial_infos_test, filename)

