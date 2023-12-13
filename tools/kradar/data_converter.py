# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from collections import OrderedDict
from pathlib import Path

from mmdet3d.core.bbox import box_np_ops
from kradar_data_utils import get_kradar_image_info

kitti_categories = ('Pedestrian', 'Cyclist', 'Car')


def _read_imageset_file(path):
    # ./tools/train_test_splitter
    f = open(path, 'r')
    lines = f.readlines()
    f.close
    list_sample = []
    for line in lines:
        sample = line.split('\n')[0]
        list_sample.append(sample)
    return list_sample


def create_kradar_info_file(data_path,
                           pkl_prefix='kradar',
                           save_path=None,
                           relative_path=True,
                           workers=8):
    """Create info file of K-Radar dataset.

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
    kradar_infos_train = get_kradar_image_info(
        data_path,
        training=True,
        lidars=False,
        radars=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
        num_worker=workers)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'kradar info train file is saved to {filename}')
    mmcv.dump(kradar_infos_train, filename)
    kradar_infos_val = get_kradar_image_info(
        data_path,
        training=True,
        lidars=False,
        radars=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path,
        num_worker=workers)

    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'kradar info val file is saved to {filename}')
    mmcv.dump(kradar_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'kradar info trainval file is saved to {filename}')
    mmcv.dump(kradar_infos_train + kradar_infos_val, filename)

    kradar_infos_test = get_kradar_image_info(
        data_path,
        training=False,
        label_info=True,
        lidars=False,
        radars=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path,
        num_worker=workers)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'kradar info test file is saved to {filename}')
    mmcv.dump(kradar_infos_test, filename)

