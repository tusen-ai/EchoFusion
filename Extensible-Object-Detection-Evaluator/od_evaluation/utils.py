import os
import numpy as np
from waymo_open_dataset.protos import metrics_pb2
import tqdm
from collections import defaultdict
from ipdb import set_trace

def read_bin(file_path):
    with open(file_path, 'rb') as f:
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
    return objects


def waymo_object_to_mmdet(obj, version, debug=False, gt=False):
    '''
    According to https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto#L33
    and the definition of LiDARInstance3DBoxes
    '''
    # if gt and obj.object.num_lidar_points_in_box == 0:
    #     print('Encounter zero-point object')
    #     return None
    box = obj.object.box


    assert version < '1.0.0', 'Only support version older than 1.0.0 for now'
    heading = -box.heading - 0.5 * np.pi

    while heading < -np.pi:
        heading += 2 * np.pi
    while heading > np.pi:
        heading -= 2 * np.pi

    result = np.array(
        [
            box.center_x,
            box.center_y,
            box.center_z,
            box.width,
            box.length,
            box.height,
            heading,
            obj.score,
            float(obj.object.type),
        ]
    )
    return result

def get_waymo_object(file_path, debug=False, gt=False):
    import mmdet3d
    mmdet3d_version = mmdet3d.__version__
    print(f'Reading {file_path} ...')
    data = read_bin(file_path)
    objects = data.objects
    obj_dict = defaultdict(list)
    print('Collecting Bboxes ...')
    for o in tqdm.tqdm(objects):
        seg_name = o.context_name
        time_stamp = o.frame_timestamp_micros
        mm_obj = waymo_object_to_mmdet(o, mmdet3d_version, debug, gt)
        if mm_obj is not None:
            obj_dict[time_stamp].append(mm_obj)
    
    new_dict = {}

    sorted_keys = sorted(list(obj_dict.keys()))

    for k in sorted_keys:
        sample = np.stack(obj_dict[k], axis=0)
        num_obj = len(sample)
        assert num_obj > 0
        boxes = sample[:, :7]
        scores = sample[:, 7]
        types = np.zeros(num_obj, dtype='<U32')
        for name, cls_id in zip(['Vehicle', 'Pedestrian', 'Cyclist'], [1, 2, 4]):
            this_mask = sample[:, 8] == cls_id
            types[this_mask] = name
        new_dict[k] = dict(
            box=boxes,
            score=scores,
            type=types,
            timestamp=k,
        )
        
    return new_dict

