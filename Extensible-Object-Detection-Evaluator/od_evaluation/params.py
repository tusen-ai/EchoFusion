from abc import ABC, abstractmethod
import numpy as np
from ipdb import set_trace

def waymo_range_breakdown(gt, pd, mode, params=None):
    assert mode in ('gt', 'pd')
    o = gt if mode == 'gt' else pd
    return np.linalg.norm(o['box'][:, :3], ord=2, axis=1)

def waymo_length_breakdown(gt, pd, mode, params=None):
    assert mode in ('gt', 'pd')
    o = gt if mode == 'gt' else pd
    return o['box'][:, 4]

def waymo_crowd_breakdown(gt, pd, mode, params=None):
    assert mode in ('gt', 'pd')

    if mode == 'gt':
        xyz = gt['box'][:, :2]
        types = np.unique(gt['type'])
        is_crowd = np.zeros(len(xyz), dtype=np.bool)
        for t in types:
            this_mask = gt['type'] == t
            this_xyz = xyz[this_mask, :]
            dist = np.linalg.norm(this_xyz[:, None, :] - this_xyz[None, :, :], axis=2, ord=2)
            is_close = dist < params.crowd_distance
            is_close = is_close.sum(1)
            this_is_crowd = is_close >= 2
            is_crowd[this_mask] = this_is_crowd
    else:
        if gt is None:
            is_crowd = np.zeros(len(pd['box']), dtype=bool)
        else:
            types = np.unique(np.concatenate([pd['type'], gt['type']]))
            gt_xyz = gt['box'][:, :2]
            pd_xyz = pd['box'][:, :2]
            is_crowd = np.zeros(len(pd['box']), dtype=bool)
            for t in types:
                gt_mask = gt['type'] == t
                pd_mask = pd['type'] == t
                if not pd_mask.any():
                    continue
                if not gt_mask.any():
                    is_crowd[pd_mask] = False
                    continue
                this_pd_xyz = pd_xyz[pd_mask]
                this_gt_xyz = gt_xyz[gt_mask]
                dist = np.linalg.norm(this_pd_xyz[:, None, :] - this_gt_xyz[None, :, :], axis=2, ord=2)
                is_close = dist < params.crowd_distance
                is_close = is_close.sum(1)
                is_crowd[pd_mask] = is_close >= 2
    return is_crowd

class BaseParam(ABC):
    def __init__(self, pd_path, gt_path, interval=1, update_sep=None, update_insep=None):
        self.pd_path = pd_path
        self.gt_path = gt_path

        self.add_breakdowns()
        self.add_iou_function()
        self.add_input_function()

        if update_sep is not None:
            self.separable_breakdowns.update(update_sep)
        if update_insep is not None:
            self.inseparable_breakdowns.update(update_insep)

        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)

        self.sampling_interval = interval
    
    @abstractmethod
    def add_breakdowns(self):
        pass

    @abstractmethod
    def add_iou_function(self):
        pass

    @abstractmethod
    def add_input_function(self):
        pass

class WaymoBaseParam(BaseParam):

    def __init__(self, pd_path, gt_path, interval=1, update_sep=None, update_insep=None):
        super().__init__(pd_path, gt_path, interval, update_sep, update_insep)
        self.iouThrs = [0.7, 0.5]
    
    def add_breakdowns(self):

        self.separable_breakdowns = {
            'type':('Vehicle', 'Pedestrian', 'Cyclist'), 
            'range':([0, 30], [30, 50], [50, 80], None), # None means the union of all ranges
        }
        self.breakdown_func_dict = {'range': waymo_range_breakdown}

        self.inseparable_breakdowns = {}
    
    def add_iou_function(self):
        from od_evaluation.ious import get_waymo_iou_matrix
        self.iou_calculate_func = get_waymo_iou_matrix

    def add_input_function(self):
        from od_evaluation.utils import get_waymo_object
        self.read_prediction_func = get_waymo_object
        self.read_groundtruth_func = get_waymo_object


class WaymoLengthParam(WaymoBaseParam):

    def __init__(self, pd_path, gt_path, length_range, interval=1, update_sep=None, update_insep=None):
        super().__init__(pd_path, gt_path, interval, update_sep, update_insep)
        self.inseparable_breakdowns['length'] = length_range
        self.breakdown_func_dict['length'] = waymo_length_breakdown

class WaymoCrowdParam(WaymoBaseParam):

    def __init__(self, pd_path, gt_path, dist=1.0, interval=1, update_sep=None, update_insep=None):
        super().__init__(pd_path, gt_path, interval, update_sep, update_insep)
        self.separable_breakdowns['crowd'] = [None, True, False]
        self.breakdown_func_dict['crowd'] = waymo_crowd_breakdown
        self.crowd_distance = dist


class RadialBaseParam(BaseParam):

    def __init__(self, pd_path, gt_path, save_folder, interval=1, iou_type='bev', update_sep=None, update_insep=None):
        assert iou_type in ['bev', '3d', 'let'], "Only supports iou of bev or 3d."
        self.iou_type = iou_type

        super().__init__(pd_path, gt_path, interval, update_sep, update_insep)
        self.save_folder = save_folder
        self.iouThrs = [0.7, 0.5]
    
    def add_breakdowns(self):

        self.separable_breakdowns = {
            'type':('Vehicle', 'Pedestrian', 'Cyclist'), 
            'range':([0, 50], [50, 100], None), # None means the union of all ranges
            # 'range': (None,),
        }
        self.breakdown_func_dict = {'range': waymo_range_breakdown}

        self.inseparable_breakdowns = {}
    
    def add_iou_function(self):
        from od_evaluation.ious import get_waymo_iou_matrix, get_waymo_iou_matrix_bev, get_let_iou_matrix_bev
        if self.iou_type == 'bev':
            self.iou_calculate_func = get_waymo_iou_matrix_bev
        elif self.iou_type == 'let':
            self.iou_calculate_func = get_let_iou_matrix_bev
        else:
            self.iou_calculate_func = get_waymo_iou_matrix

    def add_input_function(self):
        from od_evaluation.radial_utils import get_radial_pred_object, get_radial_gt_object, get_fft_radnet_pred_object, get_fft_radnet_gt_object
        self.read_prediction_func = get_radial_pred_object
        self.read_groundtruth_func = get_radial_gt_object

