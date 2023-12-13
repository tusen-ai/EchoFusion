# evaluate the waymo xx.bin files in COCO-like protocal
from od_evaluation.eval import Evaluator
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pd-path', type=str, default='./results_3d.npy')
parser.add_argument('--gt-path', type=str, default='./data/radial_kitti_format/radialx_infos_test.pkl')
parser.add_argument('--save-folder', type=str, default='')
parser.add_argument('--save-suffix', type=str, default='')
parser.add_argument('--interval', type=int, default=10)
args = parser.parse_args()

if __name__ == '__main__':
    pd_path = args.pd_path
    gt_path = args.gt_path 
    save_folder = args.save_folder
    suffix = args.save_suffix

    from od_evaluation.params import RadialBaseParam, WaymoLengthParam, WaymoCrowdParam
    update_sep = {'type':['Vehicle']}

    # params = WaymoLengthParam(pd_path, gt_path, [None, [0, 4], [4, 8], [8, 20]], interval=args.interval, update_sep=update_sep)
    # params = WaymoCrowdParam(pd_path, gt_path, 2, interval=args.interval, update_sep=update_sep)

    print('Begin BEV Evaluation...\n')
    params = RadialBaseParam(pd_path, gt_path, save_folder, iou_type='bev', interval=1, update_sep=update_sep)
    params.save_suffix = 'bev' + '_' + suffix

    evaluator = Evaluator(params, debug=False)
    evaluator.run()

    print('Begin 3D Evaluation...\n')
    params = RadialBaseParam(pd_path, gt_path, save_folder, iou_type='3d', interval=1, update_sep=update_sep)
    params.save_suffix = '3d' + '_' + suffix

    evaluator = Evaluator(params, debug=False)
    evaluator.run()
    
    print('Begin LET Evaluation...\n')
    params = RadialBaseParam(pd_path, gt_path, save_folder, iou_type='let', interval=1, update_sep=update_sep)
    params.save_suffix = 'let' + '_' + suffix

    evaluator = Evaluator(params, debug=False)
    evaluator.run()

    result = np.load(pd_path, allow_pickle=True)
    save_path = os.path.join(save_folder, 'results_' + suffix + '.npy')
    np.save(save_path, result)
    print(f'results saved at {save_path}')
