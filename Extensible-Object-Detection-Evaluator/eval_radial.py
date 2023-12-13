# evaluate the waymo xx.bin files in COCO-like protocal
from od_evaluation.eval import Evaluator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pd-path', type=str, default='')
parser.add_argument('--gt-path', type=str, default='')
parser.add_argument('--save-folder', type=str, default='')
parser.add_argument('--save-suffix', type=str, default='')
parser.add_argument('--interval', type=int, default=10)
args = parser.parse_args()

if __name__ == '__main__':
    pd_path = args.pd_path
    gt_path = args.gt_path 

    # pd_path = '/mnt/truenas/scratch/yang.liu3/Python/RadarFormer/PolarFormer/results_img_12e.npy'
    pd_path = '/mnt/truenas/scratch/yang.liu3/Python/RadarFormer/PolarFormer/results_img_12e.npy'
    # pd_path = '/mnt/truenas/scratch/feng.wang/training/RadarFormer/PolarFormer/RADIal_main/FFTRadNet_rbbox/predictions.pkl'
    gt_path = '/mnt/weka/scratch/yang.liu3/pyworkspace/EchoFusion/data/radial_kitti_format/radar_anno.pkl'

    from od_evaluation.params import RadialBaseParam, WaymoLengthParam, WaymoCrowdParam
    update_sep = {'type':['Vehicle']}

    # params = WaymoLengthParam(pd_path, gt_path, [None, [0, 4], [4, 8], [8, 20]], interval=args.interval, update_sep=update_sep)
    # params = WaymoCrowdParam(pd_path, gt_path, 2, interval=args.interval, update_sep=update_sep)
    params = RadialBaseParam(pd_path, gt_path, interval=1, update_sep=update_sep)
    params.save_suffix = args.save_suffix

    evaluator = Evaluator(params, debug=False)
    evaluator.run()
