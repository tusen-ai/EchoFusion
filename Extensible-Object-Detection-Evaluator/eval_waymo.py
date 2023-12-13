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

    from od_evaluation.params import WaymoLengthParam, WaymoCrowdParam
    update_sep = {'type':['Pedestrian']}

    # params = WaymoLengthParam(pd_path, gt_path, [None, [0, 4], [4, 8], [8, 20]], interval=args.interval, update_sep=update_sep)
    params = WaymoCrowdParam(pd_path, gt_path, 2, interval=args.interval, update_sep=update_sep)
    params.save_suffix = args.save_suffix

    evaluator = Evaluator(params, debug=False)
    evaluator.run()
