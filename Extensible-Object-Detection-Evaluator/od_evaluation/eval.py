import numpy as np
import numba
import copy
import tqdm
from collections import defaultdict
import datetime
from ipdb import set_trace
from treelib import Tree

import os
from os import path as osp
import pickle as pkl


class Evaluator(object):

    def __init__(self, params, debug=False):

        self.params = params
        self.debug = debug
        self.gt_path = params.gt_path
        self.pd_path = params.pd_path
        self.func_iou = params.iou_calculate_func
        self.func_bkd = params.breakdown_func_dict
        self.read_prediction = params.read_prediction_func
        self.read_groundtruth = params.read_groundtruth_func

        sep_bkd = self.params.separable_breakdowns
        insep_bkd = self.params.inseparable_breakdowns
        all_bkd = {**sep_bkd, **insep_bkd}
        self.all_breakdown_dict = all_bkd

        for bkd in self.func_bkd:
            assert bkd in all_bkd

        self.breakdown_name_list = list(sep_bkd.keys()) + list(insep_bkd.keys())

        self.sep_breakdown_name_list = list(sep_bkd.keys())
        self.insep_breakdown_name_list = list(insep_bkd.keys())

        self.num_choice_list = []
        for name in self.breakdown_name_list:
            self.num_choice_list.append(len(all_bkd[name]))

        self.num_all_choices = np.prod(self.num_choice_list)

        self.build_idx_breakdown_mapping()
        self.valid_ts_set = set()
        self.valid_ts_choice_set = set()
    
    def run(self):

        # sort by scores, calculate iou, add breakdown, assign bbox/gt ID.
        self._prepare()

        eval_list = []

        print('Evaluating ...')
        for sample_id in tqdm.tqdm(range(self.num_samples)):
            for i in range(self.num_all_choices):
                selected_sep_bkd, selected_insep_bkd, choice_dict = self.idx_to_choice[i]
                single_eval_result = self.evaluate_single_sample_given_breakdowns(selected_sep_bkd, selected_insep_bkd, sample_id, i)
                if single_eval_result is not None:
                    single_eval_result['choice_idx'] = i
                    single_eval_result['choice_dict'] = choice_dict
                    eval_list.append(single_eval_result)
        
        accumulated_result = self.accumulate(eval_list)
        summary = self.summarize_as_tree(accumulated_result)
        curves = self.summarize_curve(accumulated_result)
        self.save(summary, curves)

    def accumulate(self, eval_result_list):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating ...')
        p = self.params
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        breakdown_dims = self.num_choice_list # number of choice per breakdown, e.g., 3 choices of size breakdown: 'small', 'mid', 'large'

        precision   = -np.ones([T, R] + breakdown_dims) # -1 for the precision of absent categories
        fppi        = -np.ones([T, R] + breakdown_dims) # -1 for the precision of absent categories
        recall      = -np.ones([T,]   + breakdown_dims)
        scores      = -np.ones([T, R] + breakdown_dims)

        fppi        = -np.ones([T, R] + breakdown_dims)
        box_ious     = -np.ones([T,]   + breakdown_dims)
        box_errors  = -np.ones([T,]   + breakdown_dims + [self.box_dim+2,])

        eval_results_per_choice = self.split_eval_results_by_breakdowns(eval_result_list)

        for choice_idx in tqdm.tqdm(eval_results_per_choice):

            E = eval_results_per_choice[choice_idx]
            choice = self.idx_to_choice[choice_idx][-1] # e.g., {'size':0, 'range':2}
            choice_list = [choice[k] for k in self.breakdown_name_list] # for index

            num_images = len(E)
            assert num_images > 0
            # if len(E) == 0:
            #     continue
            dtScores = np.concatenate([e['pd_score'] for e in E])

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            sort_inds = np.argsort(-dtScores)
            dtScoresSorted = dtScores[sort_inds]

            pd_match  = np.concatenate([e['pd_match']  for e in E], axis=1)[:,sort_inds]
            pd_ignore = np.concatenate([e['pd_ignore'] for e in E], axis=1)[:,sort_inds]
            gt_ignore = np.concatenate([e['gt_ignore'] for e in E])

            this_ious  = np.concatenate([e['pd_matched_iou']  for e in E], axis=1)[:,sort_inds]
            pd_matched_gts  = np.concatenate([e['pd_matched_gt']  for e in E], axis=1)[:, sort_inds, :]
            pds = np.concatenate([e['pd_boxes']  for e in E], axis=0)[sort_inds, :]

            this_errors = np.abs(pds[None,...] - pd_matched_gts) # [num_thr, num_pds, box_dim]

            angle_diff = this_errors[..., 6] % np.pi
            this_errors[..., 6] = np.minimum(angle_diff, np.pi - angle_diff)
            translation_error = np.linalg.norm(np.expand_dims(pds[...,:3], axis=0) - pd_matched_gts[...,:3], axis=-1)
            
            size_error = np.linalg.norm(np.expand_dims(pds[...,3:6], axis=0) - pd_matched_gts[...,3:6], axis=-1)
            this_errors = np.concatenate([this_errors, translation_error[...,None], size_error[...,None]], axis=-1)

            assert gt_ignore.dtype == np.bool and pd_ignore.dtype == np.bool
            num_non_ig_gt = (~gt_ignore).sum()
            if num_non_ig_gt == 0:
                continue

            tps = (~pd_ignore) & (pd_match > -1)
            fps = (~pd_ignore) & (pd_match == -1)

            # only calculate tps errors
            this_ious = this_ious.astype(np.double) * tps.astype(np.double)
            this_errors = this_errors.astype(np.double) * tps.astype(np.double)[..., None]
            this_ious_sum = np.cumsum(this_ious, axis=1)
            this_errors_sum = np.cumsum(this_errors, axis=1)

            tp_sum = np.cumsum(tps, axis=1).astype(np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(np.float)

            # loop each iouthrs
            num_iou_thrs = len(self.params.iouThrs)
            for t in range(num_iou_thrs):
                tp = tp_sum[t, :]
                fp = fp_sum[t, :]
                nd = len(tp) # number of predictions
                rc = tp / num_non_ig_gt
                pr = tp / (fp + tp + np.spacing(1))
                # q  = np.zeros((R,), dtype=np.float)
                # ss = np.zeros((R,), dtype=np.float)
                # print(tp, fp, nd, rc, pr)

                indices = tuple([t,] + choice_list) # a[(1,2,3)] is equivalent to a[1,2,3]
                if nd:
                    recall[indices] = rc[-1]

                    ious_mean = this_ious_sum[t, ...] / (tp + np.spacing(1))
                    box_ious[indices] = ious_mean[-1]

                    errors_mean = this_errors_sum[t, ...] / (tp[:, None] + np.spacing(1))
                    box_errors[indices] = errors_mean[-1]
                else:
                    print(f'****** nd is {nd} ******')
                    recall[indices] = 0

                pr = pr.tolist()
                # q = q.tolist()

                # maybe use numba to speed it up
                for i in range(nd-1, 0, -1):
                    if pr[i] > pr[i-1]:
                        pr[i-1] = pr[i]

                assert len(rc) == len(pr) == len(dtScoresSorted)
                inds = np.searchsorted(rc, p.recThrs, side='left')
                for ri, pi in enumerate(inds):
                    indices = tuple([t, ri] + choice_list)
                    if pi < len(pr):
                        precision[indices] = pr[pi]
                        scores[indices] = dtScoresSorted[pi]
                        fppi[indices] = fp[pi] / num_images
                    else:
                        precision[indices] = 0
                        scores[indices] = 0
                        fppi[indices] = 0

        accumulated_result = {
            'params': p,
            'counts': [T, R, ] + breakdown_dims,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'box_ious': box_ious,
            'box_errors': box_errors,
            'fppi': fppi,
        }
        return accumulated_result

    def summarize_as_nested_dict(self, accumulated_result):
        print('Summarizing ...')
        p = self.params
        summary = {'Precision':{}, 'Recall':{}}
        def _summarize(choice_idx, iouThr=None, ap=1):
            _, _, choice =  self.idx_to_choice[choice_idx]
            choice_list = [choice[k] for k in self.breakdown_name_list] # for index
            pr_key = 'Precision' if ap == 1 else 'Recall'
            thr_key = f"IoU@{'Overall' if iouThr is None else iouThr}"
            temp_dict = summary[pr_key]
            if thr_key not in temp_dict:
                temp_dict[thr_key] = {}
            temp_dict = temp_dict[thr_key]

            if ap == 1:
                s = accumulated_result['precision']
                # IoU
                if iouThr is not None:
                    t = p.iouThrs.index(iouThr)
                    s = s[[t,]]
                num_bkd = len(choice_list)
                for i in choice_list:
                    s = s[:, :, i]
                assert s.ndim == 2 # only left dimensions of iouThrs and recall

            else:
                s = accumulated_result['recall']
                if iouThr is not None:
                    t = p.iouThrs.index(iouThr)
                    s = s[[t,]]
                for i in choice_list:
                    s = s[:, i]
                assert s.ndim == 1

            if len(s[s > -1])==0:
                mean_s = np.array([-1]).item()
            else:
                mean_s = np.mean(s[s > -1]).item()
            
            # similar to DFS
            for i, name in enumerate(self.breakdown_name_list):
                value = str(self.all_breakdown_dict[name][choice[name]])
                this_key = f'{name}_{value}'
                if this_key not in temp_dict:
                    if i == len(self.breakdown_name_list) - 1:
                        if mean_s != -1:
                            temp_dict[this_key] = mean_s
                    else:
                        temp_dict[this_key] = {}
                        temp_dict = temp_dict[this_key]
                else:
                    temp_dict = temp_dict[this_key]
                    assert isinstance(temp_dict, dict)


        for i in range(len(p.iouThrs) + 1):
            if i == len(p.iouThrs) == 1:
                break
            iou_thr = p.iouThrs[i] if i < len(p.iouThrs) else None
            for i in range(self.num_all_choices):
                _summarize(i, iou_thr, ap=1)

        return summary
    
    def summarize_curve(self, accumulated_result):
        summary_dict = {}
        p = self.params
        def _summarize(choice_idx, metric, iouThr=None):
            _, _, choice =  self.idx_to_choice[choice_idx]
            choice_list = [choice[k] for k in self.breakdown_name_list] # for index

            if metric == 'fppi':
                s = accumulated_result['fppi']
                # IoU
                if iouThr is None:
                    return
                t = p.iouThrs.index(iouThr)
                s = s[t, ...]
                num_bkd = len(choice_list)
                for i in choice_list:
                    s = s[:, i]
                assert s.ndim == 1 # only left dimensions of iouThrs and recall
            elif metric == 'precision':
                s = accumulated_result['precision']
                # IoU
                if iouThr is None:
                    return
                t = p.iouThrs.index(iouThr)
                s = s[t, ...]
                num_bkd = len(choice_list)
                for i in choice_list:
                    s = s[:, i]
                assert s.ndim == 1 # only left dimensions of iouThrs and recall
            else:
                raise NotImplementedError

            
            # similar to DFS
            key = f'IoU@{iouThr}'
            for i, name in enumerate(self.breakdown_name_list):
                value = str(self.all_breakdown_dict[name][choice[name]])
                this_key = f'{name}_{value}'
                if 'None' in this_key:
                    this_key = this_key.replace('None', 'OVERALL')
                key = f'{key}/{this_key}'

            summary_dict[key] = s

        for iou_thr in p.iouThrs:
            for i in range(self.num_all_choices):
                _summarize(i, 'precision', iou_thr)

        return summary_dict

    def summarize_as_tree(self, accumulated_result):
        print('Summarizing ...')
        p = self.params
        metric_set = ['Precision', 'Recall', 'TP_IoU', 'TP_Error']
        summary = Tree()
        summary.create_node('Results', 'root')
        summary.create_node('Precision', 'Precision', parent='root')
        summary.create_node('Recall', 'Recall', parent='root')
        summary.create_node('TP_IoU', 'TP_IoU', parent='root')
        summary.create_node('TP_Error', 'TP_Error', parent='root')
        # summary = {'Precision':{}, 'Recall':{}}
        def _summarize(choice_idx, metric, iouThr=None):
            assert metric in metric_set
            _, _, choice =  self.idx_to_choice[choice_idx]
            choice_list = [choice[k] for k in self.breakdown_name_list] # for index
            par = metric
            thr_name = f"IoU@{'Overall' if iouThr is None else iouThr}"
            iden = par + thr_name
            if iden not in summary:
                summary.create_node(thr_name, iden, parent=par)
            par = iden

            if metric == 'Precision':
                s = accumulated_result['precision']
                # IoU
                if iouThr is not None:
                    t = p.iouThrs.index(iouThr)
                    s = s[[t,]]
                num_bkd = len(choice_list)
                for i in choice_list:
                    s = s[:, :, i]
                assert s.ndim == 2 # only left dimensions of iouThrs and recall
                if len(s[s > -1]) == 0:
                    mean_s = None
                else:
                    mean_s = '{:.4f}'.format(np.mean(s[s > -1]).item() * 100)

            elif metric == 'Recall':
                s = accumulated_result['recall']
                if iouThr is not None:
                    t = p.iouThrs.index(iouThr)
                    s = s[[t,]]
                for i in choice_list:
                    s = s[:, i]
                assert s.ndim == 1

                if len(s[s > -1]) == 0:
                    mean_s = None
                else:
                    mean_s = '{:.4f}'.format(np.mean(s[s > -1]).item() * 100)

            elif metric == 'TP_IoU':
                s = accumulated_result['box_ious']
                if iouThr is not None:
                    t = p.iouThrs.index(iouThr)
                    s = s[[t,]]
                for i in choice_list:
                    s = s[:, i]
                assert s.ndim == 1
                if len(s[s > -1]) == 0:
                    mean_s = None
                else:
                    mean_s = '{:.4f}'.format(np.mean(s[s > -1]).item())

            elif metric == 'TP_Error':
                s = accumulated_result['box_errors']
                if iouThr is not None:
                    t = p.iouThrs.index(iouThr)
                    s = s[[t,]]
                for i in choice_list:
                    s = s[:, i]
                assert s.ndim == 2

                if len(s[s > -1]) == 0:
                    mean_s = None
                else:
                    s = s.mean(0).reshape(-1).tolist()
                    assert len(s) == self.box_dim+2
                    mean_s = '['
                    for i in range(len(s)):
                        mean_s += '{:.4f}, '.format(s[i])
                    mean_s += ']'
            
            else:
                raise NotImplementedError

            
            # similar to DFS
            for i, name in enumerate(self.breakdown_name_list):
                value = str(self.all_breakdown_dict[name][choice[name]])
                this_key = f'{name}_{value}'
                if 'None' in this_key:
                    this_key = this_key.replace('None', 'OVERALL')
                iden = par + this_key
                if iden not in summary:
                    if i == len(self.breakdown_name_list) - 1:
                        if mean_s is not None:
                            summary.create_node(this_key + ': ' + mean_s, iden, parent=par)
                    else:
                        summary.create_node(this_key, iden, parent=par)
                        par = iden
                else:
                    par = iden


        for i in range(len(p.iouThrs) + 1):
            iou_thr = p.iouThrs[i] if i < len(p.iouThrs) else None
            if i == len(p.iouThrs) == 1:
                break
            for i in range(self.num_all_choices):
                _summarize(i, 'Precision', iou_thr)
                _summarize(i, 'Recall', iou_thr)
                _summarize(i, 'TP_IoU', iou_thr)
                _summarize(i, 'TP_Error', iou_thr)

        return summary
        
    
    def split_eval_results_by_breakdowns(self, eval_results_list):
        results_dict = defaultdict(list)
        for r in eval_results_list:
            if r is not None:
                choice_idx = r['choice_idx']
                results_dict[choice_idx].append(r)
        # sort the list the results_dict by timestamp
        for k, v in results_dict.items():
            results_dict[k] = sorted(v, key=lambda x: x['timestamp'])

        assert len(results_dict) <= self.num_all_choices
        return results_dict

    
    def build_idx_breakdown_mapping(self):

        idx_to_choice = {}
        loop_check_set = set()

        for idx in range(self.num_all_choices):
            choice_dict = {}
            choice_list = []

            for i, key in enumerate(self.breakdown_name_list):

                if i == 0:
                    choice_idx = idx % self.num_choice_list[i]
                else: 
                    choice_idx = (idx // np.prod(self.num_choice_list[0:i])) % self.num_choice_list[i]

                choice_dict[key] = choice_idx
                choice_list.append(choice_idx)

            loop_check_set.add(tuple(choice_list))
            
            selected_sep_breakdown = {}
            selected_insep_breakdown = {}
            for name in self.sep_breakdown_name_list:
                selected_sep_breakdown[name] = self.params.separable_breakdowns[name][choice_dict[name]]

            for name in self.insep_breakdown_name_list:
                selected_insep_breakdown[name] = self.params.inseparable_breakdowns[name][choice_dict[name]]

            idx_to_choice[idx] = (selected_sep_breakdown, selected_insep_breakdown, choice_dict)

        self.idx_to_choice = idx_to_choice

        # sanity check 
        assert len(loop_check_set) == self.num_all_choices
    

    def evaluate_single_sample_given_breakdowns(self, sep_bd_dict, insep_bd_dict, sample_id, choice_idx):

        gt_attrs = copy.deepcopy(self.gts_list[sample_id])
        pd_attrs = copy.deepcopy(self.pds_list[sample_id])

        if None in (gt_attrs, pd_attrs): # Is it right to ignore fps if there is no gts.
            return None

        gt_timestamp = gt_attrs['timestamp']
        pd_timestamp = pd_attrs['timestamp']
        assert gt_timestamp == pd_timestamp

        # select valid gt & pd by seperable breakdowns
        pd_attrs, gt_attrs = self.split_by_sep_breakdowns(pd_attrs, gt_attrs, sep_bd_dict)

        if len(gt_attrs['type']) == 0 or len(pd_attrs['type']) == 0: # assume there is a 'type' breakdown
            return None
        
        self.valid_ts_set.add(gt_timestamp)
        self.valid_ts_choice_set.add((gt_timestamp, choice_idx))

        # ignore some gt by the given inseparable breakdown
        gt_ignore = self.get_gt_ignore(gt_attrs, insep_bd_dict)

        # sort gt and pd
        pd_attrs, gt_attrs, gt_ignore = self.sort_by_score(pd_attrs, gt_attrs, gt_ignore)
        
        # update valid gt, pd, iou
        valid_ious = self.func_iou(pd_attrs['box'], gt_attrs['box'])
        # print(pd_attrs['box'], gt_attrs['box'], valid_ious)
        # set_trace()

        num_gts = len(gt_attrs['type'])
        num_pds = len(pd_attrs['type'])

        # init results
        num_thrs = len(self.params.iouThrs)
        gt_match  = np.zeros((num_thrs, num_gts), dtype=np.int64)
        pd_match  = -np.ones((num_thrs, num_pds), dtype=np.int64)
        pd_ignore = np.zeros((num_thrs, num_pds), dtype=np.bool)

        if not hasattr(self, 'box_dim'):
            self.box_dim = gt_attrs['box'].shape[1]
        pd_matched_gts  = -np.ones((num_thrs, num_pds, self.box_dim), dtype=np.float)
        pd_matched_iou  = -np.ones((num_thrs, num_pds), dtype=np.float)

        # matching
        pd_ids = pd_attrs['ID']
        gt_ids = gt_attrs['ID']
        pd_ignore, pd_match, gt_match = matching(
            valid_ious,
            self.params.iouThrs,
            pd_ids,
            gt_ids,
            gt_ignore,
            pd_ignore,
            gt_match,
            pd_match
        ) 

        for i in range(num_thrs):

            this_pd_match = pd_match[i]
            pd_mask = this_pd_match > -1

            if pd_mask.any():
                pd_matched_iou[i, pd_mask] = valid_ious[pd_mask, this_pd_match[pd_mask]]
                assert (pd_matched_iou[i, pd_mask] > self.params.iouThrs[i] - 1e-5).all()

            pd_matched_gts[i, pd_mask, :] = gt_attrs['box'][this_pd_match[pd_mask]]
        


        # Deal with those inseperable breakdowns.
        pd_ignore = self.update_pd_ignore_from_insep_breakdown(pd_match, pd_ignore, pd_attrs, insep_bd_dict)


        results = dict(
            sample_id=sample_id,
            pd_ids=pd_ids,
            gt_ids=gt_ids,
            pd_match=pd_match,
            gt_match=gt_match,
            pd_ignore=pd_ignore,
            gt_ignore=gt_ignore,
            pd_score=pd_attrs['score'],
            pd_matched_iou=pd_matched_iou,
            pd_matched_gt=pd_matched_gts,
            pd_boxes=pd_attrs['box'],
            breakdowns={**sep_bd_dict, **insep_bd_dict},
            timestamp=pd_timestamp,
        )

        return results

    
    def split_by_sep_breakdowns(self, pd_attrs, gt_attrs, bd_dict):
        gt_attrs = gt_attrs.copy()
        pd_attrs = pd_attrs.copy()
        num_gts = len(gt_attrs['type']) # assume there is a 'type' breakdown
        num_pds = len(pd_attrs['type']) # assume there is a 'type' breakdown

        valid_gt_mask = np.ones(num_gts, dtype=np.bool) # all true at first
        valid_pd_mask = np.ones(num_pds, dtype=np.bool)

        for bd, bd_value in bd_dict.items():
            if isinstance(bd_value, (list, tuple)):
                assert bd_value[1] > bd_value[0]
                valid_gt_mask = valid_gt_mask & (gt_attrs[bd] > bd_value[0]) & (gt_attrs[bd] < bd_value[1])
                valid_pd_mask = valid_pd_mask & (pd_attrs[bd] > bd_value[0]) & (pd_attrs[bd] < bd_value[1])
            elif bd_value is None:
                continue # None means using all ranges
            else:
                valid_gt_mask = valid_gt_mask & (gt_attrs[bd] == [bd_value])
                valid_pd_mask = valid_pd_mask & (pd_attrs[bd] == [bd_value])
        
        for key in pd_attrs:
            value = pd_attrs[key]
            if isinstance(value, np.ndarray):
                pd_attrs[key] = value[valid_pd_mask]

        for key in gt_attrs:
            value = gt_attrs[key]
            if isinstance(value, np.ndarray):
                gt_attrs[key] = value[valid_gt_mask]
        return pd_attrs, gt_attrs

    def get_gt_ignore(self, gt_attrs, bd_dict):

        if 'ignore' in gt_attrs:
            gt_ignore = gt_attrs['ignore']
        else:
            gt_ignore = np.zeros(len(gt_attrs['box']), dtype=np.bool)

        num_gts = len(gt_ignore)
        assert (~gt_ignore).all() # no ignore at first

        for bd, bd_value in bd_dict.items():
            if isinstance(bd_value, (list, tuple)):
                assert bd_value[1] > bd_value[0]
                gt_ignore = gt_ignore | (gt_attrs[bd] < bd_value[0]) | (gt_attrs[bd] > bd_value[1])
            elif bd_value is None:
                continue
            else:
                gt_ignore = gt_ignore | (gt_attrs[bd] != bd_value)

        return gt_ignore
    
    def update_pd_ignore_from_insep_breakdown(self, pd_match, pd_ignore, pd_attrs, bd_dict):

        for bd, bd_value in bd_dict.items():
            pd_attr = pd_attrs[bd]

            if isinstance(bd_value, (list, tuple)):
                assert bd_value[1] > bd_value[0]
                a = (pd_attr < bd_value[0]) | (pd_attr > bd_value[1])
            elif bd_value is None:
                continue
            else:
                a = pd_attr == bd_value

            pd_ignore = pd_ignore | ((pd_match == -1) & a[None, :])
            # pd_ignore = pd_ignore | ((pd_match == 0) & a[None, :])

        return pd_ignore
    
    def sort_by_score(self, pd_attrs, gt_attrs, gt_ignore):

        pd_inds = np.argsort(-1 * pd_attrs['score'])

        for key in pd_attrs:
            value = pd_attrs[key]
            if isinstance(value, np.ndarray):
                pd_attrs[key] = value[pd_inds]

        gt_inds = np.argsort(gt_ignore.astype(np.float))

        for key in gt_attrs:
            value = gt_attrs[key]
            if isinstance(value, np.ndarray):
                gt_attrs[key] = value[gt_inds]
        
        gt_ignore = gt_ignore[gt_inds]
        
        return pd_attrs, gt_attrs, gt_ignore
    
    def _prepare(self):
        # add breakdown, assign bbox/gt ID.
        pds_dict = self.read_prediction(self.pd_path)
        gts_dict = self.read_groundtruth(self.gt_path)
        with open("results.pkl", "wb") as fw:
            pkl.dump(pds_dict, fw)
            pkl.dump(gts_dict, fw)
        pds_list, gts_list = self.align_samples(pds_dict, gts_dict)
        num_samples = len(pds_list)
        assert num_samples == len(gts_list)
        self.num_samples = num_samples

        total_pd_num = 0
        total_gt_num = 0

        print(f'Got {len(pds_list)} predicted samples.')

        print('Start to assign object IDs and add calculate breakdowns ...')

        for i in tqdm.tqdm(range(num_samples)):
            pd = pds_list[i]
            gt = gts_list[i]
            if pd is not None:
                num_pd = len(pd['type'])
                pd['ID'] = total_pd_num + np.arange(num_pd)
            else:
                num_pd = 0

            if gt is not None:
                num_gt = len(gt['type'])
                gt['ID'] = total_gt_num + np.arange(num_gt)
            else:
                num_gt = 0

            total_pd_num += num_pd
            total_gt_num += num_gt

            for bkd_name, bkd_func in self.func_bkd.items():
                pd[bkd_name] = bkd_func(pd=pd, gt=gt, mode='pd', params=self.params)
                gt[bkd_name] = bkd_func(pd=pd, gt=gt, mode='gt', params=self.params)

        self.pds_list = pds_list
        self.gts_list = gts_list
        self.total_pd_num = total_pd_num
        self.total_gt_num = total_gt_num
        print(f'Got {total_pd_num} predicted objects')
        print(f'Got {total_gt_num} ground-truth objects')
    
    def align_samples(self, pds_dict, gts_dict):
        gts_keys = set(gts_dict.keys())
        sorted_gts_keys = sorted(list(gts_keys))
        interval = self.params.sampling_interval
        sub_gt_keys = [k for i, k in enumerate(sorted_gts_keys) if i % interval == 0]
        all_keys = sub_gt_keys
        self.all_keys = all_keys
        pds_list, gts_list = [], []
        for k in all_keys:
            pds_list.append(pds_dict.get(k, None))
            gts_list.append(gts_dict[k])
        return pds_list, gts_list
    
    def save(self, summary, raw_result):


        if hasattr(self.params, 'save_folder'):
            save_folder = self.params.save_folder
        else:
            pd_dir = osp.dirname(osp.abspath(self.params.pd_path))
            save_folder = osp.join(pd_dir, 'evaluation_results')

        os.makedirs(save_folder, exist_ok=True)

        if hasattr(self.params, 'save_suffix'):
            suffix = '-' + self.params.save_suffix
        else:
            suffix = ''

        date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        file_name = f'{date}{suffix}.txt'
        save_path = osp.join(save_folder, file_name)
        summary.save2file(save_path)

        pkl_path = save_path.replace('.txt', '.pkl')
        with open(pkl_path, 'wb') as fw:
            pkl.dump(raw_result, fw)

        print(f'Evaluation result saved to {save_path}')


# @numba.jit(nopython=True)
def matching(ious, iouThrs, pd_ids, gt_ids, gt_ignore, pd_ignore, gt_match, pd_match):
    num_pds = ious.shape[0]
    num_gts = ious.shape[1]
    for tind, t in enumerate(iouThrs):
        for pd_ind in range(num_pds):
            # information about best match so far (m=-1 -> unmatched)
            iou = min([t, 1 - 1e-10])
            m = -1 # m is a temp gt index
            for gt_ind in range(num_gts):
                # if this gt already matched, and not a crowd, continue
                if gt_match[tind, gt_ind] > 0:
                    continue
                # if dt matched to a gt, and on ignore gt, stop; This requires all ignore gts are sorted last.
                if m > -1 and gt_ignore[m] == 0 and gt_ignore[gt_ind] == 1:
                    break
                # continue to next gt unless better match made
                if ious[pd_ind, gt_ind] < iou:
                    continue
                # if match successful and best so far, store appropriately
                iou = ious[pd_ind, gt_ind]
                m = gt_ind
            # if match made store id of match for both dt and gt
            if m == -1:
                continue

            pd_ignore[tind, pd_ind] = gt_ignore[m]
            # pd_match[tind, pd_ind]  = gt_ids[m]
            # using m seems more useful than using gt_id
            pd_match[tind, pd_ind]  = m
            gt_match[tind, m] = pd_ids[pd_ind]

    return pd_ignore, pd_match, gt_match




