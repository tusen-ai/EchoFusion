import torch
import os
import json
import numpy as np
import pkbar
import argparse
from mmcv import ProgressBar
from shapely.geometry import Polygon
from shapely.ops import unary_union

def RA_to_cartesian_box(data):
    """Used for radar coordinates"""
    L = 4
    W = 1.8
    
    boxes = []
    for i in range(len(data)):
        
        x = np.cos(np.radians(data[i][1])) * data[i][0]
        y = np.sin(np.radians(data[i][1])) * data[i][0]

        boxes.append([x-L/2, y-W/2, x-L/2, y+W/2, x+L/2, y+W/2, x+L/2, y-W/2])
             
    return boxes

def get_rotated_box(data):
    """Used for radar coordinates"""
    results = []
    for i in range(data.shape[0]):
        l = data[i, 3]
        w = data[i, 4]
        r = data[i, 6]
        loc = data[i, :2].copy()
        R = np.array([[np.cos(r), -np.sin(r)],
                    [np.sin(r), np.cos(r)]])
        corners = np.array([[-l / 2, -l / 2, l / 2, l / 2],
                            [-w / 2, w / 2, w / 2, -w / 2]])
        corners_rot = np.dot(R, corners) + loc[None, :].T
        results.append(np.transpose(corners_rot, (1, 0)).reshape(-1).tolist())

    return results

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :], sorted_box_predictions[i + 1:, :])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions
def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious

def process_predictions(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions
    
    point_cloud_reg_predictions = get_rotated_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:,-1]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
    
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]

    
    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                 nms_threshold)

    
    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_Object_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))


    return final_Object_predictions

def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions
    
    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:,-1]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
    
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]

    
    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                 nms_threshold)

    
    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_Object_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))


    return final_Object_predictions

def GetFullMetrics(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    """Used for bev evaluation in RadarFormer
    
    Args:
        predictions: list of array. Each array is in shape of [N, 2].
        object_labels: list of array. The i-th array is in shape of [M_i, 2].
        range_min: float. The minimum range of perception.
        range_max: float. The maximum range of perception.
        IOU_threshold: float. The threshold of IoU for matching.

    Returns:
        results: The result in string.
        ret_dict: dict. The detailed results.
    """
    perfs = {}
    precision = []
    recall = []
    iou_threshold = []
    RangeError = []
    AngleError = []

    out = []

    print("Evaluating ...")
    prog_bar = ProgressBar(len(np.arange(0.1,0.96,0.1)) * len(predictions))
    for threshold in np.arange(0.1,0.96,0.1):

        iou_threshold.append(threshold)

        TP = 0
        FP = 0
        FN = 0
        NbDet = 0
        NbGT = 0
        NBFrame = 0
        range_error=0
        angle_error=0
        nbObjects = 0

        for frame_id in range(len(predictions)):

            pred = predictions[frame_id]
            labels = object_labels[frame_id]

            pred_r = np.linalg.norm(pred[:, :2], axis=1)
            pred_a = np.arctan2(pred[:, 1], pred[:, 0]) * 180 / np.pi
            labels_r = np.linalg.norm(labels[:, :2], axis=1)
            labels_a = np.arctan2(labels[:, 1], labels[:, 0]) * 180 / np.pi

            # get final bounding box predictions
            Object_predictions = []
            ground_truth_box_corners = []           
            
            if(len(pred)>0):
                Object_predictions = process_predictions(pred,confidence_threshold=threshold)

            if(len(Object_predictions)>0):
                max_distance_predictions = (Object_predictions[:,1]+Object_predictions[:,3])/2
                ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
                Object_predictions = Object_predictions[ids]

            NbDet += len(Object_predictions)

            if(len(labels)>0):
                ids = np.where((labels[:,0]>=range_min) & (labels[:,0] <= range_max))
                labels = labels[ids]

            if(len(labels)>0):
                ground_truth_box_corners = np.asarray(get_rotated_box(labels))
                NbGT += ground_truth_box_corners.shape[0]

            # valid predictions and labels exist for the currently inspected point cloud
            if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

                used_gt = np.zeros(len(ground_truth_box_corners))
                for pid, prediction in enumerate(Object_predictions):
                    iou = bbox_iou(prediction[1:], ground_truth_box_corners)
                    ids = np.where(iou>=IOU_threshold)[0]
                    
                    if(len(ids)>0):
                        TP += 1
                        used_gt[ids]=1

                        # cummulate errors
                        range_error += np.sum(np.abs(labels_r[ids] - pred_r[pid]))
                        angle_error += np.sum(np.abs(labels_a[ids] - pred_a[pid]))
                        nbObjects+=len(ids)
                    else:
                        FP+=1
                FN += np.sum(used_gt==0)


            elif(len(ground_truth_box_corners)==0):
                FP += len(Object_predictions)
            elif(len(Object_predictions)==0):
                FN += len(ground_truth_box_corners)
            
            prog_bar.update()
                

        if(TP!=0):
            precision.append( TP / (TP+FP)) # When there is a detection, how much I m sure
            recall.append(TP / (TP+FN))
        else:
            precision.append( 0) # When there is a detection, how much I m sure
            recall.append(0)

        if(nbObjects > 0):
            RangeError.append(range_error/nbObjects)
            AngleError.append(angle_error/nbObjects)

    perfs['precision']=precision
    perfs['recall']=recall

    F1_score = (np.mean(precision)*np.mean(recall))/((np.mean(precision) + np.mean(recall))/2)

    result, ret_dict = '', {}

    ret_dict['mAP'] = np.mean(perfs['precision'])
    ret_dict['mAR'] = np.mean(perfs['recall'])
    ret_dict['F1'] = F1_score
    ret_dict['Range Error'] = np.mean(RangeError)
    ret_dict['Angle Error'] = np.mean(AngleError)

    result += '\n------------ Detection Scores ------------\n'
    result += '  mAP: {}\n'.format(ret_dict['mAP'])
    result += '  mAR: {}\n'.format(ret_dict['mAR'])
    result += '  F1 score: {}\n'.format(ret_dict['F1'])

    result += '------------ Regression Errors------------\n'
    result += '  Range Error: {}\n'.format(ret_dict['Range Error'])
    result += '  Angle Error: {}\n'.format(ret_dict['Angle Error'])

    return result, ret_dict

def GetFullMetricsV2(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    """Used for evaluation in RadarFormer
    
    Args:
        predictions: list of array. Each array is in shape of [N, 2].
        object_labels: list of array. The i-th array is in shape of [M_i, 2].
        range_min: float. The minimum range of perception.
        range_max: float. The maximum range of perception.
        IOU_threshold: float. The threshold of IoU for matching.

    Returns:
        results: The result in string.
        ret_dict: dict. The detailed results.
    """
    perfs = {}
    precision = []
    recall = []
    iou_threshold = []
    RangeError = []
    AngleError = []

    out = []

    print("Evaluating ...")
    prog_bar = ProgressBar(len(np.arange(0.1,0.96,0.1)) * len(predictions))
    for threshold in np.arange(0.1,0.96,0.1):

        iou_threshold.append(threshold)

        TP = 0
        FP = 0
        FN = 0
        NbDet = 0
        NbGT = 0
        NBFrame = 0
        range_error=0
        angle_error=0
        nbObjects = 0

        for frame_id in range(len(predictions)):

            pred_loc= predictions[frame_id]
            pred_r = np.linalg.norm(pred_loc[:, :2], axis=1)
            pred_a = np.arctan2(pred_loc[:, 1], pred_loc[:, 0]) * 180 / np.pi
            pred = np.stack([pred_r, pred_a, pred_loc[:,-1]], axis=-1)

            labels_loc = object_labels[frame_id]
            labels_r = np.linalg.norm(labels_loc[:, :2], axis=1)
            labels_a = np.arctan2(labels_loc[:, 1], labels_loc[:, 0]) * 180 / np.pi
            labels = np.stack([labels_r, labels_a], axis=-1)

            # get final bounding box predictions
            Object_predictions = []
            ground_truth_box_corners = []           
            
            if(len(pred)>0):
                Object_predictions = process_predictions_FFT(pred,confidence_threshold=threshold)

            if(len(Object_predictions)>0):
                max_distance_predictions = (Object_predictions[:,1]+Object_predictions[:,3])/2
                ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
                Object_predictions = Object_predictions[ids]

            NbDet += len(Object_predictions)

            if(len(labels)>0):
                ids = np.where((labels[:,0]>=range_min) & (labels[:,0] <= range_max))
                labels = labels[ids]

            if(len(labels)>0):
                ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
                NbGT += ground_truth_box_corners.shape[0]

            # valid predictions and labels exist for the currently inspected point cloud
            if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

                used_gt = np.zeros(len(ground_truth_box_corners))
                for pid, prediction in enumerate(Object_predictions):
                    iou = bbox_iou(prediction[1:], ground_truth_box_corners)
                    ids = np.where(iou>=IOU_threshold)[0]
                    
                    if(len(ids)>0):
                        TP += 1
                        used_gt[ids]=1

                        # cummulate errors
                        range_error += np.sum(np.abs(labels_r[ids] - pred_r[pid]))
                        angle_error += np.sum(np.abs(labels_a[ids] - pred_a[pid]))
                        nbObjects+=len(ids)
                    else:
                        FP+=1
                FN += np.sum(used_gt==0)


            elif(len(ground_truth_box_corners)==0):
                FP += len(Object_predictions)
            elif(len(Object_predictions)==0):
                FN += len(ground_truth_box_corners)
            
            prog_bar.update()
                

        if(TP!=0):
            precision.append( TP / (TP+FP)) # When there is a detection, how much I m sure
            recall.append(TP / (TP+FN))
        else:
            precision.append( 0) # When there is a detection, how much I m sure
            recall.append(0)

        if(nbObjects > 0):
            RangeError.append(range_error/nbObjects)
            AngleError.append(angle_error/nbObjects)

    perfs['precision']=precision
    perfs['recall']=recall

    F1_score = (np.mean(precision)*np.mean(recall))/((np.mean(precision) + np.mean(recall))/2)

    result, ret_dict = '', {}

    ret_dict['mAP'] = np.mean(perfs['precision'])
    ret_dict['mAR'] = np.mean(perfs['recall'])
    ret_dict['F1'] = F1_score
    ret_dict['Range Error'] = np.mean(RangeError)
    ret_dict['Angle Error'] = np.mean(AngleError)

    result += '\n------------ Detection Scores ------------\n'
    result += '  mAP: {}\n'.format(ret_dict['mAP'])
    result += '  mAR: {}\n'.format(ret_dict['mAR'])
    result += '  F1 score: {}\n'.format(ret_dict['F1'])

    result += '------------ Regression Errors------------\n'
    result += '  Range Error: {}\n'.format(ret_dict['Range Error'])
    result += '  Angle Error: {}\n'.format(ret_dict['Angle Error'])

    return result, ret_dict

def GetDetMetrics(predictions,object_labels,threshold=0.2,range_min=5,range_max=70,IOU_threshold=0.2):

    TP = 0
    FP = 0
    FN = 0
    NbDet=0
    NbGT=0
   
    # get final bounding box predictions
    Object_predictions = []
    ground_truth_box_corners = []    
    labels=[]       

    if(len(predictions)>0):
        Object_predictions = process_predictions_FFT(predictions,confidence_threshold=threshold)

    if(len(Object_predictions)>0):
        max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
        ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
        Object_predictions = Object_predictions[ids]

    NbDet = len(Object_predictions)
 
    if(len(object_labels)>0):
        ids = np.where((object_labels[:,0]>=range_min) & (object_labels[:,0] <= range_max))
        labels = object_labels[ids]
    if(len(labels)>0):
        ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
        NbGT = len(ground_truth_box_corners)

    # valid predictions and labels exist for the currently inspected point cloud
    if NbDet>0 and NbGT>0:

        used_gt = np.zeros(len(ground_truth_box_corners))

        for pid, prediction in enumerate(Object_predictions):
            iou = bbox_iou(prediction[1:], ground_truth_box_corners)
            ids = np.where(iou>=IOU_threshold)[0]

            if(len(ids)>0):
                TP += 1
                used_gt[ids]=1
            else:
                FP+=1
        FN += np.sum(used_gt==0)

    elif(NbGT==0):
        FP += NbDet
    elif(NbDet==0):
        FN += NbGT
        
    return TP,FP,FN


def GetSegMetrics(PredMap,label_map):

    # Segmentation
    pred = PredMap.reshape(-1)>=0.5
    label = label_map.reshape(-1)

    intersection = np.abs(pred*label).sum()
    union = np.sum(label) + np.sum(pred) -intersection
    iou = intersection /union

    return iou

class Metrics():
    def __init__(self,):
        
        self.iou = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.recall = 0
        self.mIoU =0

    def update(self,PredMap,label_map,ObjectPred,Objectlabels,threshold=0.2,range_min=5,range_max=70):

        if(len(PredMap)>0):
            pred = PredMap.reshape(-1)>=0.5
            label = label_map.reshape(-1)

            intersection = np.abs(pred*label).sum()
            union = np.sum(label) + np.sum(pred) -intersection
            self.iou.append(intersection /union)

        TP,FP,FN = GetDetMetrics(ObjectPred,Objectlabels,threshold=0.2,range_min=range_min,range_max=range_max)

        self.TP += TP
        self.FP += FP
        self.FN += FN

    def reset(self,):
        self.iou = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.mIoU =0

    def GetMetrics(self,):
        
        if(self.TP+self.FP!=0):
            self.precision = self.TP / (self.TP+self.FP)
        if(self.TP+self.FN!=0):
            self.recall = self.TP / (self.TP+self.FN)

        if(len(self.iou)>0):
            self.mIoU = np.asarray(self.iou).mean()

        return self.precision,self.recall,self.mIoU 


