import os
import numpy as np
import pickle as pkl

def get_fft_radnet_pred_object(file_path):
    test_info = pkl.load(open('/mnt/weka/scratch/yang.liu3/pyworkspace/EchoFusion/data/radial_kitti_format/radialx_infos_test.pkl',"rb"))
    test_ids = []
    for i in range(len(test_info)):
        test_ids.append(test_info[i]['image']['image_idx'])
    sorted_test_ids = np.sort(test_ids)

    predictions = pkl.load(open(file_path, "rb"))
    new_dict = {}
    for i in range(len(predictions['prediction']['objects'])):
        #convert xy to ralabels
        ras = predictions['prediction']['objects'][i]
        if len(ras) == 0:
            new_dict[sorted_test_ids[i]] = dict(
                box=np.zeros((0, 7)),
                score=np.zeros((0,)),
                type=np.zeros(0, dtype='<U32'),
                timestamp=sorted_test_ids[i],
            )
        else:
            ras = ras[ras[:,-1]>0.05]
            xys = ras.copy()
            xys[:,0] = np.sqrt(ras[:,0]**2 + ras[:,1]**2)
            xys[:,1] = np.arctan2(ras[:,1], ras[:,0])
            xys = np.stack([xys[:,0],xys[:,1],np.zeros((len(xys))),xys[:,2],xys[:,3],np.ones((len(xys))),xys[:,4],xys[:,5]], axis=1)
            types = np.zeros(len(xys), dtype='<U32')
            types[:] = 'Vehicle'
            new_dict[sorted_test_ids[i]] = dict(
                box=xys[:, :7],
                score=xys[:, 7],
                type=types,
                timestamp=sorted_test_ids[i],
            )
    print(new_dict[1627])
    return new_dict

def get_fft_radnet_gt_object(file_path):
    test_info = pkl.load(open('/mnt/weka/scratch/yang.liu3/pyworkspace/EchoFusion/data/radial_kitti_format/radialx_infos_test.pkl',"rb"))
    test_ids = []
    for i in range(len(test_info)):
        test_ids.append(test_info[i]['image']['image_idx'])
    sorted_test_ids = np.sort(test_ids)

    predictions = pkl.load(open(file_path, "rb"))
    new_dict = {}
    for i in range(len(predictions['label']['objects'])):
        #convert xy to ralabels
        ras = predictions['label']['objects'][i]
        if len(ras) == 0:
            new_dict[sorted_test_ids[i]] = dict(
                box=np.zeros((0, 7)),
                score=np.zeros((0,)),
                type=np.zeros(0, dtype='<U32'),
                timestamp=sorted_test_ids[i],
            )
        else:
            ras = ras[ras[:,-1]>0.05]
            xys = ras.copy()
            xys[:,0] = np.sqrt(ras[:,0]**2 + ras[:,1]**2)
            xys[:,1] = np.arctan2(ras[:,1], ras[:,0])
            types = np.zeros(len(xys), dtype='<U32')
            types[:] = 'Vehicle'
            new_dict[sorted_test_ids[i]] = dict(
                box=xys[:, :7],
                score=xys[:, 7],
                type=types,
                timestamp=sorted_test_ids[i],
            )
    # print(new_dict[1627])
    return new_dict

def get_radial_pred_object(file_path):
    my_prediction = np.load(file_path, allow_pickle=True)

    new_dict = {}
    for i in range(len(my_prediction)):
        if len(my_prediction[i]) == 0:
            new_dict[i] = dict(
                box=np.zeros((0, 7)),
                score=np.zeros((0,)),
                type=np.zeros(0, dtype='<U32'),
                timestamp=i,
            )
        else:
            filter_prediction = my_prediction[i][my_prediction[i][:, 7]>0.01]
            types = np.zeros(len(filter_prediction), dtype='<U32')
            types[:] = 'Vehicle'
            new_dict[i] = dict(
                box=filter_prediction[:, :7],
                score=filter_prediction[:, 7],
                type=types,
                timestamp=i,
            )

    return new_dict

def get_radial_gt_object(gt_file='./data/radial_kitti_format/radialx_infos_test.pkl'):
    test_info = pkl.load(open(gt_file,"rb"))

    new_dict = {}
    for idx in range(len(test_info)):
        gt_anno = test_info[idx]['annos']
        gt_bbox = np.concatenate((gt_anno['location'], gt_anno['dimensions'],  gt_anno['rotation_y'][:, None]), axis=-1)
        if len(gt_bbox) == 0:
            new_dict[idx] = dict(
                box=np.zeros((0, 7)),
                score=np.zeros((0,)),
                type=np.zeros(0, dtype='<U32'),
                timestamp=idx,
            )
        else:
            types = np.zeros(len(gt_bbox), dtype='<U32')
            types[:] = 'Vehicle'
            new_dict[idx] = dict(
                box=gt_bbox,
                score=np.ones((len(gt_bbox),)),
                type=types,
                timestamp=idx,
            )

    return new_dict