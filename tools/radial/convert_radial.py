from DBReader.DBReader import SyncReader
from mmcv import ProgressBar
from concurrent import futures as futures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2
import os
import argparse
from rpl import RadarSignalProcessing

Sequences = {'Validation':['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
            'Test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47','RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}

def create_dir(out_dir, clear=False):
    """Create data structure."""
    # check and create files
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'ImageSets')):
        os.makedirs(os.path.join(out_dir, 'ImageSets'))
    if not os.path.exists(os.path.join(out_dir, 'images')):
        os.makedirs(os.path.join(out_dir, 'images'))
    if not os.path.exists(os.path.join(out_dir, 'radars_adc')):
        os.makedirs(os.path.join(out_dir, 'radars_adc'))
    if not os.path.exists(os.path.join(out_dir, 'radars_rt')):
        os.makedirs(os.path.join(out_dir, 'radars_rt'))
    if not os.path.exists(os.path.join(out_dir, 'radars_pcd')):
        os.makedirs(os.path.join(out_dir, 'radars_pcd'))
    if not os.path.exists(os.path.join(out_dir, 'radars_ra')):
        os.makedirs(os.path.join(out_dir, 'radars_ra'))
    if not os.path.exists(os.path.join(out_dir, 'radars_rd')):
        os.makedirs(os.path.join(out_dir, 'radars_rd'))
    if not os.path.exists(os.path.join(out_dir, 'lidars')):
        os.makedirs(os.path.join(out_dir, 'lidars'))
    if not os.path.exists(os.path.join(out_dir, 'labels')):
        os.makedirs(os.path.join(out_dir, 'labels'))
    
    if clear:
        # clear txt in ImageSets
        for file in os.listdir(os.path.join(out_dir, 'ImageSets')):
            os.remove(os.path.join(out_dir, 'ImageSets', file))

def convert(root_dir, 
            out_dir,
            clear=False,
            num_worker=8):
    """ Parallelized conversion process. """
    root_dir = args.root_path
    out_dir = args.out_dir
    create_dir(out_dir, clear)  # note that ImageSets will be cleared

    # read labels from csv
    labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
    unique_ids = np.unique(labels[:,0]).tolist()
    label_dict = {}
    for i,ids in enumerate(unique_ids):
        sample_ids = np.where(labels[:,0]==ids)[0]
        label_dict[ids]=sample_ids
    sample_keys = list(label_dict.keys())
    pkbar = ProgressBar(len(unique_ids))

    def map_func(frame_id):
        # From the sample id, retrieve all the labels ids
        entries_indexes = label_dict[frame_id]
        # Get the objects labels
        box_infos = labels[entries_indexes]

        record_name = box_infos[:, -3][0]
        data_root = os.path.join(root_dir, record_name)
        db = SyncReader(data_root, tolerance=20000, silent=True);

        idx = box_infos[:, -2][0]
        sample = db.GetSensorData(idx)

        # save ImageSets
        if record_name in Sequences['Validation']:
            set_name = 'val'
        elif record_name in Sequences['Test']:
            set_name = 'test'
        else:
            set_name = 'train'
        with open(os.path.join(out_dir, 'ImageSets', set_name + '.txt'), 'a') as f:
            f.write('%06d' % frame_id + '\n')

        # save image
        image = sample['camera']['data']  # load camera data, [1080, 1920, 3]
        image_name = os.path.join(out_dir, 'images', '%06d.png' % frame_id)
        cv2.imwrite(image_name, image)

        # save lidar as binary
        pts_lidar = sample['scala']['data'].astype(dtype=np.float32) # load lidar data, [15608, 11], no compensation
        lidar_name = os.path.join(out_dir, 'lidars', '%06d.bin' % frame_id)
        pts_lidar.tofile(lidar_name)

        # save radar as binary
        clibration_path = os.path.join(root_dir, 'CalibrationTable.npy')

        RSP = RadarSignalProcessing(clibration_path, method='ADC',device='cuda',silent=True)
        adc=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']) # load radar data after range fft

        RSP = RadarSignalProcessing(clibration_path, method='RT',device='cuda',silent=True)
        rt=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']) # load radar data after range fft
        
        RSP = RadarSignalProcessing(clibration_path, method='PC',silent=True)
        pcd=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']).astype(dtype=np.float32) # load radar pcd

        RSP = RadarSignalProcessing(clibration_path, method='RA',silent=True)
        ra=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']) # load radar ra map
        
        RSP = RadarSignalProcessing(clibration_path, method='RD',silent=True)
        rd=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']) # load radar ra map

        radar_name = os.path.join(out_dir, 'radars_adc', '%06d.bin' % frame_id)
        adc.tofile(radar_name)

        radar_name = os.path.join(out_dir, 'radars_rt', '%06d.bin' % frame_id)
        rt.tofile(radar_name)

        radar_name = os.path.join(out_dir, 'radars_pcd', '%06d.bin' % frame_id)
        pcd.tofile(radar_name)

        radar_name = os.path.join(out_dir, 'radars_ra', '%06d.bin' % frame_id)
        ra.tofile(radar_name)

        radar_name = os.path.join(out_dir, 'radars_rd', '%06d.bin' % frame_id)
        rd.tofile(radar_name)

        # save labels as txt
        label_name = os.path.join(out_dir, 'labels', '%06d.txt' % frame_id)
        np.savetxt(label_name, box_infos, fmt='%d %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %s %d %d')

        pkbar.update()

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, unique_ids)

    print('\nConversion Done!')

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='data/radial/raw',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='data/radial_kitti_format',
    help='specify the output path of dataset')
parser.add_argument(
    '--workers',
    type=int,
    default=8,
    help='whether to clear ImageSets')
parser.add_argument(
    '--clear',
    action="store_true",
    help='whether to clear ImageSets')
args = parser.parse_args()


if __name__ == '__main__':
    root_dir = args.root_path
    out_dir = args.out_dir
    clear = args.clear
    num_worker = args.workers
    
    convert(root_dir, out_dir, clear=clear, num_worker=num_worker)

    # open txt in ImageSet and check if there are repeated lines
    txt_files = [f for f in os.listdir(os.path.join(out_dir, 'ImageSets')) if f.endswith('.txt')]
    for txt_file in txt_files:
        with open(os.path.join(out_dir, 'ImageSets', txt_file), 'r') as f:
            lines = f.readlines()
            if not len(lines) == len(set(lines)):
                print('Repeated lines in %s.' % txt_file)
    print('Done.')


