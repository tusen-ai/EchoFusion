from mmcv import ProgressBar
import pickle as pkl
import pandas as pd
import numpy as np
import time
import os
import argparse

parser = argparse.ArgumentParser(description='K-Radar data unzipper arg parser')
parser.add_argument(
    '--source-path',
    type=str,
    default='data/k-radar-zip/1-20',
    help='specify the zip path of dataset')
parser.add_argument(
    '--target-dir',
    type=str,
    default='data/k-radar',
    help='specify the output path of dataset')
args = parser.parse_args()


if __name__ == '__main__':
    src_dir = args.source_path
    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # read all zip files in src_dir and sort them
    zip_files = os.listdir(src_dir)
    zip_files.sort()
    # 5, 7, 16, 17, 20

    # iterate on zip files
    for zip_file in zip_files:
        # extract index name from zip file name
        index = zip_file.split('_')[0]
        # unzip file to indexed file in target_dir
        zip_path = os.path.join(src_dir, zip_file)
        out_path = os.path.join(target_dir, index)
        os.system(f'unzip {zip_path} -d {out_path}')
        # print progress
        print(f'{zip_file} done.')
    
    print('Done.')
