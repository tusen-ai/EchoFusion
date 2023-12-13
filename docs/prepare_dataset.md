## RADIal

Download RADIal raw dataset data [HERE](https://drive.google.com/drive/folders/1JHPLQsjwtO0SOgAgkq8KOqXndKaOGwFM?usp=sharing). Prepare the KITTI format RADIal data by running

**Download data and convert format**

```
cd EchoFusion
# download RADIal data to ./data/radial
python tools/radial/convert_radial.py --root-path ./data/radial --out-dir data/radial_kitti_format
```

**Copy camera_calib.npy**

Copy camera_calib.npy (shared by all frames) to the folder of the KITTI format RADIal data by running:

```
mkdir data/radial_kitti_format/calibs
cp RADIal/DBReader/examples/camera_calib.npy data/radial_kitti_format/calibs
```

**Generate info pickle files**
```
python tools/radial/create_data.py radial
```

**Copy CalibrationTable.npy used for data processing**
```
cp RADIal/SignalProcessing/CalibrationTable.npy data/radial_kitti_format
```

**Prepare 3D labels**

Download 3D labels of RADIAl from [HERE](https://drive.google.com/drive/folders/1R0uRgUtYMKdmps4B1noKTj1atv5zFWi6?usp=sharing)  and save the txt files to `data/radial_kitti_format/labels_x/`. Then create info pickle files for 3D labels using:
```
python tools/radialx/create_data.py radialx
```

**Folder structure**
```
EchoFusion
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50_bev.pth
├── data/
│   ├── radial/
│   │   ├── RECORD@2020-11-21_11.54.31/
│   │   ├── RECORD@2020-11-21_11.58.53/
│   │   ├── ...
│   │   ├── RECORD@2020-11-22_12.56.17/
|   |   ├── CalibrationTable.npy
|   |   ├── labels.csv
│   ├── radial_kitti_format/
│   │   ├── calibs/
│   │   ├── images/
│   │   ├── ImageSets/
│   │   ├── labels/
│   │   ├── labels_x/
|   |   ├── lidars/
|   |   ├── radars_adc/
|   |   ├── radars_pcd/
|   |   ├── radars_ra/
|   |   ├── radars_rd/
|   |   ├── radars_rt/
|   |   ├── CalibrationTable.npy
|   |   ├── radial_infos_test.pkl
|   |   ├── radial_infos_train.pkl
|   |   ├── radial_infos_trainval.pkl
|   |   ├── radial_infos_val.pkl
|   |   ├── radialx_infos_test.pkl
|   |   ├── radialx_infos_train.pkl
|   |   ├── radialx_infos_trainval.pkl
|   |   ├── radialx_infos_val.pkl
```

## KRadar-20
**Download and unzip data**

Download the first 20 sequences of KRadar dataset from [HERE](https://drive.google.com/drive/folders/1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_) and unzip.
```
cd EchoFusion
# download KRadar data to ./data/k-radar-zip/1-20
# unzip data to ./data/k-radar
python tools/kradar/unzip_data.py
```
**Prepare sparse radar tensor**

Use our modified KRadar codebase to prepare sparse radar tensor. It is required to change `LIST_DIR` and `DIR_SPARSE_CB` to your own path to k-radar.
```
cd K-Radar-main
python datasets/kradar_detection_v1_0.py
cd ..
mkdir data/k-radar/ImageSets
python K-Radar-main/tools/train_test_splitter/train_val_splitter.py
python K-Radar-main/tools/train_test_splitter/train_test_splitter.py
```
**Generate info pickle files**
```
python tools/kradar/create_data.py kradar
```

**Folder structure**
```
EchoFusion
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50_bev.pth
├── data/
│   ├── radial/
│   ├── radial_kitti_format/
│   ├── k-radar/
│   │   ├── 1/
│   │   │   ├── cam-front/
│   │   │   ├── cam-left/
│   │   │   ├── cam-rear/
│   │   │   ├── cam-right/
│   │   │   ├── images/
│   │   │   ├── info_calib/
│   │   │   ├── info_label/
│   │   │   ├── os1-128/
│   │   │   ├── os2-64/
│   │   │   ├── radar_dense_pcd/
│   │   │   ├── radar_ra/
│   │   │   ├── radar_zyx_cube/
│   │   │   ├── sp_rdr_cube/
│   │   │   ├── time_info/
│   │   │   ├── description.txt
│   │   ├── 2/
│   │   ├── ...
│   │   ├── 19/
│   │   ├── 20/
|   |   ├── ImageSets/
|   |   ├── kradar_infos_test.pkl
|   |   ├── kradar_infos_train.pkl
|   |   ├── kradar_infos_trainval.pkl
|   |   ├── kradar_infos_val.pkl
```
