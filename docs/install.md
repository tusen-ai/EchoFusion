# Step-by-step installation instructions
**a. Clone EchoFusion.**
```bash
git clone https://github.com/tusen-ai/EchoFusion
cd EchoFusion
mkdir ckpts ###pretrain weights
mkdir data ###dataset
```
The requied `ckpts/resnet50_bev.pth` can be downloaded from [HERE](https://drive.google.com/drive/folders/1R0uRgUtYMKdmps4B1noKTj1atv5zFWi6?usp=sharing).

**b. Create a conda virtual environment and activate it.**
```bash
conda create -n echofusion python=3.7 -y
conda activate echofusion
```

**c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```bash
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge
# Recommended torch>=1.9
```

**d. Install mmcv-full.**
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
examples:
```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
```

**e. Install mmdet and mmseg.**
```bash
pip install mmdet==2.19.0
pip install mmsegmentation==0.20.0
```

**f. Install mmdet3d from source code.**
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.3 
```
Then use mmdet3d_supp/core/bbox/iou_calculators provided by EchoFusion repo to add support of let-iou. 
```bash
rm -rf ./mmdet3d/core/bbox/iou_calculators
cp -r ../mmdet3d_supp/core/bbox/iou_calculators ./mmdet3d/core/bbox
rm -rf ./mmdet3d/core/bbox/__init__.py
cp ../mmdet3d_supp/core/bbox/__init__.py ./mmdet3d/core/bbox
pip install -v -e .
cd ..
```

**g. Install RADIal from source code following the [official instructions](https://github.com/valeoai/RADIal/tree/main).**
```bash
git clone https://github.com/valeoai/RADIal
cd RADIal/DBReader
pip install .
conda install -c intel intel-aikit-modin
pip install cupy-cuda111
pip3 install pkbar
pip3 install polarTransform
pip3 install --upgrade git+https://github.com/klintan/pypcd.git
cd ../..
```

**h. Install modified Extensible-Object-Detection-Evaluator from provided code.**
```bash
cd Extensible-Object-Detection-Evaluator
pip install treelib
pip install ipdb
pip install .
cd ..
```

**i. Other dependencies.**
```bash
pip install cplxmodule
pip install spconv-cu113
conda update ffmpeg
pip install opencv-python
```


