import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from ipdb import set_trace
import numpy as np

sns.set_theme(style="darkgrid")

path = '/mnt/truenas/scratch/yang.liu3/Python/RadarFormer/PolarFormer/evaluation_results/20230428-165024-.pkl'
with open(path, 'rb') as f:
    eval_data = pkl.load(f)
fppi = eval_data['IoU@0.5/type_Vehicle/range_[5, 100]']

recall = np.arange(101) / 100
print(fppi)

plt.plot(recall[fppi>0], fppi[fppi>0])
plt.savefig('test.png')