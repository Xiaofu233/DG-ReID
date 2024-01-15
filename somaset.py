import os
import glob
import copy
import random
from collections import defaultdict
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

@DATASET_REGISTRY.register()
class Somaset(ImageDataset):
    dataset_name = "Somaset"

    def __init__(self, root='datasets', **kwargs):
        train = []
        self.root = '/data/wuwei/Synthetic_ReID_datasets/somaset'
        pid_str_list = os.listdir(self.root)
        pid2label = {pid: label for label, pid in enumerate(pid_str_list)}
        for pid_str in pid_str_list:
            data_dir = os.path.join(self.root, pid_str)
            pathlist = os.listdir(data_dir)
            for path in pathlist:
                img_dir = os.path.join(data_dir, path)
                img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
                
                pid = self.dataset_name + "_" + str(pid2label[pid_str])
                camid = self.dataset_name + "_0"
                for img_path in img_paths:
                    train.append((img_path, pid, camid))
            
        
        query = []
        gallery = []

        super(Somaset, self).__init__(train, query, gallery, **kwargs)
