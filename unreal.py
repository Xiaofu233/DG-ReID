import os
import glob
import copy
import random
import re
from collections import defaultdict
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

@DATASET_REGISTRY.register()
class Unreal(ImageDataset):
    dataset_name = "Unreal"

    def __init__(self, root='datasets', **kwargs):
        train = []
        self.root = '/data/wuwei/Synthetic_ReID_datasets/filtered_unreal'
        img_paths = glob.glob(os.path.join(self.root, '*.jpg'))
        #print(len(img_paths))

        pid_container = set()
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            pid = img_name.split('_')[0]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        #print(len(pid2label.keys()))
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            pid = img_name.split('_')[0]

            label = self.dataset_name + "_" + str(pid2label[pid])
            camid = self.dataset_name + "_0"
            train.append((img_path, label, camid))
        
        query = []
        gallery = []

        super(Unreal, self).__init__(train, query, gallery, **kwargs)
