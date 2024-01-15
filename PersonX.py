import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PersonX(ImageDataset):
    dataset_name = "PersonX"

    def __init__(self, root='datasets', **kwargs):
        self.root = '/data/wuwei/Synthetic_ReID_datasets/PersonX_v1'
        dataset_dir = ['1','2','3','4','5','6']
        train = []
        query = []
        gallery = []

        for dir in dataset_dir:
            self.data_dir = osp.join(self.root, dir)

            self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
            self.query_dir = osp.join(self.data_dir, 'query')
            self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

            required_files = [
                self.data_dir,
                self.train_dir,
                self.query_dir,
                self.gallery_dir,
            ]
            self.check_before_run(required_files)

            sub_train = self.process_dir(self.train_dir)
            sub_query = self.process_dir(self.query_dir, is_train=False)
            sub_gallery = self.process_dir(self.gallery_dir, is_train=False)

            train.extend(sub_train)
            query.extend(sub_query)
            gallery.extend(sub_gallery)
        
        super(PersonX, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            pid -= 1
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
