# encoding: utf-8
import os
from glob import glob
from collections import defaultdict
import copy
import random

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['SYSUmm', ]


@DATASET_REGISTRY.register()
class SYSUmm(ImageDataset):
    """sysu mm
    """
    dataset_dir = "SYSU-MM01"
    dataset_name = "SYSUmm"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        #train = self.process_train(self.train_path)
        train, query, gallery =self.process_data(self.train_path)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, train_path):
        data = []

        file_path_list = ['cam1', 'cam2', 'cam4', 'cam5']

        for file_path in file_path_list:
            camid = self.dataset_name + "_" + file_path
            pid_list = os.listdir(os.path.join(train_path, file_path))
            for pid_dir in pid_list:
                pid = self.dataset_name + "_" + pid_dir
                img_list = glob(os.path.join(train_path, file_path, pid_dir, "*.jpg"))
                for img_path in img_list:
                    data.append([img_path, pid, camid])
        return data

#write by hby
    #random choose 50% ids as test
    def prepare_split(self, train_path, split_path):
        if not os.path.exists(split_path):
            print('Creating splits ...')
            file_path_list = ['cam1', 'cam2', 'cam4', 'cam5']
            pid_dict = defaultdict(list)
            for cam in file_path_list:
                camid = self.dataset_name + "_" + cam
                pid_list = os.listdir(os.path.join(train_path, cam))
                for pid_dir in pid_list:
                    pid = int(pid_dir)
                    img_path = glob(os.path.join(train_path, cam, pid_dir, "*.jpg"))
                    pid_dict[pid].extend(img_path)
            
            pids = list(pid_dict.keys())
            num_pids = len(pids)
            assert num_pids == 510, 'There should be 510 identities, ' \
                                    'but got {}, please check the data'.format(num_pids)

            num_train_pids = int(num_pids * 0.5)

            splits = []
            for _ in range(10):
                # randomly choose num_train_pids train IDs and the rest for test IDs
                pids_copy = copy.deepcopy(pids)
                random.shuffle(pids_copy)
                train_pids = pids_copy[:num_train_pids]
                test_pids = pids_copy[num_train_pids:]
                #print(test_pids)
                
                train = []
                query = []
                gallery = []

                for pid_ in train_pids:
                    img_paths = pid_dict[pid_]
                    for img_path in img_paths:
                        #print(img_path)
                        #/data/hby0728/All_ReID_Datasets/clear_all_datasets/SYSU-MM01/cam1/pid/*.jpg
                        cam = img_path.split('/')[6][3] #cam1-4
                        camid = self.dataset_name + '_' + cam
                        pid = self.dataset_name + "_" + str(pid_)
                        train.append([img_path, pid, camid])

                # for each test ID, choose 2 images, one for query and one for gallery
                for pid_ in test_pids:
                    img_names = pid_dict[pid_]
                    selected_img_paths = random.sample(img_names, 2)
                    pid = int(pid_)
                    # first image for query
                    camid = int(selected_img_paths[0].split('/')[6][3])
                    query.append([selected_img_paths[0], pid, camid])

                    # other  images for gallery
                    camid = int(selected_img_paths[1].split('/')[6][3])
                    gallery.append([selected_img_paths[1], pid, camid])
                
                split = {'train': train, 'query': query, 'gallery': gallery}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            self.write_json(splits, split_path)
            print('Split file is saved to {}'.format(split_path))
    
    def process_data(self, train_path):
        split_id = 0
        split_path = os.path.join(train_path, 'splits.json')
        self.prepare_split(train_path, split_path)
        splits = self.read_json(split_path)
        split = splits[split_id]
        return split['train'], split['query'], split['gallery']


    def read_json(self, fpath):
        import json
        """Reads json file from a path."""
        with open(fpath, 'r') as f:
            obj = json.load(f)
        return obj

    def write_json(self, obj, fpath):
        import json
        """Writes to a json file."""
        self.mkdir_if_missing(os.path.dirname(fpath))
        with open(fpath, 'w') as f:
            json.dump(obj, f, indent=4, separators=(',', ': '))

    def mkdir_if_missing(self, dirname):
        import errno
        """Creates dirname if it is missing."""
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
