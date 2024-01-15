# encoding: utf-8
import os
from glob import glob
from collections import defaultdict
import copy
import random

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['Shinpuhkan', ]


@DATASET_REGISTRY.register()
class Shinpuhkan(ImageDataset):
    """shinpuhkan
    """
    dataset_dir = "shinpuhkan"
    dataset_name = 'Shinpuhkan'

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

        for root, dirs, files in os.walk(train_path):
            img_names = list(filter(lambda x: x.endswith(".jpg"), files))
            # fmt: off
            if len(img_names) == 0: continue
            # fmt: on
            for img_name in img_names:
                img_path = os.path.join(root, img_name)
                split_path = img_name.split('_')
                pid = self.dataset_name + "_" + split_path[0]
                camid = self.dataset_name + "_" + split_path[2]
                data.append((img_path, pid, camid))

        return data
    
    #write by hby
    #random choose 50% ids as test
    def prepare_split(self, train_path, split_path):
        if not os.path.exists(split_path):
            print('Creating splits ...')
            pid_dict = defaultdict(list)
            for root, dirs, files in os.walk(train_path):
                img_names = list(filter(lambda x: x.endswith(".jpg"), files))
                for img_name in img_names:
                    img_path = os.path.join(root, img_name)
                    pid = int(img_name.split('_')[0])
                    pid_dict[pid].append(img_path)
            pids = list(pid_dict.keys())
            num_pids = len(pids)
            assert num_pids == 24, 'There should be 24 identities, ' \
                                    'but got {}, please check the data'.format(num_pids)

            num_train_pids = int(num_pids * 0.5)

            splits = []
            for _ in range(10):
                # randomly choose num_train_pids train IDs and the rest for test IDs
                pids_copy = copy.deepcopy(pids)
                random.shuffle(pids_copy)
                train_pids = pids_copy[:num_train_pids]
                #print(train_pids)
                test_pids = pids_copy[num_train_pids:]
                #print(test_pids)
                
                train = []
                query = []
                gallery = []

                # for train IDs, all images are used in the train set.
                for pid_ in train_pids:
                    img_paths = pid_dict[pid_]
                    pid = self.dataset_name + '_' + str(pid_)
                    for img_path in img_paths:
                        cam = int(img_path.split('/')[-1].split('_')[1])
                        camid = self.dataset_name + '_' + str(cam)
                        train.append([img_path, pid, camid])

                # for each test ID, choose 2 images, one for query and one for gallery
                for pid_ in test_pids:
                    pid = pid_
                    img_names = pid_dict[pid_]
                    selected_img_paths = random.sample(img_names, 2)
                    
                    # first image for query
                    camid = int(selected_img_paths[0].split('/')[-1].split('_')[1])
                    query.append([selected_img_paths[0], pid, camid])

                    # other  images for gallery
                    camid = int(selected_img_paths[1].split('/')[-1].split('_')[1])
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
        #print(fpath)
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



