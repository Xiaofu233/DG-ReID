# encoding: utf-8
import os
from glob import glob
from collections import defaultdict
import copy
import random
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['PKU', ]


@DATASET_REGISTRY.register()
class PKU(ImageDataset):
    """PKU
    """
    dataset_dir = "PKUv1a_128x48"
    dataset_name = 'PKU'

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
        img_paths = glob(os.path.join(train_path, "*.png"))

        for img_path in img_paths:
            split_path = img_path.split('/')
            img_info = split_path[-1].split('_')
            pid = self.dataset_name + "_" + img_info[0]
            camid = self.dataset_name + "_" + img_info[1]
            data.append([img_path, pid, camid])
        return data
    
    # #write by hby
    #random choose 50% ids as test
    def prepare_split(self, train_path, split_path):
        if not os.path.exists(split_path):
            print('Creating splits ...')

            img_names = glob(os.path.join(train_path, "*.png"))
            pid_dict = defaultdict(list)
            for img_name in img_names:
                pid = int(img_name.split('/')[-1].split('_')[0])
                pid_dict[pid].append(img_name)
            pids = list(pid_dict.keys())
            num_pids = len(pids)
            assert num_pids == 114, 'There should be 114 identities, ' \
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

                # for train IDs, all images are used in the train set.
                for pid_ in train_pids:
                    img_paths = pid_dict[pid_]
                    pid_ = pid_ - 1
                    pid = self.dataset_name + '_' + str(pid_)
                    for img_path in img_paths:
                        cam = img_path.split('/')[-1].split('_')[1]
                        camid = self.dataset_name + '_' + cam
                        train.append([img_path, pid, camid])

                # for each test ID, choose 2 images, one for query and others for gallery
                for pid_ in test_pids:
                    pid = pid_ - 1
                    img_names = pid_dict[pid_]
                    selected_img_paths = random.sample(img_names, 1)
                    
                    # first image for query
                    camid = int(selected_img_paths[0].split('/')[-1].split('_')[1])
                    query.append([selected_img_paths[0], pid, camid])

                    # other  images for gallery
                    # camid = int(selected_img_paths[1].split('/')[-1].split('_')[1])
                    # gallery.append([selected_img_paths[1], pid, camid])
                    for img_path in img_names:
                        if img_path is not selected_img_paths[0]:
                            camid = int(img_path.split('/')[-1].split('_')[1])
                            gallery.append([img_path, pid, camid])
                
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

