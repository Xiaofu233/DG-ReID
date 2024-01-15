# encoding: utf-8
import os
from glob import glob
from collections import defaultdict
import copy
import random

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class WildTrackCrop(ImageDataset):
    """WildTrack.
    Reference:
        WILDTRACK: A Multi-camera HD Dataset for Dense Unscripted Pedestrian Detection
            T. Chavdarova; P. Baqué; A. Maksai; S. Bouquet; C. Jose et al.
    URL: `<https://www.epfl.ch/labs/cvlab/data/data-wildtrack/>`_
    Dataset statistics:
        - identities: 313
        - images: 33979 (train only)
        - cameras: 7
    Args:
        data_path(str): path to WildTrackCrop dataset
        combineall(bool): combine train and test sets as train set if True
    """
    dataset_url = None
    dataset_dir = 'Wildtrack_crop_dataset'
    dataset_name = 'WildTrackCrop'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        self.train_dir = os.path.join(self.dataset_dir, "crop")

        #train = self.process_train(self.train_path)
        train, query, gallery =self.process_data(self.train_dir)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        """
        :param dir_path: directory path saving images
        Returns
            data(list) = [img_path, pid, camid]
        """
        data = []
        for dir_name in os.listdir(dir_path):
            img_lists = glob(os.path.join(dir_path, dir_name, "*.png"))
            for img_path in img_lists:
                pid = self.dataset_name + "_" + dir_name
                camid = img_path.split('/')[-1].split('_')[0]
                camid = self.dataset_name + "_" + camid
                data.append([img_path, pid, camid])
        return data


    #write by hby
    #random choose 50% ids as test
    def prepare_split(self, train_path, split_path):
        if not os.path.exists(split_path):
            print('Creating splits ...')
            pid_str_list = os.listdir(train_path)
            pid_dict = defaultdict(list)
            for pid_str in pid_str_list:
                if 'splits_M3L.json' not in pid_str :
                    pid = int(pid_str)
                    img_lists = glob(os.path.join(train_path, pid_str, "*.png"))
                    pid_dict[pid].extend(img_lists)
            pids = list(pid_dict.keys())
            pids_set = set(pids)
            pid2label = {pid: label for label, pid in enumerate(pids_set)}
            num_pids = len(pids_set)
            image_list = list(pid_dict.values())
            num_images = 0
            for l in image_list:
                #print(len(l))
                num_images += len(l)
            assert num_pids == 313, 'There should be 313 identities, ' \
                                    'but got {}, please check the data'.format(num_pids)
            assert num_images == 33979, 'There should be 33979 images, ' \
                                    'but got {}, please check the data'.format(num_images)

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
                    pid_ = pid2label[pid_]
                    pid = self.dataset_name + '_' + str(pid_)
                    for img_path in img_paths:
                        cam = int(img_path.split('/')[-1].split('_')[0])
                        camid = self.dataset_name + '_' + str(cam)
                        train.append((img_path, pid, camid))

                # for each test ID, choose 2 images, one for query and one for gallery
                for pid_ in test_pids:
                    img_names = pid_dict[pid_]
                    pid = pid2label[pid_]
                    
                    selected_img_paths = random.sample(img_names, 1)
                        
                    # first image for query
                    camid = int(selected_img_paths[0].split('/')[-1].split('_')[0])
                    query.append((selected_img_paths[0], pid, camid))

                    # other  images for gallery
                    # camid = int(selected_img_paths[1].split('/')[-1].split('_')[0])
                    # gallery.append([selected_img_paths[1], pid, camid])
                    for img_path in img_names:
                        if img_path is not selected_img_paths[0]:
                            camid = int(img_path.split('/')[-1].split('_')[0])
                            gallery.append((img_path, pid, camid))
                    
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


