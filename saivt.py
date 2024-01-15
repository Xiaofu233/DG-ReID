# encoding: utf-8
import os
from glob import glob
import copy
import random
from collections import defaultdict
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['SAIVT', ]


@DATASET_REGISTRY.register()
class SAIVT(ImageDataset):
    """SAIVT
    """
    dataset_dir = "SAIVT-SoftBio"
    dataset_name = "SAIVT"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        #train = self.process_train(self.train_path)
        train, query, gallery = self.process_data(self.train_path)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, train_path):
        data = []

        pid_path = os.path.join(train_path, "cropped_images")
        pid_list = os.listdir(pid_path)

        for pid_name in pid_list:
            pid = self.dataset_name + '_' + pid_name
            img_list = glob(os.path.join(pid_path, pid_name, "*.jpeg"))
            for img_path in img_list:
                img_name = os.path.basename(img_path)
                camid = self.dataset_name + '_' + img_name.split('-')[2]
                data.append([img_path, pid, camid])
        return data
    
    #write by hby
    #random choose 50% ids as test
    def prepare_split(self, train_path, split_path):
        if not os.path.exists(split_path):
            print('Creating splits ...')

            pid_path = os.path.join(train_path, "cropped_images")
            pid_list = os.listdir(pid_path)
            #pids = [int(p) for p in pid_list]
            num_pids = len(pid_list)
            assert num_pids == 152, 'There should be 152 identities, ' \
                                    'but got {}, please check the data'.format(num_pids)

            num_train_pids = int(num_pids * 0.5)

            splits = []
            for _ in range(10):
                # randomly choose num_train_pids train IDs and the rest for test IDs
                pids_copy = copy.deepcopy(pid_list)
                random.shuffle(pids_copy)
                train_pids = pids_copy[:num_train_pids]
                test_pids = pids_copy[num_train_pids:]
                #print(test_pids)
                
                train = []
                query = []
                gallery = []

                # for train IDs, all images are used in the train set.
                for pid_ in train_pids:
                    pid = self.dataset_name + '_' + pid_
                    img_list = glob(os.path.join(pid_path, pid_, "*.jpeg"))
                    for img_path in img_list:
                        img_name = os.path.basename(img_path)
                        cam = img_name.split('-')[2]
                        camid = self.dataset_name + '_' + cam
                        train.append([img_path, pid, camid])

                # for each test ID, choose 2 images, one for query and one for gallery
                '''
                for pid_ in test_pids:
                    #pid = self.dataset_name + '_' + pid_
                    pid = int(pid_)
                    img_list = glob(os.path.join(pid_path, pid_, "*.jpeg"))
                    #print(len(img_list))
                    choose_list = random.sample(img_list, 2)
                    
                    # first image for query
                    img_name = os.path.basename(choose_list[0])
                    camid = int(img_name.split('-')[2][4])
                    query.append([choose_list[0], pid, camid])

                    # other  images for gallery
                    img_name = os.path.basename(choose_list[1])
                    camid = int(img_name.split('-')[2][4])
                    gallery.append([choose_list[1], pid, camid])
                '''
                # for each test ID, choose one for query and all others for gallery
                for pid_ in test_pids:
                    pid = int(pid_)
                    img_list = glob(os.path.join(pid_path, pid_, "*.jpeg"))
                    # choose one image for query
                    choose_list = random.sample(img_list, 1)
                    img_name = os.path.basename(choose_list[0])
                    camid = int(img_name.split('-')[2][4])
                    query.append([choose_list[0], pid, camid])
                    for img_path in img_list:
                        if img_path is not choose_list[0]:
                            img_name = os.path.basename(img_path)
                            camid = int(img_name.split('-')[2][4])
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
