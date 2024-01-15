# encoding: utf-8
"""
''
"""

import os
from scipy.io import loadmat
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
import pdb

__all__ = ['GRID',]


@DATASET_REGISTRY.register()
class GRID(ImageDataset):
    dataset_dir = "grid"
    dataset_name = 'GRID'

    def __init__(self, root='datasets', split_id = 0, **kwargs):

        if isinstance(root, list):
            split_id = root[1]
            self.root = root[0]
        else:
            self.root = root
            split_id = 0
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        self.probe_path = os.path.join(
            self.dataset_dir, 'probe'
        )
        self.gallery_path = os.path.join(
            self.dataset_dir, 'gallery'
        )
        self.split_mat_path = os.path.join(
            self.dataset_dir, 'features_and_partitions.mat'
        )
        self.split_path = os.path.join(self.dataset_dir, 'splits.json')

        required_files = [
            self.dataset_dir, self.probe_path, self.gallery_path,
            self.split_mat_path
        ]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = self.read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = [tuple([os.path.join(self.dataset_dir, item[0])] + item[1:]) for item in train]
        query = [tuple([os.path.join(self.dataset_dir, item[0])] + item[1:]) for item in query]
        gallery = [tuple([os.path.join(self.dataset_dir, item[0])] + item[1:]) for item in gallery]
        # train = [tuple(item) for item in train]
        # query = [tuple(item) for item in query]
        # gallery = [tuple(item) for item in gallery]

        super(GRID, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not os.path.exists(self.split_path):
            print('Creating 10 random splits')
            split_mat = loadmat(self.split_mat_path)
            trainIdxAll = split_mat['trainIdxAll'][0] # length = 10
            probe_img_paths = sorted(
                glob(os.path.join(self.probe_path, '*.jpeg'))
            )
            gallery_img_paths = sorted(
                glob(os.path.join(self.gallery_path, '*.jpeg'))
            )

            splits = []
            for split_idx in range(10):
                train_idxs = trainIdxAll[split_idx][0][0][2][0].tolist()
                assert len(train_idxs) == 125
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, query, gallery = [], [], []

                # processing probe folder
                for img_path in probe_img_paths:
                    img_name = os.path.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        # add by hby, for train
                        # pid = self.dataset_name + "_" + str(idx2label[img_idx])
                        # camid = self.dataset_name + "_" + str(camid)
                        train.append((os.path.relpath(img_path, self.dataset_dir), img_idx, camid))
                    else:
                        query.append((os.path.relpath(img_path, self.dataset_dir), img_idx, camid))

                # process gallery folder
                for img_path in gallery_img_paths:
                    img_name = os.path.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        # add by hby, for train
                        # pid = self.dataset_name + "_" + str(idx2label[img_idx])
                        # camid = self.dataset_name + "_" + str(camid)
                        train.append((os.path.relpath(img_path, self.dataset_dir), img_idx, camid))
                    else:
                        gallery.append((os.path.relpath(img_path, self.dataset_dir), img_idx, camid))
                    
                all_pid = []
                for img_path, pid, camid in train:
                    all_pid.append(pid)
                for img_path, pid, camid in query:
                    all_pid.append(pid)
                for img_path, pid, camid in gallery:
                    all_pid.append(pid)
                all_pid = set(all_pid)
                #print(len(all_pid))
                assert len(all_pid) == 251
                all_id2label = {pid: label for label, pid in enumerate(all_pid)}
                final_query = []
                final_gallery = []
                final_train = []
                for img_path, pid, camid in train:
                    pid = self.dataset_name + "_" + str(all_id2label[pid]-1)
                    camid = self.dataset_name + "_" + str(camid)
                    final_train.append((img_path, pid, camid))
                for img_path, pid, camid in query:
                    final_query.append((img_path, all_id2label[pid] - 1, camid))
                i=250
                for img_path, pid, camid in gallery:
                    if all_id2label[pid] == 0:
                        new_pid = i
                        i = i+1
                        final_gallery.append((img_path, new_pid, camid))
                    else:
                        final_gallery.append((img_path, all_id2label[pid] - 1, camid))
                    
                split = {
                    'train': final_train,
                    'query': final_query,
                    'gallery': final_gallery,
                    'num_train_pids': 125,
                    'num_query_pids': 125,
                    'num_gallery_pids': 900
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            self.write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))


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