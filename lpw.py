# encoding: utf-8
import os
from glob import glob
from collections import defaultdict
import copy
import random

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['LPW', ]


@DATASET_REGISTRY.register()
class LPW(ImageDataset):
    """LPW
    """
    dataset_dir = "pep_256x128/data_slim"
    dataset_name = "LPW"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        #train = self.process_train(self.train_path)
        train, query, gallery =self.process_train(self.train_path)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, train_path):
        train = []
        query = []
        gallery = []

        # scen 1 for test, scene 2 and scene 3 for training
        # In scene1 , persons in view 2 will be the probe and in other two views will be the gallery
        scene = 'scen2'
        pid_list_2 = []
        cam_list2 = os.listdir(os.path.join(train_path, scene))
        for cam in cam_list2:
            pid_list = os.listdir(os.path.join(train_path, scene, cam))
            pid_list_2.extend(pid_list)
        pid2_set = set(pid_list_2)
        pid2label_2 = {pid: label for label, pid in enumerate(pid2_set, start=0)}
        num_pids_scene2 = len(list(pid2label_2.values())) #1751
        assert num_pids_scene2 == 1751
        
        # print(min(list(pid2label_2.values())))
        # print(max(list(pid2label_2.values())))

        scene = 'scen3'
        pid_list_3 = []
        cam_list3 = os.listdir(os.path.join(train_path, scene))
        for cam in cam_list3:
            pid_list = os.listdir(os.path.join(train_path, scene, cam))
            pid_list_3.extend(pid_list)
        pid3_set = set(pid_list_3)
        pid2label_3 = {pid: label+num_pids_scene2 for label, pid in enumerate(pid3_set)}
        num_pids_scene3 = len(list(pid2label_3.values()))
        #print(num_pids_scene3)
        assert num_pids_scene3 == 224

        # print(min(list(pid2label_3.values())))
        # print(max(list(pid2label_3.values())))
        

        scene = 'scen2'
        cam_list = os.listdir(os.path.join(train_path, scene))
        for cam in cam_list:
            camid = self.dataset_name + "_" + cam[4]
            pid_list = os.listdir(os.path.join(train_path, scene, cam))
            for pid_dir in pid_list:
                img_paths = glob(os.path.join(train_path, scene, cam, pid_dir, "*.jpg"))
                for img_path in img_paths:
                        label = pid2label_2[pid_dir]
                        pid = self.dataset_name + "_" + str(label)
                        train.append((img_path, pid, camid))

        
        scene = 'scen3'
        cam_list = os.listdir(os.path.join(train_path, scene))
        for cam in cam_list:
            camid = self.dataset_name + "_" + cam[4]
            pid_list = os.listdir(os.path.join(train_path, scene, cam))
            for pid_dir in pid_list:
                img_paths = glob(os.path.join(train_path, scene, cam, pid_dir, "*.jpg"))
                for img_path in img_paths:
                        label = pid2label_3[pid_dir]
                        pid = self.dataset_name + "_" + str(label)
                        train.append((img_path, pid, camid))
        
        scene = 'scen1'
        pid_list_1 = []
        cam_list1 = os.listdir(os.path.join(train_path, scene))
        for cam in cam_list1:
            pid_list = os.listdir(os.path.join(train_path, scene, cam))
            pid_list_1.extend(pid_list)
        pid1_set = set(pid_list_1)
        pid2label_1 = {pid: label+num_pids_scene2+num_pids_scene3 for label, pid in enumerate(pid1_set, start=0)}
        num_pids_scene1 = len(list(pid2label_1.values()))
        assert (num_pids_scene1==756)

        scene = 'scen1'
        cam_list = os.listdir(os.path.join(train_path, scene))
        for cam in cam_list:
            camid = int(cam[4])
            if cam=='view2':
                pid_list = os.listdir(os.path.join(train_path, scene, cam))
                for pid_dir in pid_list:
                    img_paths = glob(os.path.join(train_path, scene, cam, pid_dir, "*.jpg"))
                    for img_path in img_paths:
                            label = pid2label_1[pid_dir]
                            query.append((img_path, label, camid))
            else:
                pid_list = os.listdir(os.path.join(train_path, scene, cam))
                for pid_dir in pid_list:
                    img_paths = glob(os.path.join(train_path, scene, cam, pid_dir, "*.jpg"))
                    for img_path in img_paths:
                            label = pid2label_1[pid_dir]
                            gallery.append((img_path, label, camid))

                    
        return train, query, gallery



    
'''
    #random choose 50% ids as test
    def prepare_split(self, train_path, split_path):
        if not os.path.exists(split_path):
            print('Creating splits ...')
            file_path_list = ['scen1', 'scen2', 'scen3']
            pid_dict = defaultdict(list)
            for scene in file_path_list:
                cam_list = os.listdir(os.path.join(train_path, scene))
                for cam in cam_list:
                    camid = self.dataset_name + "_" + cam
                    pid_list = os.listdir(os.path.join(train_path, scene, cam))
                    for pid_dir in pid_list:
                        pid = scene + "-" + pid_dir
                        img_path = glob(os.path.join(train_path, scene, cam, pid_dir, "*.jpg"))
                        pid_dict[pid].extend(img_path)
            
            pids = list(pid_dict.keys())
            num_pids = len(pids)
            assert num_pids == 2731, 'There should be 2731 identities, ' \
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

                pid_int = set()
                for pid_str in pids:
                    if int(pid_str.split('-')[0][4]) == 1:
                        pid_int.add(int(pid_str.split('-')[1])+ 3000)
                    elif int(pid_str.split('-')[0][4]) == 2:
                        pid_int.add(int(pid_str.split('-')[1]) + 6001)
                    elif int(pid_str.split('-')[0][4]) == 3:
                        pid_int.add(int(pid_str.split('-')[1]) + 9002)
                pid2label = {pid: label for label, pid in enumerate(pid_int)}

                # for train IDs, all images are used in the train set.
                for pid_ in train_pids:
                    if int(pid_.split('-')[0][4]) == 1:
                        pid = (int(pid_.split('-')[1]) + 3000)
                    elif int(pid_.split('-')[0][4]) == 2:
                        pid = (int(pid_.split('-')[1]) + 6001)
                    elif int(pid_.split('-')[0][4]) == 3:
                        pid = (int(pid_.split('-')[1]) + 9002)
                    newpid = pid2label[pid]
                    newpid = self.dataset_name + "_" + str(newpid)
                    img_paths = pid_dict[pid_]
                    for img_path in img_paths:
                        #print(img_path)
                        #/data/hby0728/All_ReID_Datasets/clear_all_datasets/pep_256x128/data_slim/scen1/view1/pid/*.jpg
                        cam = img_path.split('/')[8][4] #view1-4
                        camid = self.dataset_name + '_' + cam
                        train.append([img_path, newpid, camid])

                # for each test ID, choose 2 images, one for query and others for gallery
                for pid_ in test_pids:
                    img_names = pid_dict[pid_]
                    selected_img_paths = random.sample(img_names, 1)
                    if int(pid_.split('-')[0][4]) == 1:
                        pid = (int(pid_.split('-')[1]) + 3000)
                    elif int(pid_.split('-')[0][4]) == 2:
                        pid = (int(pid_.split('-')[1]) + 6001)
                    elif int(pid_.split('-')[0][4]) == 3:
                        pid = (int(pid_.split('-')[1]) + 9002)
                    newpid = pid2label[pid]
                    # first image for query
                    camid = int(selected_img_paths[0].split('/')[8][4])
                    query.append([selected_img_paths[0], newpid, camid])

                    # other  images for gallery
                    # camid = int(selected_img_paths[1].split('/')[8][4])
                    # gallery.append([selected_img_paths[1], pid, camid])
                    for img_path in img_names:
                        if img_path is not selected_img_paths[0]:
                            camid = int(img_path.split('/')[8][4])
                            gallery.append([img_path, newpid, camid])
                
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
    
'''
