import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset



__all__ = ['AirportALERT', ]

@DATASET_REGISTRY.register()
class AirportALERT(ImageDataset):
    """Airport 

    """
    dataset_dir = "AirportALERT"
    dataset_name = "AirportALERT"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)
        self.train_file = os.path.join(self.root, self.dataset_dir, 'filepath.txt')

        required_files = [self.train_file, self.train_path]
        self.check_before_run(required_files)

        #train = self.process_train(self.train_path, self.train_file)
        train, query, gallery = self.process_data(self.train_path, self.train_file)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, dir_path, train_file):
        data = []
        with open(train_file, "r") as f:
            img_paths = [line.strip('\n') for line in f.readlines()]

        for path in img_paths:
            split_path = path.split('\\')
            img_path = '/'.join(split_path)
            camid = self.dataset_name + "_" + split_path[0]
            pid = self.dataset_name + "_" + split_path[1]
            img_path = os.path.join(dir_path, img_path)
            # if 11001 <= int(split_path[1]) <= 401999:
            if 11001 <= int(split_path[1]):
                data.append((img_path, pid, camid))

        return data
    
    # write by hby
    # use the reappearing IDs: image captured by cam37 as query, others as gallery
    def process_data(self, dir_path, train_file):
        data = []
        query = []
        gallery = []
        with open(train_file, "r") as f:
            img_paths = [line.strip('\n') for line in f.readlines()]

        pidset = set()
        for path in img_paths:
            split_path = path.split('\\')
            img_path = '/'.join(split_path)
            camid = int(split_path[0][3:5]) #'cam32'
            pid = int(split_path[1])
            pidset.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pidset)}

        for path in img_paths:
            split_path = path.split('\\')
            img_path = '/'.join(split_path)
            camid = int(split_path[0][3:5]) #'cam32'
            pid = int(split_path[1])
            img_path = os.path.join(dir_path, img_path)
            # if 11001 <= int(split_path[1]):
            #     data.append([img_path, pid, camid])
            if 11001 <= int(split_path[1]) <= 401999:
                
                if split_path[0] == 'cam37':
                    pid = pid2label[pid]
                    query.append([img_path, pid, camid])
                else:
                    pid = pid2label[pid]
                    gallery.append((img_path, pid, camid))
            else:
                pid = pid2label[pid]
                camid = self.dataset_name + "_" + str(camid) #'cam32'
                pid = self.dataset_name + "_" + str(pid)
                data.append((img_path, pid, camid))
        return data, query, gallery



if __name__ == '__main__':
    raw_mat_path = '/data/hby0728/All_ReID_Datasets/clear_all_datasets/AirportALERT/Partition_airport.mat'
    mat = scipy.io.loadmat(raw_mat_path)
    #print(mat['partition'].shape)
    import numpy
    import scipy.io
    import pandas as pd
    data = mat['partition']
    #print(data[0,0][1])
    print(f'length of train samples:{len(data[0,0][0][0])}')
    print(f'length of test samples:{len(data[0,0][1][0])}')
    print(f'length of probe samples:{len(data[0,0][2])}')
    print(f'length of 0th probe samples:{len(data[0,0][2][0])}')
    print(f'length of gallery samples:{len(data[0,0][3])}')
    print(f'length of 0th gallery samples:{len(data[0,0][3][0])}')

    print(sum(data[0,0][0][0]))
    print(sum(data[0,0][1][0]))
    
    numpy.savetxt("./partition.txt", mat['partition'],fmt="%s",  delimiter=",")
