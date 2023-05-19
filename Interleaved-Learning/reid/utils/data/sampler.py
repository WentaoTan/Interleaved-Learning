from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]
def cut_list(list,num_element_each_part):
    n = int(len(list)/num_element_each_part)
    new_list = []
    for i in range(n):
        new_list.append(list[i*num_element_each_part:(i+1)*num_element_each_part])
    return new_list
class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _,_) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.index_cam = defaultdict(list)
        self.num_instances = num_instances

        for index, (fname, pid, cam) in enumerate(data_source):

            if (pid<0): continue
            self.index_pid[index] = pid
            self.index_cam[index] = cam
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)


    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()#in fact,this sampler just samples 4 images each person
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]#fpath,pid,camid,sourceid

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)#the i-th sample's camid is 2(pid2cam is [0,0,1,1,1,2,2,2,5,5]),we select the camid != 2 's inmage to train

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])#we select 4 images from one identity but has different camid

            else:
                select_indexes = No_index(index, i)#if all the last samples belonging to i-th person are from the same camera,we random select self.num_instances images
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)


class RandomMultipleCameraSampler(Sampler):
    def __init__(self, data_source, num_cameras,sample_per_cam = 4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.index_cam = defaultdict(list)
        self.cam_index = defaultdict(list)
        self.pid_cam_index = defaultdict(list)
        self.num_cameras = num_cameras
        self.num_sample_per_cam = sample_per_cam
        for index, (_, pid, cam, _) in enumerate(data_source):
            if (pid<0): continue
            self.index_pid[index] = pid
            self.index_cam[index] = cam
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)
            self.cam_index[cam].append(index)
            # self.pid_cam_index[pid].append((cam,index))
        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)
        self.length = self.num_cameras * self.num_sample_per_cam

    def __len__(self):
        return self.length

    def __iter__(self):
        cam_index = copy.deepcopy(self.cam_index)
        ret = []
        cam_id = list(cam_index.keys())
        random.shuffle(cam_id)
        batch_each_cam = []
        for i in cam_id:
            random.shuffle(cam_index[i])
            cam_index[i] = cut_list(cam_index[i],self.num_sample_per_cam)
            batch_each_cam.append(len(cam_index[i]))
        loop_count = min(batch_each_cam)
        for j in range(loop_count):
            for k in cam_id:
                ret+=(cam_index[k].pop(0))
                # for index in cam_batch:
                #     cur_pid = self.index_pid[index]
                #     person_batch = random.sample(self.pid_index[cur_pid],k=2)

        self.length = len(ret)

        return iter(ret)


class RandomMultipleAggregateSampler(Sampler):
    def __init__(self, data_source, num_instances=4,num_classes = []):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.index_cam = defaultdict(list)
        self.num_instances = num_instances
        self.num_classes = num_classes

        for index, (fname, pid, cam, datsetID) in enumerate(data_source):

            pid = pid + sum(self.num_classes[:datsetID])

            if (pid<0): continue
            self.index_pid[index] = pid
            self.index_cam[index] = cam
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)


    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()#in fact,this sampler just samples 4 images each person
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam, _ = self.data_source[i]#fpath,pid,camid,sourceid

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)#the i-th sample's camid is 2(pid2cam is [0,0,1,1,1,2,2,2,5,5]),we select the camid != 2 's inmage to train

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])#we select 4 images from one identity but has different camid

            else:
                select_indexes = No_index(index, i)#if all the last samples belonging to i-th person are from the same camera,we random select self.num_instances images
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)