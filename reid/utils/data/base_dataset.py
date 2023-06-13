# encoding: utf-8
import numpy as np
import os

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []

        for items in data:#fpath,pid,camid,domainid
            pid = items[1]
            camid = items[2]

            pids += [pid]
            cams += [camid]

        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery,mix_dataset=None):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)
        if mix_dataset is not None:
            num_mix_pids, num_mix_imgs, num_mix_cams = self.get_imagedata_info(mix_dataset)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        if mix_dataset is not None: print("  mix      | {:5d} | {:8d} | {:9d}".format(num_mix_pids, num_mix_imgs, num_mix_cams))
        print("  ----------------------------------------")
    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))