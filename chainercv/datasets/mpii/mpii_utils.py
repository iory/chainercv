import numpy as np
import os

from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/mpii'
urls = ['http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz',
        'http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz']


def get_mpii():
    data_root = download.get_dataset_directory(root)
    for url in urls:
        download_file_path = utils.cached_download(url)
        utils.extractall(
            download_file_path, data_root, os.path.splitext(url)[1])
    return data_root
