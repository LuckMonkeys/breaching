
from cProfile import label
from cmath import pi
from fileinput import filename
from genericpath import isdir
import re
import json
from utils_img import calculate_psnr 
from typing import List
import os
from pathlib import Path
from collections import defaultdict
import shutil


# match_0_396 = get_losses_from_log('/home/zx/Gitrepo/breaching/outputs/single_imagenet_resnet18_optimization_SR_zeros_pattern_None_1000_1/2022-09-13/09-38-41/single_imagenet_resnet18_optimization_SR_zeros_pattern_None_1000_1_data_generate.log')
# match_396_1000 = get_losses_from_log('/home/zx/Gitrepo/breaching/outputs/single_imagenet_resnet18_optimization_SR_zeros_pattern_None_1000_1/2022-09-14/01-46-37/single_imagenet_resnet18_optimization_SR_zeros_pattern_None_1000_1_data_generate.log')

# match_0_396.update(match_396_1000) 
# match_0_1000 = match_0_396
# filenames = filter_img_by_task_loss(match_0_1000)
    
# dir_H = '/home/zx/data/GitRepo/KAIR-master/trainsets/trainH'
# dir_L = '/home/zx/data/GitRepo/KAIR-master/trainsets/trainL'


# total_filenames = [f'{i}.png' for i in range(1000)]

# print(filenames) 
# psnrs = cal_psrn(dir_H=dir_H, dir_L=dir_L, filenames=filenames)
# psnrs = cal_psrn(dir_H=dir_H, dir_L=dir_L, filenames=total_filenames)
# import math
# print(mean(psnrs))
# reverse_delete_files(dir=dir_H, filenames=filenames)
# reverse_delete_files(dir=dir_L, filenames=filenames)


#-----------------------------------------------------------------------------------------------------------------------------------------------
##read lq images filenames, filter out filenames based on task loss, get common filenames with hq_5000 images, move move images to KAIR trainset
#-----------------------------------------------------------------------------------------------------------------------------------------------

# from utils_img import read_filenames_from_log, get_common_filenames, copy_file

# normal_dir = '/root/data/GitRepo/breaching/rec_datasets/normal/'
# scale_4 = '/root/data/GitRepo/breaching/rec_datasets/scale_4'

# dst_dir = '/root/data/GitRepo/KAIR-master/trainsets'

# log_dir = '/root/data/GitRepo/breaching/outputs/single_imagenet_resnet18_optimization_SR_zeros_pattern_None_1000_1/2022-09-26'
# lq_filenames_dict = read_filenames_from_log(log_dir)
# get_common_filenames(lq_filenames_dict=lq_filenames_dict, normal_dir=normal_dir)

# total_data_size = 1200
# data_size =0
# for key, value in lq_filenames_dict.items():
#     for img_name in value:
#         if data_size >= total_data_size:
#             exit(0)
#         lq_src = scale_4 + f'/{key}/lq_1000/{img_name}'
#         lq_dst = dst_dir + f'/trainL/{key}_{img_name}'
#         copy_file(lq_src, lq_dst) 

#         h5000_src = normal_dir + f'/{key}/lq_5000/{img_name}'
#         h5000_dst = dst_dir + f'/trainH_5000/{key}_{img_name}'
#         copy_file(h5000_src, h5000_dst) 

#         data_size += 1
    




#-----------------------------------
##calculate task loss for hq5000
#-----------------------------------

# dir = '/home/zx/Gitrepo/breaching/rec_datasets/normal/'
# dir = '/home/zx/Gitrepo/breaching/rec_datasets/scale_4'

# imgs_dict = defaultdict(list)
# for idx_dir in Path(dir).iterdir():
#     # idx_dir = Path(os.path.join(dir, '0'))
#     idx_dir = Path(dir) 
#     if idx_dir.is_dir():
#         lq_dir = idx_dir / 'lq_1000'
#         for img_path in lq_dir.iterdir():
#             if img_path.name.endswith('.png'):
#                 labels = int(img_path.name.split('.')[0])
#                 t_labels = torch.tensor([labels])
#                 loss = get_task_loss(model=model, loss_fn=loss_fn, labels=t_labels, img_path=str(img_path), transforms=transforms)
#                 if loss <= 0.001:
#                     imgs_dict[idx_dir.name].append(labels)
#     break
# # imgs_dict['0'].sort()
# imgs_dict['scale_4'].sort()

# # print(len(imgs_dict['scale_4']))
# print(imgs_dict['scale_4'])


# from utils_img import copy_file
# home_dir = str(Path.home())
# L_dir = home_dir + '/data/GitRepo/KAIR-master/trainsets/trainL'
# for img_path in Path(L_dir).iterdir():
#     k, name = img_path.name.split('_')

#     src_path = home_dir + '/data/GitRepo/breaching/rec_datasets/scale_4' + f'/{k}/hq/{name}'
#     dst_path = home_dir + '/data/GitRepo/KAIR-master/trainsets/trainH_original' + f'/{img_path.name}'
#     print(src_path, dst_path)
#     # break

#     copy_file(src=src_path, dst=dst_path)

    
# from utils_img import resize_img_and_save

# resize_img_and_save(home_dir + '/data/GitRepo/KAIR-master/trainsets/trainH_original', home_dir + '/data/GitRepo/KAIR-master/trainsets/trainH_original_scaled', (56,56))



# print(Path().home())



#-----------------------------------
##Handle Celeba Hq attribute list file
#-----------------------------------


# hq_list = '/home/zx/data/celeba_hq/image_list.txt'
# celeba_attr_list = '/home/zx/data/celeba/list_attr_celeba.txt'

# hq_attr_list = '/home/zx/data/celeba_hq/list_attr_celeba.txt' 
# hq_partition_list = '/home/zx/data/celeba_hq/list_eval_partition.txt' 

# #read celeba attr list
# import csv
# from collections import namedtuple
# from typing import Any, Callable, List, Optional, Tuple, Union

# CSV_celeba = namedtuple("CSV", ["header", "index", "data"])
# CSV_chq = namedtuple("CSV", ["header", "index", "ori_filenames"])

# def load_celeba_csv(
#     filename: str,
#     header: Optional[int] = None,
# ) -> CSV_celeba:
#     with open(filename) as csv_file:
#         data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

#     if header is not None:
#         headers = data[header]
#         data = data[header + 1 :]
#     else:
#         headers = []

#     indices = [row[0] for row in data]
#     data = [row[1:] for row in data]
#     # data = [[row[1], row[3], row[5], row[7], row[9], row[2], row[4], row[6] , row[8], row[10]]for row in data]
#     # data_int = [list(map(int, i)) for i in data]

#     return CSV_celeba(headers, indices,data)


# def load_chq_csv(
#     filename: str,
#     header: Optional[int] = None,
# ) -> CSV_chq:
#     with open(filename) as csv_file:
#         data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

#     if header is not None:
#         headers = data[header]
#         data = data[header + 1 :]
#     else:
#         headers = []

#     indices = [row[0] for row in data]
#     ori_filenames = [row[2] for row in data]

#     return CSV_chq(headers, indices,ori_filenames)


# hq_head, hq_indices, hq_ori_filenames = load_chq_csv(hq_list, header=0)    
# c_head, c_filenames, c_data =  load_celeba_csv(celeba_attr_list, header=1)

# # '00001.jpg':['1', '-1', ......]
# celeba_name_data_dict = {}
# for fname, data in zip(c_filenames, c_data):
#     celeba_name_data_dict[fname] = data


# # '0':['1', '-1', ......]
# hq_data = defaultdict(list)
# for hq_idx, ori_fname in zip(hq_indices, hq_ori_filenames):
#     hq_data[ hq_idx ] = celeba_name_data_dict[ori_fname]



# #Write list_attr txt
# with open(hq_attr_list, 'w') as f:
#     fnames = list(hq_data.keys())
#     #total nubmer
#     f.write(str(len(fnames)) + '\n')
#     #meta attributes name
#     f.write(' '.join(c_head) + '\n')

#     # 00001.jpg 1  -1, ......
#     contents = []
#     for i in range(1, len(fnames)+1):
#         save_name = (5-len(str(i)))*'0'  + f'{int(i)}.jpg'
#         data = hq_data[str(i-1)] 
#         data.insert(0, save_name)

#         content = ' '.join(data) + '\n'
#         contents.append(content)
#     f.writelines(contents)


# # write list_eval_partition txt
# with open(hq_partition_list, 'w') as f:
#     ds_num = len(list(hq_data.keys()))

#     # 00001.jpg 0 .... 24001.jpg 1....27001 jpg 2
#     contents = []
#     for i in range(1, ds_num+1):
#         save_name = (5-len(str(i)))*'0'  + f'{int(i)}.jpg'

#         if i < ds_num * 0.8:
#             partition = 0
#         elif i < ds_num * 0.9:
#             partition = 1
#         elif i < ds_num:
#             partition = 2
            
#         content = f'{save_name} {partition}' + '\n'
#         contents.append(content)
#     f.writelines(contents)

    


import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL
import torch

from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

CSV = namedtuple("CSV", ["header", "index", "data"])



class CelebaHQ_Gender(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba_hq"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        # ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        # ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        # ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        # if download:
        #     self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        # identity = self._load_csv("identity_CelebA.txt")
        # bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        # landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        # self.identity = identity.data[mask]
        # self.bbox = bbox.data[mask]
        # self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        return os.path.isdir(os.path.join(self.root, self.base_folder, "data256"))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "data256", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target[20].item()


    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)



    
dataset = CelebaHQ_Gender('/home/zx/data', split='train')
print(dataset, len(dataset))
