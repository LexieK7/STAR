import os
import h5py
import numpy as np

import open_clip
import torch
import heapq


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


sample_list = os.listdir("./biopsy_wsi_fea_10x/h5_files/")

base_h5_save_dir = "H5 SAVE PATH"
base_pt_save_dir = "PT SAVE PATH"

for i in range(len(sample_list)):

    wsi_sample_path = "./biopsy_wsi_fea_10x/h5_files/" + sample_list[i]
    clus_sample_path = "./biopsy_10x_1024/h5_files/" + sample_list[i]
    print(wsi_sample_path)

    f_wsi = h5py.File(wsi_sample_path)
    wsi_np = np.array(f_wsi['features'][:])
    wsi_cor = np.array(f_wsi['coords'][:])

    f_clus = h5py.File(clus_sample_path)
    clus_np = np.array(f_clus['features'][:])
    clus_cor = np.array(f_clus['coords'][:])

    # according the dimension to create array

    dim_less = clus_np.shape[0] - wsi_np.shape[0]
    # print(dim_less)

    if dim_less < 0:
        print('small')
    else:

        zero_array = np.zeros((dim_less, 256))

        wsi_np = np.concatenate((wsi_np, zero_array), axis=0)

        # concatenate wsi feature and patch feature

        all_np = np.concatenate((wsi_np, clus_np), axis=1)  # concatenate at dim-2

        all_cor = clus_cor

        now_h5_path = base_h5_save_dir + sample_list[i]
        now_pt_path = base_pt_save_dir + sample_list[i].replace('h5', 'pt')

        # save as the same h5
        asset_dict = {'features': all_np, 'coords': all_cor}
        save_hdf5(now_h5_path, asset_dict, attr_dict=None, mode='w')

        # save as the same pt

        features = torch.from_numpy(all_np)
        torch.save(features, now_pt_path)
