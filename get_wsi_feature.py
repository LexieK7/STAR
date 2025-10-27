import h5py
import os
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from transformer import Transformer


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


sample_list = os.listdir("./feature/patch_feature/h5_files")
base_h5_save_dir = "./feature/slide_feature/h5_files"
base_pt_save_dir = "./feature/slide_feature/pt_files"

# load model

type_model = Transformer(num_classes=3)
type_weight = torch.load("./model/type.pt")
type_model.load_state_dict(type_weight, strict=False)
type_model.eval()
type_model.cuda()

grade_model = Transformer(num_classes=2)
grade_weight = torch.load("./model/grade.pt")
grade_model.load_state_dict(grade_weight, strict=False)
grade_model.eval()
grade_model.cuda()

invasion_model = Transformer(num_classes=5)
invasion_weight = torch.load("./model/invasion.pt")
invasion_model.load_state_dict(invasion_weight, strict=False)
invasion_model.eval()
invasion_model.cuda()

venous_model = Transformer(num_classes=2)
venous_weight = torch.load("./model/venous.pt")
venous_model.load_state_dict(venous_weight, strict=False)
venous_model.eval()
venous_model.cuda()

perineural_model = Transformer(num_classes=2)
perineural_weight = torch.load("/.model/perineural.pt")
perineural_model.load_state_dict(perineural_weight, strict=False)
perineural_model.eval()
perineural_model.cuda()

# aggragate patch feature.

for i in range(len(sample_list)):
    now_sample = "./feature/patch_feature/h5_files" + sample_list[i]
    print(now_sample)

    f = h5py.File(now_sample)

    patch_np = np.array(f['features'][:])
    patch_cor = np.array(f['coords'][:])
    image_features = torch.from_numpy(patch_np).cuda()

    # print(patch_cor)

    image_features = torch.unsqueeze(image_features, dim=0)

    type_feat = type_model(image_features)
    grade_feat = grade_model(image_features)
    invasion_feat = invasion_model(image_features)
    venous_feat = venous_model(image_features)
    perineural_feat = perineural_model(image_features)

    # print(type_feat.shape, grade_feat.shape, invasion_feat.shape, venous_feat.shape, perineural_feat.shape)
    fea = torch.cat((type_feat, grade_feat, invasion_feat, venous_feat, perineural_feat))
    fea = fea.cpu().detach().numpy()

    cor = np.array([1, 2, 3, 4, 5])  # 1: type, 2: grade, 3: invasion, 4: venous, 5:perineural
    # print(cor.shape)
    # print(fea.shape)

    now_h5_path = base_h5_save_dir + sample_list[i]
    now_pt_path = base_pt_save_dir + sample_list[i].replace('h5', 'pt')

    asset_dict = {'features': fea, 'coords': cor}
    save_hdf5(now_h5_path, asset_dict, attr_dict=None, mode='w')

    # save as the same pt

    features = torch.from_numpy(fea)
    torch.save(features, now_pt_path)




