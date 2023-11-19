import numpy as np
import json

import torch

from volume_element.VE_dataset import read_structure
from volume_element.VE_model import ModelContainer, SubNet
from volume_element.volume_element import all_element_vertices, align_with_template, collect_snn_area_vertices
import matplotlib.pyplot as plt

def volumeELement_run(cell_path, n_center=30, rotational_align=True):

    center_atom = 'N'

    n_clo = 12
    # loading trained model
    saved_path = "models/energy_model_N.pth"
    model = ModelContainer(SubNet(n_clo, c=20))
    model.model.load_state_dict(torch.load(saved_path))

    # loading trained parameters
    with open("models/model_detail_N.json", 'r') as Nf:
        N_model_info = json.load(Nf)

    ds_train_mean, ds_train_std = np.array(N_model_info["ds_train mean"]), np.array(N_model_info["ds_train std"])
    lat, abc, xyz, atom = read_structure(cell_path)

    # processing volume elements
    all_ele = all_element_vertices(lat, abc, atom,
                                   return_frac=False,
                                   cyclic_sort=True,
                                   shift_to_center=True,
                                   center_atom=center_atom)

    # rotational alignment
    if rotational_align:
        all_ele = align_with_template(all_ele)

    # making prediction
    X = (all_ele[:, :n_clo // 2, :2].reshape(n_center, -1) - ds_train_mean) / (1e-5+ds_train_std)

    en_sum = (-527 / 30) * n_center

    for _x in X:
        _x = torch.from_numpy(_x)
        sub_energy = model.model(_x).item()
        en_sum += sub_energy

    print("Predicted total energy for this system:", en_sum)


if __name__ == '__main__':

    example_path = "sample_BN/sample1_rot55.vasp"
    volumeELement_run(example_path)