import numpy as np
import json
from pymatgen.io.vasp import Poscar
# from area_element.symmetry import radial_sym, angular_sym, pairwise_dist, pairwise_vec

from torch.utils.data import Dataset

def read_structure(fname:str):
    abc, xyz, atom = [], [], []
    poscar = Poscar.from_file(fname)
    lattice = poscar.as_dict()['structure']['lattice']['matrix']
    for site in poscar.as_dict()['structure']['sites']:
        abc.append(site['abc'])
        xyz.append(site['xyz'])
        atom.append(site['label'])
    return np.array(lattice), np.array(abc), np.array(xyz), atom

'''
Outdated dataset, now for generating features only
'''
class CoordsEnergyDataset(Dataset):
    def __init__(self, structure_paths, label_paths, length=4450):
        self.structure_paths = structure_paths
        self.length = length

        self.label_path = label_paths
        with open(self.label_path) as json_file:
            self.label_dict = json.load(json_file)
        self.energy = np.full(self.length, np.nan, dtype=float)
        self.perturb_strength = np.full(self.length, np.nan, dtype=float)

        for key in self.label_dict.keys():
            self.energy[int(key)] = self.label_dict[key].get("total_energy", np.nan)
            self.perturb_strength[int(key)] = self.label_dict[key].get("perturb_strength", np.nan)

    def __getitem__(self, index):

        #spath = f'../../Volume-Element-Project/data/AE_root/perturb_{index}/POSCAR'
        spath = f'../../Volume-Element-Project/data/' + self.label_dict[str(index)]["path"] +'/POSCAR'

        lat, abc, xyz, atom = read_structure(spath)
        label = self.energy[index]
        return lat, abc, xyz, atom, label

    def __len__(self):

        return self.length


'''
Symmetry function-Energy Dataset

asym_df (pandas dataframe): angular symmetry
rsym_df (pandas dataframe): radial symmetry
label_df (pandas dataframe): energy symmetry
shuffle_X (bool): shuffle the atoms order every time fetching X

Iterating it returns (X, y), where X first column is anuglar, X second column is radial, y is energy
'''
class SymEnergyDataset(Dataset):
    def __init__(self, asym_df, rsym_df, label_df, shuffle_X=True):
        self.asym_df = asym_df
        self.rsym_df = rsym_df
        self.energy = label_df['total_energy'].values
        self.perturb_strength = label_df['ps'].values
        self.length = len(self.energy)
        self.shuffle_X = shuffle_X

    def normalize_X(self, mode='standard', pre_as_mean=None, pre_as_std=None, pre_as_max=None, pre_as_min=None, pre_rs_mean=None, pre_rs_std=None, pre_rs_max=None, pre_rs_min=None):
        if mode == "standard":
            self.min_shift = 1e-5
            self.as_mean = pre_as_mean if pre_as_mean is not None else self.asym_df.mean()
            self.as_std = pre_as_std if pre_as_std is not None else self.asym_df.std()
            self.rs_mean = pre_rs_mean if pre_rs_mean is not None else self.rsym_df.mean()
            self.rs_std = pre_rs_std if pre_rs_std is not None else self.rsym_df.std()
            self.asym_df = (self.asym_df - self.as_mean) / (self.as_std + self.min_shift)
            self.rsym_df = (self.rsym_df - self.rs_mean) / (self.rs_std + self.min_shift)
        elif mode == "standard_allaxis":
            self.as_mean = pre_as_mean if pre_as_mean is not None else np.nanmean(self.asym_df)
            self.as_std = pre_as_std if pre_as_std is not None else np.nanstd(self.asym_df)
            self.rs_mean = pre_rs_mean if pre_rs_mean is not None else np.nanmean(self.rsym_df)
            self.rs_std = pre_rs_std if pre_rs_std is not None else np.nanstd(self.rsym_df)

            self.asym_df = (self.asym_df - self.as_mean) / self.as_std
            self.rsym_df = (self.rsym_df - self.rs_mean) / self.rs_std
        elif mode == "minmax":
            self.min_shift = 1e-5
            self.asym_df = (self.asym_df - self.asym_df.min()) / (self.asym_df.max() - self.asym_df.min()) + self.min_shift
            self.rsym_df = (self.rsym_df - self.rsym_df.min()) / (self.rsym_df.max() - self.rsym_df.min()) + self.min_shift
        elif mode == "minmax_allaxis":
            self.asym_df = (self.asym_df - self.asym_df.min().min()) / (self.asym_df.max().max() - self.asym_df.min().min()) + self.min_shift
            self.rsym_df = (self.rsym_df - self.rsym_df.min().min()) / (self.rsym_df.max().max() - self.rsym_df.min().min()) + self.min_shift
        else:
            print("Unrecognized method")

    def normalize_y(self, mode='standard', offset=0, negate=False):
        if negate:
            self.energy = -self.energy
        
        if mode == "standard":
            self.en_std = np.nanstd(self.energy)
            self.en_mean = np.nanmean(self.energy)
            self.energy = (self.energy - self.en_mean) / self.en_std
        elif mode == "minmax":
            self.min_shift = 1e-5
            self.en_min = np.nanmin(self.energy)
            self.en_max = np.nanmax(self.energy)
            self.energy = (self.energy - self.en_min) / (self.en_max - self.en_min) + self.min_shift
        elif mode == "offset":
            self.energy += offset
        else:
            print("Unrecognized method, not normalization applied")

    def __getitem__(self, index):
        label = self.energy[index]
        a = self.asym_df.loc[index].to_numpy()
        r = self.rsym_df.loc[index].to_numpy()
        X = np.append(a, r).reshape(27, -1).T

        if self.shuffle_X:
            np.random.shuffle(X)
        return X, label

    def __len__(self):
        if self.length:
            return self.length
        else:
            return len(self.asym_df)

'''
Volume Element features-Energy Dataset

f_df (pandas dataframe): features dataframe (can be vertices coords, PCA)
label_df (pandas dataframe): energy symmetry
n_col (int): takes the first n columns of f_df only as X
n_ve (int): number of volume elements per structure
shuffle_X (bool): shuffle the volume element order every time fetching X

Iterating it returns (X, y), where X first column is anuglar, X second column is radial, y is energy
'''
class VEFeaturesEnergyDataset(Dataset):
    def __init__(self, f_df, label_df, n_col, n_ve=30, shuffle_X=True):
        self.f_df = f_df.iloc[:, 0:n_col]
        self.energy = label_df['total_energy'].values
        self.perturb_strength = label_df['ps'].values
        self.n_ve = n_ve
        self.length = len(self.energy)
        self.shuffle_X = shuffle_X

    def normalize_X(self, mode='standard', pre_mean=None, pre_std=None, pre_max=None, pre_min=None):
        if mode == "standard":
            self.pca_mean = pre_mean if pre_mean is not None else self.f_df.mean()
            self.pca_std = pre_std if pre_std is not None else self.f_df.std()
            self.f_df = (self.f_df - self.pca_mean) / self.pca_std
        elif mode == "minmax":
            self.min_shift = 1e-5
            self.pca_max = pre_max if pre_max is not None else self.f_df.max()
            self.pca_min = pre_min if pre_min is not None else self.f_df.min()
            self.f_df = (self.f_df - self.pca_min) / (self.pca_max - self.pca_min) + self.min_shift
        # elif mode == ""
        else:
            print("Unrecognized method, not normalization applied")

    def normalize_y(self, mode='standard', offset=0, negate=False):
        if negate:
            self.energy = -self.energy
        
        if mode == "standard":
            self.en_std = np.nanstd(self.energy)
            self.en_mean = np.nanmean(self.energy)
            self.energy = (self.energy - self.en_mean) / self.en_std
        elif mode == "minmax":
            self.min_shift = 1e-5
            self.en_min = np.nanmin(self.energy)
            self.en_max = np.nanmax(self.energy)
            self.energy = (self.energy - self.en_min) / (self.en_max - self.en_min) + self.min_shift
        elif mode == "offset":
            self.energy += offset
        else:
            print("Unrecognized method, not normalization applied")

    def __getitem__(self, index):
        label = self.energy[index]
        f_tmp = self.f_df.iloc[index*self.n_ve:(index+1)*self.n_ve].values
        if self.shuffle_X:
            np.random.shuffle(f_tmp)
        return f_tmp, label

    def __len__(self):
        return self.length


class VEFeaturesEnergyStressDataset(Dataset):
    def __init__(self, f_df, label_df, n_col, n_ve=30, shuffle_X=False, spring_model=False):
        self.unpertrubed_template = np.array([-7.25126348e-01, -1.25636927e+00, -1.45044589e+00, 2.86353100e-06,
                                              -7.25126709e-01, 1.25637367e+00, 7.25511424e-01, 1.25637240e+00,
                                              1.45083049e+00, 1.89193682e-06, 7.25511784e-01, -1.25636959e+00])
        self.f_clo = ['fx', 'fy']
        self.f_df = f_df.iloc[:, 0:n_col]
        print("Load data: ", self.f_df.shape)
        self.forces = f_df[self.f_clo]
        self.energy = label_df['total_energy'].values
        self.perturb_strength = label_df['ps'].values
        self.n_ve = n_ve
        self.length = len(self.energy)
        self.min_shift = 1e-5
        self.shuffle_X = shuffle_X
        self.spring_model = spring_model

        if self.spring_model: self.delta_xs = self.f_df - self.unpertrubed_template


    def normalize_X(self, mode='standard', pre_mean=None, pre_std=None, pre_max=None, pre_min=None):
        if mode == "standard":
            self.pca_mean = pre_mean if pre_mean is not None else self.f_df.mean()
            self.pca_std = pre_std if pre_std is not None else self.f_df.std()
            self.f_df = (self.f_df - self.pca_mean) / (self.pca_std+self.min_shift)

        elif mode == "minmax":

            self.pca_max = pre_max if pre_max is not None else self.f_df.max()
            self.pca_min = pre_min if pre_min is not None else self.f_df.min()
            self.f_df = (self.f_df - self.pca_min) / ((self.pca_max - self.pca_min) + self.min_shift)
        # elif mode == ""
        else:
            print("Unrecognized method, not normalization applied")

    def normalize_y(self, mode='standard', offset=0, negate=False):
        if negate:
            self.energy = -self.energy

        if mode == "standard":
            self.en_std = np.nanstd(self.energy)
            self.en_mean = np.nanmean(self.energy)
            self.energy = (self.energy - self.en_mean) / (self.en_std)
        elif mode == "minmax":
            self.min_shift = 1e-5
            self.en_min = np.nanmin(self.energy)
            self.en_max = np.nanmax(self.energy)
            self.energy = (self.energy - self.en_min) / ((self.en_max - self.en_min) + self.min_shift)
        elif mode == "offset":
            self.energy += offset
        else:
            print("Unrecognized method, not normalization applied")

    def __getitem__(self, index):
        label = self.energy[index]
        f_tmp = self.f_df.iloc[index * self.n_ve:(index + 1) * self.n_ve].values
        forces = self.forces.iloc[index*self.n_ve:(index+1)*self.n_ve].values
        if self.shuffle_X:
            np.random.shuffle(f_tmp)

        if self.spring_model:
            delta_xs = self.delta_xs.iloc[index * self.n_ve:(index + 1) * self.n_ve].values
            return f_tmp, delta_xs, forces, label
        else:
            return f_tmp, forces, label

    def __len__(self):
        return self.length

