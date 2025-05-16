import os.path
import os.path as osp
import pdb
import time
import mdtraj
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Any, Callable, List
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset

class GPCR(InMemoryDataset):
    molecule_class = dict({
        '10193_trj_12.xtc': 0,
        '10341_trj_30.xtc': 1,
        '10358_trj_32.xtc': 1,
        '10374_trj_34.xtc': 1,
        '10386_trj_35.xtc': 0,
        '10438_trj_41.xtc': 0,
        '10457_trj_43.xtc': 0,
        '10470_trj_45.xtc': 1,
        '10488_trj_47.xtc': 1,
        '10504_trj_49.xtc': 0,
        '10529_trj_52.xtc': 0,
        '10612_trj_61.xtc': 0,
        '10647_trj_65.xtc': 0,
        '10673_trj_68.xtc': 1,
        '10705_trj_72.xtc': 1,
        '10713_trj_73.xtc': 0,
        '10799_trj_82.xtc': 1,
        '10916_trj_96.xtc': 0,
        '10953_trj_100.xtc': 0,
        '11054_trj_111.xtc': 0,
        '11073_trj_113.xtc': 1,
        '11085_trj_114.xtc': 1,
        '11105_trj_116.xtc': 1,
        '11171_trj_122.xtc': 1,
        '11181_trj_123.xtc': 0,
        '11423_trj_151.xtc': 1,
        # '11763_trj_161.xtc': 0,
    })
    available_molecules = list(molecule_class.keys())

    def __init__(self, root, traj_nums, groups, folds, val_fold_idx=0, transform=None, pre_transform=None, dataset_arg=None, atom_type='all'):
        self.dataset_arg = dataset_arg
        self.traj_nums = traj_nums
        self.groups = groups
        self.folds = folds
        self.atom_type = atom_type
        self.enable_norm = False # Normalization
        self.remove_hydrogen = True
        self._val_fold_idx = val_fold_idx % folds
        super(GPCR, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1])

    def len(self):
        return sum(len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all)

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(GPCR, self).get(idx - self.offsets[data_idx])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.atom_type + '_' + '{}G_{}F'.format(self.groups, self.folds))
        # return osp.join(self.root, 'processed', self.atom_type + '_' + '{}G_{}F_N3'.format(self.groups, self.folds)) # _noH

    @property
    def raw_file_names(self):
        file_names = os.listdir(self.raw_dir)
        file_names.sort()
        traj_names = [file_name for file_name in file_names if file_name.endswith('xtc')]
        top_names = [file_name for file_name in file_names if file_name.endswith('psf') or file_name.endswith('pdb')]
        assert len(traj_names) == len(top_names), "Traj files do not match top files"
        return traj_names, top_names

    @property
    def processed_file_names(self):
        return ['{}_{}.pt'.format(self.dataset_arg, file_name.replace("/", "_")[:-4]) for file_name in self.raw_file_names[0]]

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        traj_names, top_names = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(traj_names, Callable):
            traj_names = traj_names()
        traj_files = [osp.join(self.raw_dir, traj_name) for traj_name in traj_names]
        top_files = [osp.join(self.raw_dir, top_name) for top_name in top_names]

        return traj_files, top_files

    def process(self):
        for idx, (traj_path, top_path, processed_path) in enumerate(zip(*self.raw_paths, self.processed_paths)):
            print('Processing id:{}, Traj path:{}, Top path:{}'.format(idx, traj_path, top_path))
            # if idx < 18: continue
            # Step 1: load traj
            traj = mdtraj.load_xtc(traj_path, top=top_path)

            # Step 2.1: remove water
            water_indices = traj.topology.select('resname HOH')  # =water, ≠ resname WAT
            non_water_mask = [i for i in range(traj.n_atoms) if i not in water_indices]
            print('Original atoms:{}, remove water:{}, kept atoms:{}'.format(traj.n_atoms, len(water_indices),
                                                                             traj.n_atoms - len(water_indices)))
            # Step 2.1: remove hydrogen
            # hydrogen_indices = traj.topology.select('name H')
            # non_water_mask = [i for i in range(traj.n_atoms) if
            #                   i not in water_indices and i not in hydrogen_indices]
            # print('Original atoms:{}, remove water:{} and hydrogen:{}, kept atoms:{}'.format(traj.n_atoms,
            # len(water_indices), len(hydrogen_indices), traj.n_atoms-len(water_indices)-len(hydrogen_indices)))

            traj = traj.atom_slice(non_water_mask)

            if self.atom_type == 'all':
                position = traj.xyz
                topology = traj.topology
                for atom in topology.atoms:
                    atom.name = str(atom.index)  # Rename the atom to its 0-based index
                graph = topology.to_bondgraph()
            elif self.atom_type == 'alpha_carbon':
                alpha_carbon_indices = traj.topology.select('name CA')
                # alpha_carbon_indices = traj.topology.select('name CA and protein')

                # Calculate distances between alpha-carbon atoms
                atom_pairs = [(atom_a, atom_b) for atom_a in alpha_carbon_indices for atom_b in alpha_carbon_indices]
                distances = mdtraj.compute_distances(traj[0], atom_pairs=atom_pairs)
                distances = distances.reshape(len(alpha_carbon_indices), len(alpha_carbon_indices))

                graph = nx.Graph()
                for i, idx in enumerate(alpha_carbon_indices):
                    graph.add_node(idx, atom_index=idx, residue_index=traj.topology.atom(idx).residue.index)

                # Add edges between connected alpha-carbon atoms based on distance criteria
                cutoff_distance = 5.0  # Adjust this cutoff distance as needed
                for i, idx1 in enumerate(alpha_carbon_indices):
                    for j, idx2 in enumerate(alpha_carbon_indices):
                        if i != j and distances[i, j] < cutoff_distance:
                            graph.add_edge(idx1, idx2)
                position = traj.xyz[:, alpha_carbon_indices]
            elif self.atom_type == 'residue':
                position = []
                graph = nx.Graph()
                for residue in traj.topology.residues:
                    # Extract information for the residue
                    residue_index, residue_name = residue.index, residue.name
                    atom_indices = traj.topology.select('residue {}'.format(residue_index))
                    if len(atom_indices) > 0:
                        graph.add_node(residue_index, residue_name=residue_name)
                        # print('Residue {}: {} : Traj: {}, Atoms: {}'.format(residue_index, residue_name, traj.xyz[:, atom_indices, :].shape, atom_indices))
                        position.append(np.mean(traj.xyz[:, atom_indices, :], axis=1, keepdims=True))

                # Iterate through the bonds in the topology
                for bond in traj.topology.bonds:
                    atom1_idx, atom2_idx = bond[0].index, bond[1].index
                    residue1_idx = traj.topology.atom(atom1_idx).residue.index
                    residue2_idx = traj.topology.atom(atom2_idx).residue.index
                    if graph.has_node(residue1_idx) and graph.has_node(residue2_idx):
                        graph.add_edge(residue1_idx, residue2_idx)
                position = np.concatenate(position, axis=1)
            elif self.atom_type == 'backbone':
                selection = traj.top.select('protein and backbone')
                position = traj.xyz[:, selection]
                graph = nx.Graph()
                atom_idx_new = np.arange(len(selection))
                graph.add_nodes_from(atom_idx_new) # selection
                atom_mapping = dict(zip(selection, atom_idx_new))

                for residue in traj.topology.residues:
                    residue_atoms = [atom.index for atom in residue.atoms]
                    intersection = set(residue_atoms).intersection(set(selection))

                    if len(intersection) > 0:
                        for i in range(len(residue_atoms) - 1):
                            for j in range(i + 1, len(residue_atoms)):
                                if residue_atoms[i] in intersection and residue_atoms[j] in intersection:
                                    # graph.add_edge(residue_atoms[i], residue_atoms[j])
                                    graph.add_edge(atom_mapping[residue_atoms[i]], atom_mapping[residue_atoms[j]])

            else:
                raise NotImplementedError('atom_type {} not implemented'.format(self.atom_type))

            print('Atoms:{}, bonds:{}'.format(graph.number_of_nodes(), graph.number_of_edges()))

            position = torch.from_numpy(position)

            if self.enable_norm:
                position_min, _ = torch.min(position.contiguous().view(-1, position.shape[-1]), dim=0)
                position_max, _ = torch.max(position.contiguous().view(-1, position.shape[-1]), dim=0)
                position = 3 * 2 * ((position - position_min) / (position_max - position_min) - 0.5)

            edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[1.0] for _ in range(graph.number_of_edges())])

            # label
            molecule_name = traj_path.split('/')[-1]
            y = torch.tensor([self.molecule_class[molecule_name]]).long().unsqueeze(0)

            # split into sequences
            position_split = torch.split(position, position.size(0) // (self.groups * self.folds))

            if position_split[0].size(0) != position_split[-1].size(0):
                position_split = position_split[:-1]

            samples = []
            for sub_position in tqdm(position_split, total=len(position_split)):
                sub_position = torch.permute(sub_position, (1, 0, 2)) # num_nodes, time_stamps, node_attr_dim
                data = Data(pos=sub_position, edge_index=edge_index, edge_attr=edge_attr, y=y)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)

    @property
    def val_fold_idx(self):
        return self._val_fold_idx

    def get_idx_split(self, train_size=20, test_size=6):
        traj_idxs = np.arange(self.traj_nums, dtype=np.int64)
        traj_test_idx = np.array([13, 15, 22, 17, 25, 11]) # 5, 7
        traj_train_idx = np.array([idx for idx in traj_idxs if idx not in traj_test_idx])

        traj_test_idx = traj_test_idx * self.groups * self.folds
        traj_trainval_idx = traj_train_idx * self.groups * self.folds

        shift_idx = np.arange(self.groups * self.folds, dtype=int)
        test_idx = np.repeat(shift_idx[np.newaxis, ...], len(traj_test_idx), axis=0) + traj_test_idx[..., np.newaxis]
        trainval_idx = np.repeat(shift_idx[np.newaxis, ...], len(traj_trainval_idx), axis=0) + traj_trainval_idx[..., np.newaxis]
        trainval_idx = trainval_idx.reshape(trainval_idx.shape[0], self.groups, self.folds)

        train_fold_idx = list(np.arange(self.folds))
        train_fold_idx.remove(self.val_fold_idx)

        train_idx = trainval_idx[:, :, train_fold_idx]
        val_idx = trainval_idx[:, :, self.val_fold_idx]
        train_idx = train_idx.reshape(-1)
        val_idx = val_idx.reshape(-1)
        test_idx = test_idx.reshape(-1)

        return train_idx, val_idx, test_idx


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    data_root = '/hdd2/qtx/Datasets/GPCR-2'
    data_reg = 'gpcr'
    all_type = 'backbone'
    traj_nums = 26
    batch_size = 8
    num_workers = 0
    groups = 5
    folds = 5

    start_time = time.time()
    gpcr_dataset = GPCR(data_root, traj_nums=traj_nums, groups=groups, folds=folds, dataset_arg=data_reg, atom_type=all_type)
    # print(egfr_dataset.raw_file_names)
    for idx, file_name in enumerate(gpcr_dataset.raw_file_names[0]):
        print(idx, file_name)

    idx_train, id_val, idx_test = gpcr_dataset.get_idx_split()
    pdb.set_trace()
    train_loader = DataLoader(Subset(gpcr_dataset, idx_train), batch_size=16, shuffle=True, drop_last=False)
    val_loader = DataLoader(Subset(gpcr_dataset, idx_train), batch_size=groups, shuffle=False)
    test_loader = DataLoader(Subset(gpcr_dataset, idx_test), batch_size=groups * folds, shuffle=False)

    print('Load cost time:{}'.format(time.time() - start_time)) # 1828.675179719925 431.7890655994415

    for epoch in range(1):
        for idx, data in enumerate(train_loader):
            cur_x, cur_y = data.pos, data.y
            print('Epoch:{} || Iteration:{}, data shape:{}, label shape:{}'.format(epoch, idx, cur_x.shape, cur_y.shape))
            # print(torch.min(cur_x), torch.max(cur_x))
            # print(cur_x)
            print(cur_y)

    # for idx, data in enumerate(val_loader):
    #     cur_x, cur_y = data.pos, data.y
    #     print('Val || Iteration:{}, data shape:{}, label shape:{}'.format(idx, cur_x.shape, cur_y.shape))
    #     # print(torch.min(cur_x), torch.max(cur_x))
    #     # print(cur_x)
    #     print(cur_y)
    #
    # for idx, data in enumerate(test_loader):
    #     cur_x, cur_y = data.pos, data.y
    #     print('Test || Iteration:{}, data shape:{}, label shape:{}'.format(idx, cur_x.shape, cur_y.shape))
    #     # print(torch.min(cur_x), torch.max(cur_x))
    #     # print(cur_x)
    #     print(cur_y)
    # """

# ['cMet-dimer/1000C797S_noH.pdb', 'cMet-dimer/1000G719S_noH.pdb', 'cMet-dimer/1000LT_noH.pdb', 'cMet-dimer/1000L_noH.pdb', 'cMet-dimer/1000T854A_noH.pdb', 'cMet-dimer/1000WT_noH.pdb',
# 'EGFR-dimer/1000C797S_noH.pdb', 'EGFR-dimer/1000G719S_noH.pdb', 'EGFR-dimer/1000LT_noH.pdb', 'EGFR-dimer/1000L_noH.pdb', 'EGFR-dimer/1000T854A_noH.pdb', 'EGFR-dimer/1000WT_noH.pdb',
# 'ErbB2-dimer/1000C797S_noH.pdb', 'ErbB2-dimer/1000G719S_noH.pdb', 'ErbB2-dimer/1000LT_noH.pdb', 'ErbB2-dimer/1000L_noH.pdb', 'ErbB2-dimer/1000T854A_noH.pdb', 'ErbB2-dimer/1000WT_noH.pdb',
# 'IGFR-dimer/1000C797S_noH.pdb', 'IGFR-dimer/1000G719S_noH.pdb', 'IGFR-dimer/1000LT_noH.pdb', 'IGFR-dimer/1000L_noH.pdb', 'IGFR-dimer/1000T854A_noH.pdb', 'IGFR-dimer/1000WT_noH.pdb']