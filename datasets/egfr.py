import os.path
import os.path as osp
import pdb
import mdtraj
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Any, Callable, List

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset


class EGFR(InMemoryDataset):
    molecule_class = dict({
        '1000C797S_noH': 0,
        '1000G719S_noH': 1,
        '1000L_noH': 1,
        '1000LT_noH': 0,
        '1000T854A_noH': 0,
        '1000WT_noH': 0
    })
    available_molecules = list(molecule_class.keys())

    def __init__(self, root, traj_nums, groups, folds, val_fold_idx=0, transform=None, pre_transform=None, dataset_arg=None, atom_type='all'):
        self.dataset_arg = dataset_arg
        self.traj_nums = traj_nums
        self.groups = groups
        self.folds = folds
        self.atom_type = atom_type
        self.enable_norm = False # Normalization
        self._val_fold_idx = val_fold_idx % folds
        super(EGFR, self).__init__(root, transform, pre_transform)

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
        return super(EGFR, self).get(idx - self.offsets[data_idx])

    @property
    def raw_dir(self) -> list:
        return ['cMet-dimer', 'EGFR-dimer', 'ErbB2-dimer', 'IGFR-dimer']

    @property
    def processed_dir(self) -> str:
        if self.enable_norm:
            return osp.join(self.root, 'processed', self.atom_type + '_' + '{}G_{}F_N3'.format(self.groups, self.folds))
        return osp.join(self.root, 'processed', self.atom_type + '_' + '{}G_{}F'.format(self.groups, self.folds))

    @property
    def raw_file_names(self):
        all_file_names = []
        if isinstance(self.raw_dir, List):
            for file in self.raw_dir:
                file_path = osp.join(self.root, file)
                if osp.isdir(file_path):
                    traj_names = os.listdir(file_path)
                    traj_names.sort()
                    traj_names = [osp.join(file, traj_name) for traj_name in traj_names if traj_name.endswith('pdb')]
                    all_file_names.extend(traj_names)
        return all_file_names

    @property
    def processed_file_names(self):
        return ['{}_{}.pt'.format(self.dataset_arg, file_name.replace("/", "_")[:-4]) for file_name in self.raw_file_names]

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.root, f) for f in files]

    def process(self):
        for idx, (path, processed_path) in enumerate(zip(self.raw_paths, self.processed_paths)):
            print('Processing id:{}, Path:{}'.format(idx, path))
            traj = mdtraj.load_pdb(path)
            if self.atom_type == 'all':
                position = traj.xyz
                topology = traj.topology
                selection = traj.top.select('all')
                graph = nx.Graph()
                graph.add_nodes_from(selection)
                bond_graph = topology.to_bondgraph()
                for edge in bond_graph.edges():
                    atom1, atom2 = edge
                    graph.add_edge(atom1.index, atom2.index)
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
                graph.add_nodes_from(selection)

                for residue in traj.topology.residues:
                    residue_atoms = [atom.index for atom in residue.atoms]
                    intersection = set(residue_atoms).intersection(set(selection))

                    if len(intersection) > 0:
                        for i in range(len(residue_atoms) - 1):
                            for j in range(i + 1, len(residue_atoms)):
                                if residue_atoms[i] in intersection and residue_atoms[j] in intersection:
                                    graph.add_edge(residue_atoms[i], residue_atoms[j])
            else:
                raise NotImplementedError('atom_type {} not implemented'.format(self.atom_type))

            print('Atoms:{}, bonds:{}'.format(graph.number_of_nodes(), graph.number_of_edges()))

            position = torch.from_numpy(position)

            if self.enable_norm:
                position_min, _ = torch.min(position.view(-1, position.shape[-1]), dim=0)
                position_max, _ = torch.max(position.view(-1, position.shape[-1]), dim=0)
                position = 3 * 2 * ((position - position_min) / (position_max - position_min) - 0.5)

            edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[1.0] for _ in range(graph.number_of_edges())])

            # label
            molecule_name = path.split('/')[-1].split('.')[0]
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

    def get_idx_split(self, train_size=18, test_size=6):
        traj_idxs = np.arange(self.traj_nums, dtype=np.int64)
        traj_train_idx, traj_test_idx = traj_idxs[:train_size], traj_idxs[train_size:]

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