from itertools import accumulate
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None):

        self.transform = transform

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names' or k == 'receptors':
                self.data[k] = v
                continue

            #sections = np.where(np.diff(data['lig_mask']))[0] + 1 \
            #    if 'lig' in k \
            #    else np.where(np.diff(data['pocket_mask']))[0] + 1
            if 'lig' in k : sections = np.where(np.diff(data['lig_mask']))[0] + 1
            elif 'pocket' in k: sections = np.where(np.diff(data['pocket_mask']))[0] + 1
            elif 'inter' in k:
                t = 0
                sections = []
                if len(data['inter_id'][0])!=0:
                    if data['inter_mask'][0]!=0:
                        for l in range(data['inter_mask'][0]): sections.append(0)
                    for i in np.diff(data['inter_mask']):
                        if i==1: sections.append(t+1)
                        elif i>1: 
                            for d in range(i): sections.append(t+1)
                        t+=1
                    if data['inter_mask'][-1]<len(data['names'])-1:
                        for l in range(len(data['names'])-data['inter_mask'][-1]-1): sections.append(len(data['inter_mask']))

            self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # add number of nodes for convenience
            if k == 'lig_mask':
                self.data['num_lig_atoms'] = \
                    torch.tensor([len(x) for x in self.data['lig_mask']])
            elif k == 'pocket_mask':
                self.data['num_pocket_nodes'] = \
                    torch.tensor([len(x) for x in self.data['pocket_mask']])
            elif k == 'inter_mask':
                num_inter_nodes = \
                    torch.tensor([len(x)*2 for x in self.data['inter_mask']])

        if center:
            for i in range(len(self.data['lig_coords'])):
                mean = (self.data['lig_coords'][i].sum(0) +
                        self.data['pocket_coords'][i].sum(0)) / \
                       (len(self.data['lig_coords'][i]) + len(self.data['pocket_coords'][i]))
                self.data['lig_coords'][i] = self.data['lig_coords'][i] - mean
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean

        # num_pocket - num_inter
        self.data['num_pocket_nodes'] = self.data['num_pocket_nodes'] - num_inter_nodes

    def __len__(self):
        return len(self.data['names'])
    """
    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}
        if self.transform is not None:
            data = self.transform(data)
        return data
    """

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' \
                    or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out
