from torch.utils.data import Dataset
import bidsio
from typing import Optional


class ATLASDataset(Dataset):
    def __init__(self, data_entities: list, target_entities: list, data_derivatives_names: list,
                 target_derivatives_names: list, root_dir: str,
                 transform: Optional[callable] = None, target_transform: Optional[callable] = None):
        self.bids_loader = bidsio.BIDSLoader(
            data_entities=data_entities,
            target_entities=target_entities,
            data_derivatives_names=data_derivatives_names,
            target_derivatives_names=target_derivatives_names,
            batch_size=1,  # Batch size is determined by our PyTorch dataloader, not the BIDS loader.
            root_dir=root_dir
        )
        print(f'Found {len(self.bids_loader)} datapoints')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.bids_loader)

    def __getitem__(self, idx):
        data, target = self.bids_loader.load_sample(idx, data_only=False)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target


data_entities = [{'subject': '',
                  'session': '',
                  'suffix': 'T1w',
                  'space': 'MNI152NLin2009aSym'}]

target_entities = [{'suffix': 'mask',
                    'label': 'L',
                    'desc': 'T1lesion'}]
