import torch
import pytorch_lightning as pl

from quantmidi.data.dataset import QuantMIDIDataset
from quantmidi.data.constants import (
    batch_size_note_sequence,
    batch_size_baseline,
    batch_size_proposed,
)

class QuantMIDIDataModule(pl.LightningDataModule):
    def __init__(self, feature_folder, model_type, data_aug_args, workers, proposed_model_version):
        super().__init__()
        self.feature_folder = feature_folder
        self.model_type = model_type
        self.data_aug_args = data_aug_args
        self.workers = workers
        self.proposed_model_version = proposed_model_version

        if self.model_type == 'note_sequence':
            self.bs = batch_size_note_sequence
        elif self.model_type == 'baseline':
            self.bs = batch_size_baseline
        elif self.model_type == 'proposed':
            self.bs = batch_size_proposed

    def train_dataloader(self):
        dataset = QuantMIDIDataset(
            self.feature_folder, 
            'train', 
            self.model_type, 
            self.proposed_model_version,
            data_aug_args=self.data_aug_args,
        )
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        train_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=self.bs,
            sampler=sampler,
            num_workers=self.workers,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        dataset = QuantMIDIDataset(self.feature_folder, 'valid', self.model_type, self.proposed_model_version)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        val_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=self.bs,
            sampler=sampler,
            num_workers=self.workers,
            drop_last=True
        )
        return val_loader

    def test_dataloader(self):
        dataset = QuantMIDIDataset(self.feature_folder, 'test', self.model_type, self.proposed_model_version)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        test_loader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            drop_last=False
        )
        return test_loader
