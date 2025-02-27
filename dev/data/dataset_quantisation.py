import torch
import numpy as np

from data.dataset_base import BaseDataset
from pm2s.constants import *
from configs import *

class QuantisationDataset(BaseDataset):

    def __init__(self, workspace, split):
        super().__init__(workspace, split, from_asap=False)

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        # onset positions
        onsets = np.round(annotations['onsets_musical'] * N_per_beat)
        onsets[onsets == N_per_beat] = 0 # reset one beat onset to 0

        # note values
        note_values = np.round(annotations['note_value'] * N_per_beat)
        note_values[note_values > max_note_value] = 0 # clip note values to [0, max_note_value], index 0 will be ignored during training

        # padding
        length = len(note_sequence)
        if length < max_length:
            note_sequence = np.concatenate([note_sequence, np.zeros((max_length - length, 4))])
            onsets = np.concatenate([onsets, np.zeros(max_length - length)])
            note_values = np.concatenate([note_values,  np.zeros(max_length - length)])

        return note_sequence, onsets, note_values, length
