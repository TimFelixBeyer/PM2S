import os, sys
sys.path.insert(0, os.path.abspath('./'))
import pandas as pd
import pickle
import torch
from pathlib import Path
from enum import IntEnum, auto
import matplotlib.pyplot as plt

from quantmidi.data.constants import *

# feature_folder = '/import/c4dm-datasets/A2S_transcription/working/workspace/MIDI-quantisation/features'
# metadata = pd.read_csv(str(Path(feature_folder, 'metadata.csv')))
# metadata = metadata[metadata['split'] == 'test']

# ibis = torch.Tensor([])

# for i, row in metadata.iterrows():
#     note_sequence, annotations = pickle.load(open(str(Path(feature_folder, row['feature_file'])), 'rb'))
#     beats = annotations['beats']
#     ibis = torch.cat((ibis, beats[1:] - beats[:-1]))

# ibis = ibis.numpy()

# plt.figure()
# plt.hist(ibis, bins=100)
# plt.savefig('debug.png')


class Action(IntEnum):
    NO_ACTION = auto()
    INSERT_ONE = auto()
    INSERT_TWO = auto()
    INSERT_THREE = auto()
    DELETE_ONE = auto()
    DELETE_TWO = auto()
    DELETE_THREE = auto()
    INSERT_ONE_DELETE_ONE = auto()
    INSERT_ONE_DELETE_TWO = auto()
    INSERT_ONE_DELETE_THREE = auto()
    INSERT_TWO_DELETE_ONE = auto()
    INSERT_TWO_DELETE_TWO = auto()
    INSERT_TWO_DELETE_THREE = auto()
    INSERT_THREE_DELETE_ONE = auto()
    INSERT_THREE_DELETE_TWO = auto()
    INSERT_THREE_DELETE_THREE = auto()

d = {Action.NO_ACTION:0, 1:1}
print(len(Action))
for a in Action:
    if a == 0:
        print(d[a])