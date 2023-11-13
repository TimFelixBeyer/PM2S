from data.dataset_base import BaseDataset
import sys
sys.path.insert(0, "/rhome/beyer/thesis/")
from song import Song
import numpy as np


dataset = BaseDataset("..", "test")
for idx in range(1, 2):#,len(dataset)):
    # Load inputs
    row = dataset._sample_row(idx)
    # Load ground truth
    note_sequence, annotations = dataset._load_data(row)
    print("From MIDI")
    song = Song.from_midi(row["performance_MIDI_external"].replace("{A_MAPS}", "../pm2s/data/A-MAPS_1"))
    print("Now loading MuseScore")
    song_musescore = Song.from_mxl(row["performance_MIDI_external"].replace("{A_MAPS}", ".").replace(".mid", ".mxl"))
    print(song.compare(song_musescore))

    # for ns, a, nm, nm_ms in zip(note_sequence, annotations["note_value"], song.notes_musical, song_musescore.notes_musical):
    #     print(nm, nm_ms, "|", ns, a)
    offset = 0
    open("me.txt", "w").write("\n".join(str(n) for n, ns in zip(song.notes_musical, song.notes_seconds)))
    open("ms.txt", "w").write("\n".join(str((n[0]-offset, n[1], n[2], n[3])) for n, ns in zip(song_musescore.notes_musical, song_musescore.notes_seconds)))
    print(len(song.notes_musical), len(song.notes_musical))
    print(len(song.notes_seconds), len(song_musescore.notes_seconds))
    print("Onset\n - acc", np.mean((np.array(song.notes_musical)[:, 0] + offset) == np.array(song_musescore.notes_musical)[:, 0]))
    print(" - mae", np.mean(np.abs(np.array(song.notes_musical)[:, 0] + offset - np.array(song_musescore.notes_musical)[:, 0])))
    print("Duration\n - acc", np.mean(np.array(song.notes_musical)[:, 1] == np.array(song_musescore.notes_musical)[:, 1]))
    print(" - mae", np.mean(np.abs(np.array(song.notes_musical)[:, 1] - np.array(song_musescore.notes_musical)[:, 1])))