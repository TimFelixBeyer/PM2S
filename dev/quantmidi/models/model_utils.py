import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report

from quantmidi.data.constants import resolution, max_length_pr

class ModelUtils():

    @staticmethod
    def input_feature_ablation(x, features):
        """
        Remove features from input.

        Args:
            x (torch.Tensor): Input tensor.
            features (list): List of features to keep.

        Returns:
            torch.Tensor: Tensor with features removed.
        """
        feature2idx = {'pitch': 0, 'onset': 1, 'duration': 2, 'velocity': 3}
        x_list = []
        for feature, idx in feature2idx.items():
            if feature in features:
                x_list.append(x[:,:,idx:idx+1].clone())
        x = torch.cat(x_list, dim=2)
        return x
        
    @staticmethod
    def encode_input_feature(x, features, pitch_encoding, onset_encoding, duration_encoding):
        """
        Encode input features.

        Args:
            x (torch.Tensor): Input tensor.
            features (list): List of features.
            pitch_encoding (str): Encoding of the pitch. Select one from ['midi', 'chroma'].
            onset_encoding (str): Encoding of onset features. Select one from ['absolute-raw', 'shift-raw',
                            'absolute-onehot', 'shift-onehot'].
            duration_encoding (str): Encoding of duration features. Select one from ['raw', 'onehot'].

        Returns:
            torch.Tensor: Tensor with features encoded.
        """
        feature2idx = dict([(feature, idx) for idx, feature in enumerate(features)])
        x_list = []

        # ======== Encode ========
        for feature in features:
            x_feature = x[:,:,feature2idx[feature]].clone()

            # pitch
            if feature == 'pitch':
                if pitch_encoding == 'midi':
                    vocab_size = 128
                elif pitch_encoding == 'chroma':
                    vocab_size = 12
                    x_feature = x_feature % 12
                x_encoded = F.one_hot(x_feature.long(), vocab_size).float()

            # onset
            elif feature == 'onset':
                if onset_encoding == 'absolute-raw':
                    # restart from 0s
                    x_feature = x_feature - x_feature[:,0:1].clone()
                    x_encoded = x_feature.float().unsqueeze(2)

                elif onset_encoding == 'shift-raw':
                    onsets_shift = x_feature[:,1:] - x_feature[:,:-1]
                    # set first onset shift to 0
                    x_encoded = torch.zeros(x.shape[0], x.shape[1], 1).float().to(x.device)  
                    x_encoded[:,1:,0] = onsets_shift

                elif onset_encoding == 'absolute-onehot':
                    # restart from 0s
                    x_feature = x_feature - x_feature[:,0:1].clone()
                    # restart every 20s
                    x_feature = x_feature % 20.0
                    # to one hot index
                    x_feature = torch.round(x_feature / resolution).long()
                    # to one hot
                    x_encoded = F.one_hot(x_feature, int(20.0 / resolution) + 1).float()
                    
                elif onset_encoding == 'shift-onehot':
                    onsets_shift_raw = x_feature[:,1:] - x_feature[:,:-1]
                    # set first onset shift to 0
                    onsets_shift = torch.zeros(x.shape[0], x.shape[1]).float().to(x.device)
                    onsets_shift[:,1:] = onsets_shift_raw
                    # maximum filter - set maximum onset shift to 4s
                    onsets_shift_filted = onsets_shift.clone()
                    onsets_shift_filted[onsets_shift > 4.0] = 4.0
                    # remove negative values introduced by zero padding
                    onsets_shift_filted[onsets_shift < 0.0] = 0.0
                    # to one hot index
                    onsets_shift_idx = torch.round(onsets_shift_filted / resolution).long()
                    # to one hot
                    x_encoded = F.one_hot(onsets_shift_idx, int(4.0 / resolution) + 1).float()

            # duration
            elif feature == 'duration':
                if duration_encoding == 'raw':
                    x_encoded = x_feature.float().unsqueeze(2)
                elif duration_encoding == 'onehot':
                    # maximum filter - set maximum duration to 4s
                    durations_filted = x_feature.clone()
                    durations_filted[durations_filted > 4.0] = 4.0
                    # to one hot index
                    durations_idx = torch.round(durations_filted / resolution).long()
                    # to one hot
                    x_encoded = F.one_hot(durations_idx, int(4.0 / resolution) + 1).float()

            # velocity
            elif feature == 'velocity':
                x_encoded = x_feature.float().unsqueeze(2) / 127.0

            x_list.append(x_encoded)

        x = torch.cat(x_list, dim=2)
        return x

    @staticmethod
    def get_encoding_in_features(features, pitch_encoding, onset_encoding, duration_encoding):
        """
        Get encoding in features.

        Args:
            features (list): List of features.
            pitch_encoding (str): Encoding of pitch features. Select one from ['midi', 'chroma'].
            onset_encoding (str): Encoding of onset features. Select one from ['absolute-raw', 'shift-raw',
                            'absolute-onehot', 'shift-onehot'].
            duration_encoding (str): Encoding of duration features. Select one from ['raw', 'onehot'].

        Returns:
            int: Number of in_features after encoding.
        """
        feature2idx = dict([(feature, idx) for idx, feature in enumerate(features)])
        in_features = 0

        for feature in features:
            # pitch
            if feature == 'pitch':
                if pitch_encoding == 'midi':
                    in_features += 128
                elif pitch_encoding == 'chroma':
                    in_features += 12

            # onset
            elif feature == 'onset':
                if onset_encoding == 'absolute-raw':
                    in_features += 1
                elif onset_encoding == 'shift-raw':
                    in_features += 1
                elif onset_encoding == 'absolute-onehot':
                    in_features += int(20.0 / resolution) + 1  # maximum 20s
                elif onset_encoding == 'shift-onehot':
                    in_features += int(4.0 / resolution) + 1   # maximum 4s

            # duration
            elif feature == 'duration':
                if duration_encoding == 'raw':
                    in_features += 1
                elif duration_encoding == 'onehot':
                    in_features += int(4.0 / resolution) + 1   # maximum 4s

            # velocity
            elif feature == 'velocity':
                in_features += 1

        return in_features

    @staticmethod
    def get_pianoroll_from_batch_data(x, length):
        # x.shape = (batch_size, length, len(features))
        assert x.shape[2] == 4, 'Number of features must be 4.'

        pr = torch.zeros(x.shape[0], 128, max_length_pr).float().to(x.device)

        for bi in range(x.shape[0]):
            t0 = x[bi, 0, 1]
            for ni in range(x.shape[1]):
                pitch = x[bi, ni, 0]
                onset = x[bi, ni, 1] - t0
                offset = x[bi, ni, 1] + x[bi, ni, 2] - t0
                velocity = x[bi, ni, 3] / 127.0

                start = min(max_length_pr, max(0, int(torch.round(onset / resolution))))
                end = min(max_length_pr, max(0, int(torch.round(offset / resolution))))
                pr[bi, pitch.long(), start:end] = velocity

        return pr

    @staticmethod
    def f_measure_framewise(y, y_hat):
        acc = (y_hat == y).float().mean()
        TP = torch.logical_and(y_hat==1, y==1).float().sum()
        FP = torch.logical_and(y_hat==1, y==0).float().sum()
        FN = torch.logical_and(y_hat==0, y==1).float().sum()

        p = TP / (TP + FP + np.finfo(float).eps)
        r = TP / (TP + FN + np.finfo(float).eps)
        f = 2 * p * r / (p + r + np.finfo(float).eps)
        return acc, p, r, f

    @staticmethod
    def classification_report_framewise(y, y_hat):
        report = classification_report(y.tolist(), y_hat.tolist(), output_dict=True)

        prec_macro = report['macro avg']['precision']
        rec_macro = report['macro avg']['recall']
        f1_macro = report['macro avg']['f1-score']
        prec_weighted = report['weighted avg']['precision']
        rec_weighted = report['weighted avg']['recall']
        f1_weighted = report['weighted avg']['f1-score']
        
        return prec_macro, rec_macro, f1_macro, prec_weighted, rec_weighted, f1_weighted