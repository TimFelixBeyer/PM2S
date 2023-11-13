import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from quantmidi.models.model_utils import ModelUtils
from quantmidi.data.constants import (
    ibiVocab,
    tsNumeVocabSize,
    tsDenoVocabSize,
    keyVocabSize,
    N_per_beat,
    max_note_value,
    onsetVocabSize,
    noteValueVocabSize,
)

learning_rate = 1e-3
dropout = 0.15
hidden_size=512
kernel_size=9
gru_layers=2

class ProposedModel(pl.LightningModule):
    def __init__(self,
        input_features=['pitch', 'onset', 'duration', 'velocity'],
        pitch_encoding='midi',
        onset_encoding='shift-onehot',
        duration_encoding='raw',
        version=1,
    ):
        super().__init__()

        self.input_features = input_features
        self.pitch_encoding = pitch_encoding
        self.onset_encoding = onset_encoding
        self.duration_encoding = duration_encoding
        self.version = version

        in_features = ModelUtils.get_encoding_in_features(input_features, pitch_encoding, onset_encoding, duration_encoding)

        # ======================================================================================
        # Version 1:
        # ======================================================================================
        if self.version == 1:
            # beats, downbeats and tempo
            self.convs_b_db_tempo = ConvBlock(in_features=in_features)
            self.gru_beat = GRUBlock(in_features=hidden_size)
            self.gru_downbeat = GRUBlock(in_features=hidden_size)
            self.gru_tempo = GRUBlock(in_features=hidden_size)
            self.out_beat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
            self.out_downbeat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
            self.out_tempo = LinearOutput(in_features=hidden_size, out_features=ibiVocab, activation_type='softmax')

            # time signatures
            self.conv_time = ConvBlock(in_features=in_features)
            self.gru_time = GRUBlock(in_features=hidden_size)
            self.out_time_nume = LinearOutput(in_features=hidden_size, out_features=tsNumeVocabSize, activation_type='softmax')
            self.out_time_deno = LinearOutput(in_features=hidden_size, out_features=tsDenoVocabSize, activation_type='softmax')

            # key signatures
            self.conv_key = ConvBlock(in_features=in_features)
            self.gru_key = GRUBlock(in_features=hidden_size)
            self.out_key = LinearOutput(in_features=hidden_size, out_features=keyVocabSize, activation_type='softmax')

            # onsets musical and note value
            self.conv_o_v = ConvBlock(in_features=in_features)
            self.gru_onset = GRUBlock(in_features=hidden_size)
            self.gru_value = GRUBlock(in_features=hidden_size)
            self.out_onset = LinearOutput(in_features=hidden_size, out_features=N_per_beat, activation_type='softmax')
            self.out_value = LinearOutput(in_features=hidden_size, out_features=max_note_value+1, activation_type='softmax')

            # hands
            self.conv_hands = ConvBlock(in_features=in_features)
            self.gru_hands = GRUBlock(in_features=hidden_size)
            self.out_hands = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')

        # ======================================================================================
        # Version 2:
        # ======================================================================================
        elif self.version == 2:
            # beats, downbeats and tempo
            self.conv_beat = ConvBlock(in_features=in_features)
            self.conv_downbeat = ConvBlock(in_features=in_features)
            self.conv_tempo = ConvBlock(in_features=in_features)

            self.gru_beat = GRUBlock(in_features=hidden_size+1+ibiVocab)
            self.gru_downbeat = GRUBlock(in_features=hidden_size+ibiVocab)
            self.gru_tempo = GRUBlock(in_features=hidden_size)

            self.out_beat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
            self.out_downbeat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
            self.out_tempo = LinearOutput(in_features=hidden_size, out_features=ibiVocab, activation_type='softmax')

            # time signatures
            self.conv_time = ConvBlock(in_features=in_features)
            self.gru_time = GRUBlock(in_features=hidden_size)
            self.out_time_nume = LinearOutput(in_features=hidden_size, out_features=tsNumeVocabSize, activation_type='softmax')
            self.out_time_deno = LinearOutput(in_features=hidden_size, out_features=tsDenoVocabSize, activation_type='softmax')

            # key signatures
            self.conv_key = ConvBlock(in_features=in_features)
            self.gru_key = GRUBlock(in_features=hidden_size)
            self.out_key = LinearOutput(in_features=hidden_size, out_features=keyVocabSize, activation_type='softmax')

            # hands
            self.conv_hands = ConvBlock(in_features=in_features)
            self.gru_hands = GRUBlock(in_features=hidden_size)
            self.out_hands = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')

            # onsets musical and note value
            self.conv_o = ConvBlock(in_features=in_features)
            self.conv_v = ConvBlock(in_features=in_features)

            self.gru_o = GRUBlock(in_features=hidden_size+noteValueVocabSize)
            self.gru_v = GRUBlock(in_features=hidden_size)

            self.out_o = LinearOutput(in_features=hidden_size, out_features=onsetVocabSize, activation_type='softmax')
            self.out_v = LinearOutput(in_features=hidden_size, out_features=noteValueVocabSize, activation_type='softmax')

    def forward(self, x):
        # x: (batch_size, sequence_length, len(features))

        # ======== Encoding ========
        x_encoded = ModelUtils.encode_input_feature(x, self.input_features, self.pitch_encoding, self.onset_encoding, self.duration_encoding)
        # (batch_size, sequence_length, in_features)

        # ======================================================================================
        # Version 1:
        # ======================================================================================
        if self.version == 1:
            # beats, downbeats and tempo
            x_b_db_tempo = self.convs_b_db_tempo(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_beat = self.gru_beat(x_b_db_tempo)  # (batch_size, sequence_length, hidden_size)
            x_gru_downbeat = self.gru_downbeat(x_b_db_tempo)  # (batch_size, sequence_length, hidden_size)
            x_gru_tempo = self.gru_tempo(x_b_db_tempo)  # (batch_size, sequence_length, hidden_size)
            y_b = self.out_beat(x_gru_beat) # (batch_size, sequence_length, 1)
            y_db = self.out_downbeat(x_gru_downbeat)  # (batch_size, sequence_length, 1)
            y_tempo = self.out_tempo(x_gru_tempo)  # (batch_size, sequence_length, ibiVocab)

            # time signatures
            x_conv_time = self.conv_time(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_time = self.gru_time(x_conv_time)  # (batch_size, sequence_length, hidden_size)
            y_time_nume = self.out_time_nume(x_gru_time)  # (batch_size, sequence_length, tsNumeVocabSize)
            y_time_deno = self.out_time_deno(x_gru_time)  # (batch_size, sequence_length, tsDenoVocabSize)

            # key signatures
            x_conv_key = self.conv_key(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_key = self.gru_key(x_conv_key)  # (batch_size, sequence_length, hidden_size)
            y_key = self.out_key(x_gru_key)  # (batch_size, sequence_length, keyVocabSize)

            # onsets musical and note value
            x_conv_o_v = self.conv_o_v(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_onset = self.gru_onset(x_conv_o_v)  # (batch_size, sequence_length, hidden_size)
            x_gru_value = self.gru_value(x_conv_o_v)  # (batch_size, sequence_length, hidden_size)
            y_onset = self.out_onset(x_gru_onset)  # (batch_size, sequence_length, N_per_beat)
            y_value = self.out_value(x_gru_value)  # (batch_size, sequence_length, max_note_value+1)

            # hands
            x_conv_hands = self.conv_hands(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_hands = self.gru_hands(x_conv_hands)  # (batch_size, sequence_length, hidden_size)
            y_hands = self.out_hands(x_gru_hands)  # (batch_size, sequence_length, 1)

        # ======================================================================================
        # Version 2:
        # ======================================================================================
        elif self.version == 2:
            # beats, downbeats and tempo
            x_conv_tempo = self.conv_tempo(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_tempo = self.gru_tempo(x_conv_tempo)  # (batch_size, sequence_length, hidden_size)
            y_tempo = self.out_tempo(x_gru_tempo)  # (batch_size, sequence_length, ibiVocab)

            x_conv_db = self.conv_downbeat(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_db_input = torch.cat((x_conv_db, y_tempo), dim=2)
            x_gru_db = self.gru_downbeat(x_gru_db_input)  # (batch_size, sequence_length, hidden_size)
            y_db = self.out_downbeat(x_gru_db)  # (batch_size, sequence_length, 1)

            x_conv_b = self.conv_beat(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_b_input = torch.cat((x_conv_b, y_db, y_tempo), dim=2)
            x_gru_b = self.gru_beat(x_gru_b_input)  # (batch_size, sequence_length, hidden_size)
            y_b = self.out_beat(x_gru_b)  # (batch_size, sequence_length, 1)

            # time signatures
            x_conv_time = self.conv_time(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_time = self.gru_time(x_conv_time)  # (batch_size, sequence_length, hidden_size)
            y_time_nume = self.out_time_nume(x_gru_time)  # (batch_size, sequence_length, tsNumeVocabSize)
            y_time_deno = self.out_time_deno(x_gru_time)  # (batch_size, sequence_length, tsDenoVocabSize)

            # key signatures
            x_conv_key = self.conv_key(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_key = self.gru_key(x_conv_key)  # (batch_size, sequence_length, hidden_size)
            y_key = self.out_key(x_gru_key)  # (batch_size, sequence_length, keyVocabSize)

            # hands
            x_conv_hands = self.conv_hands(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_hands = self.gru_hands(x_conv_hands)  # (batch_size, sequence_length, hidden_size)
            y_hands = self.out_hands(x_gru_hands)  # (batch_size, sequence_length, 1)

            # onsets musical and note value
            x_conv_v = self.conv_v(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_v = self.gru_v(x_conv_v)  # (batch_size, sequence_length, hidden_size)
            y_value = self.out_v(x_gru_v)  # (batch_size, sequence_length, noteValueVocabSize)

            x_conv_o = self.conv_o(x_encoded)  # (batch_size, sequence_length, hidden_size)
            x_gru_o_input = torch.cat((x_conv_o, y_value), dim=2)
            x_gru_o = self.gru_o(x_gru_o_input)  # (batch_size, sequence_length, hidden_size)
            y_onset = self.out_o(x_gru_o)  # (batch_size, sequence_length, onsetVocabSize)

        # squeeze and transpose
        y_b = y_b.squeeze(dim=-1)  # (batch_size, sequence_length)
        y_db = y_db.squeeze(dim=-1)  # (batch_size, sequence_length)
        y_tempo = y_tempo.transpose(1, 2)  # (batch_size, ibiVocab, sequence_length)
        y_time_nume = y_time_nume.transpose(1, 2)  # (batch_size, tsNumeVocabSize, sequence_length)
        y_time_deno = y_time_deno.transpose(1, 2)  # (batch_size, tsDenoVocabSize, sequence_length)
        y_key = y_key.transpose(1, 2)  # (batch_size, keyVocabSize, sequence_length)
        y_onset = y_onset.transpose(1, 2)  # (batch_size, N_per_beat, sequence_length)
        y_value = y_value.transpose(1, 2)  # (batch_size, max_note_value+1, sequence_length)
        y_hands = y_hands.squeeze(dim=-1)  # (batch_size, sequence_length)

        return y_b, y_db, y_tempo, y_time_nume, y_time_deno, y_key, y_onset, y_value, y_hands

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=(0.8, 0.8),
            eps=1e-4,
            weight_decay=0.01,
        )
        scheduler_lrdecay = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10000,
            gamma=0.1,
        )
        return [optimizer], [scheduler_lrdecay]

    def configure_callbacks(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_f_beat',
            mode='max',
            save_top_k=3,
            filename='{epoch}-{val_f_beat:.4f}',
            save_last=True,
        )
        earlystop_callback = pl.callbacks.EarlyStopping(
            monitor='val_f_beat',
            patience=200,
            mode='max',
        )
        return [checkpoint_callback]#, earlystop_callback]

    def training_step(self, batch, batch_idx):
        # data
        x, y_b, y_db, y_ibi, y_tn, y_td, y_key, y_o, y_o_m, y_v, y_v_m, y_h, y_h_m, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ibi = y_ibi.long()
        y_tn = y_tn.long()
        y_td = y_td.long()
        y_key = y_key.long()
        y_o = y_o.long()
        y_o_m = y_o_m.float()
        y_v = y_v.long()
        y_v_m = y_v_m.float()
        y_h = y_h.float()
        y_h_m = y_h_m.float()
        length = length.long()

        # predict
        y_b_hat, y_db_hat, y_ibi_hat, y_tn_hat, y_td_hat, y_key_hat, y_o_hat, y_v_hat, y_h_hat = self(x)

        # mask out the padding part (avoid inplace operation)
        mask = torch.ones(y_b_hat.shape).to(y_b_hat.device)
        for i in range(y_b_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_b_hat = y_b_hat * mask
        y_db_hat = y_db_hat * mask
        y_ibi_hat = y_ibi_hat * mask.unsqueeze(1)
        y_tn_hat = y_tn_hat * mask.unsqueeze(1)
        y_td_hat = y_td_hat * mask.unsqueeze(1)
        y_key_hat = y_key_hat * mask.unsqueeze(1)
        y_o_hat = y_o_hat * mask.unsqueeze(1)
        y_v_hat = y_v_hat * mask.unsqueeze(1)
        y_h_hat = y_h_hat * mask
        # apply masks from data loader
        y_o_hat = y_o_hat * y_o_m.unsqueeze(1)
        y_v_hat = y_v_hat * y_v_m.unsqueeze(1)
        y_h_hat = y_h_hat * y_h_m

        # loss
        loss_b = F.binary_cross_entropy(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy(y_db_hat, y_db)
        loss_ibi = nn.NLLLoss(ignore_index=0)(y_ibi_hat, y_ibi)
        loss_tn = nn.NLLLoss(ignore_index=0)(y_tn_hat, y_tn)
        loss_td = nn.NLLLoss(ignore_index=0)(y_td_hat, y_td)#; print(y_v.min(), y_v.max())
        loss_key = nn.NLLLoss()(y_key_hat, y_key)#; input('1')
        loss_o = nn.NLLLoss()(y_o_hat, y_o)#; input('2')
        loss_v = nn.NLLLoss(ignore_index=0)(y_v_hat, y_v)#; input('3')
        loss_h = F.binary_cross_entropy(y_h_hat, y_h)#; input('4')
        loss = loss_b + loss_db + loss_ibi + 0.2 * loss_tn + 0.2 * loss_td + loss_key + loss_o + loss_v + loss_h

        # logs
        logs = {
            'train_loss': loss,
            'train_loss_b': loss_b,
            'train_loss_db': loss_db,
            'train_loss_ibi': loss_ibi,
            'train_loss_tn': loss_tn,
            'train_loss_td': loss_td,
            'train_loss_key': loss_key,
            'train_loss_o': loss_o,
            'train_loss_v': loss_v,
            'train_loss_h': loss_h,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        # data
        x, y_b, y_db, y_ibi, y_tn, y_td, y_key, y_o, y_o_m, y_v, y_v_m, y_h, y_h_m, length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        y_ibi = y_ibi.long()
        y_tn = y_tn.long()
        y_td = y_td.long()
        y_key = y_key.long()
        y_o = y_o.long()
        y_o_m = y_o_m.float()
        y_v = y_v.long()
        y_v_m = y_v_m.float()
        y_h = y_h.float()
        y_h_m = y_h_m.float()
        length = length.long()

        # predict
        y_b_hat, y_db_hat, y_ibi_hat, y_tn_hat, y_td_hat, y_key_hat, y_o_hat, y_v_hat, y_h_hat = self(x)

        # mask out the padding part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            y_db_hat[i, length[i]:] = 0
            y_ibi_hat[i, :, length[i]:] = 0
            y_tn_hat[i, :, length[i]:] = 0
            y_td_hat[i, :, length[i]:] = 0
            y_key_hat[i, :, length[i]:] = 0
            y_o_hat[i, :, length[i]:] = 0
            y_v_hat[i, :, length[i]:] = 0
            y_h_hat[i, length[i]:] = 0
        # apply masks from data loader
        y_o_hat = y_o_hat * y_o_m.unsqueeze(1)
        y_v_hat = y_v_hat * y_v_m.unsqueeze(1)
        y_h_hat = y_h_hat * y_h_m

        # loss
        loss_b = F.binary_cross_entropy(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy(y_db_hat, y_db)
        loss_ibi = nn.NLLLoss(ignore_index=0)(y_ibi_hat, y_ibi)
        loss_tn = nn.NLLLoss(ignore_index=0)(y_tn_hat, y_tn)
        loss_td = nn.NLLLoss(ignore_index=0)(y_td_hat, y_td)
        loss_key = nn.NLLLoss()(y_key_hat, y_key)
        loss_o = nn.NLLLoss()(y_o_hat, y_o)
        loss_v = nn.NLLLoss(ignore_index=0)(y_v_hat, y_v)
        loss_h = F.binary_cross_entropy(y_h_hat, y_h)
        loss = loss_b + loss_db + loss_ibi + 0.2 * loss_tn + 0.2 * loss_td + loss_key + loss_o + loss_v + loss_h

        # metrics
        accs_b, precs_b, recs_b, fs_b = 0, 0, 0, 0
        accs_db, precs_db, recs_db, fs_db = 0, 0, 0, 0

        ave_error_rate_ibi = 0

        precs_macro_tn, recs_macro_tn, fs_macro_tn = 0, 0, 0
        precs_weighted_tn, recs_weighted_tn, fs_weighted_tn = 0, 0, 0

        precs_macro_td, recs_macro_td, fs_macro_td = 0, 0, 0
        precs_weighted_td, recs_weighted_td, fs_weighted_td = 0, 0, 0

        precs_macro_key, recs_macro_key, fs_macro_key = 0, 0, 0
        precs_weighted_key, recs_weighted_key, fs_weighted_key = 0, 0, 0

        precs_macro_o, recs_macro_o, fs_macro_o = 0, 0, 0
        precs_weighted_o, recs_weighted_o, fs_weighted_o = 0, 0, 0
        count_o = 0

        precs_macro_v, recs_macro_v, fs_macro_v = 0, 0, 0
        precs_weighted_v, recs_weighted_v, fs_weighted_v = 0, 0, 0
        count_v = 0

        accs_h, precs_h, recs_h, fs_h = 0, 0, 0, 0
        count_h = 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_b_hat_i = torch.round(y_b_hat[i, :length[i]])
            y_db_hat_i = torch.round(y_db_hat[i, :length[i]])
            y_ibi_hat_i = y_ibi_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_tn_hat_i = y_tn_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_td_hat_i = y_td_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_key_hat_i = y_key_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_o_hat_i = y_o_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_v_hat_i = y_v_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            y_h_hat_i = torch.round(y_h_hat[i, :length[i]])

            y_b_i = y_b[i, :length[i]]
            y_db_i = y_db[i, :length[i]]
            y_ibi_i = y_ibi[i, :length[i]]
            y_tn_i = y_tn[i, :length[i]]
            y_td_i = y_td[i, :length[i]]
            y_key_i = y_key[i, :length[i]]
            y_o_i = y_o[i, :length[i]]
            y_v_i = y_v[i, :length[i]]
            y_h_i = y_h[i, :length[i]]

            # filter out masked indexes from data loader
            y_o_hat_i = y_o_hat_i[y_o_m[i, :length[i]] == 1]
            y_o_i = y_o_i[y_o_m[i, :length[i]] == 1]
            y_v_hat_i = y_v_hat_i[y_v_m[i, :length[i]] == 1]
            y_v_i = y_v_i[y_v_m[i, :length[i]] == 1]
            y_h_hat_i = y_h_hat_i[y_h_m[i, :length[i]] == 1]
            y_h_i = y_h_i[y_h_m[i, :length[i]] == 1]

            # filter out ignore indexes
            y_ibi_hat_i = y_ibi_hat_i[y_ibi_i != 0]
            y_ibi_i = y_ibi_i[y_ibi_i != 0]
            y_tn_hat_i = y_tn_hat_i[y_tn_i != 0]
            y_tn_i = y_tn_i[y_tn_i != 0]
            y_td_hat_i = y_td_hat_i[y_td_i != 0]
            y_td_i = y_td_i[y_td_i != 0]
            y_v_hat_i = y_v_hat_i[y_v_i != 0]
            y_v_i = y_v_i[y_v_i != 0]

            # get accuracy
            acc_b, prec_b, rec_b, f_b = ModelUtils.f_measure_framewise(y_b_i, y_b_hat_i)
            acc_db, prec_db, rec_db, f_db = ModelUtils.f_measure_framewise(y_db_i, y_db_hat_i)
            error_rate_ibi = torch.abs(((y_ibi_hat_i+1e-10) / (y_ibi_i+1e-10)).log()).mean()
            (
                prec_macro_tn,
                rec_macro_tn,
                f1_macro_tn,
                prec_weighted_tn,
                rec_weighted_tn,
                f1_weighted_tn
            ) = ModelUtils.classification_report_framewise(y_tn_i, y_tn_hat_i)
            (
                prec_macro_td,
                rec_macro_td,
                f1_macro_td,
                prec_weighted_td,
                rec_weighted_td,
                f1_weighted_td
            ) = ModelUtils.classification_report_framewise(y_td_i, y_td_hat_i)
            (
                prec_macro_key,
                rec_macro_key,
                f1_macro_key,
                prec_weighted_key,
                rec_weighted_key,
                f1_weighted_key
            ) = ModelUtils.classification_report_framewise(y_key_i, y_key_hat_i)
            if y_o_i.shape[0] > 0:
                (
                    prec_macro_o,
                    rec_macro_o,
                    f1_macro_o,
                    prec_weighted_o,
                    rec_weighted_o,
                    f1_weighted_o
                ) = ModelUtils.classification_report_framewise(y_o_i, y_o_hat_i)
            if y_v_i.shape[0] > 0:
                (
                    prec_macro_v,
                    rec_macro_v,
                    f1_macro_v,
                    prec_weighted_v,
                    rec_weighted_v,
                    f1_weighted_v
                ) = ModelUtils.classification_report_framewise(y_v_i, y_v_hat_i)
            if y_h_i.shape[0] > 0:
                acc_h, prec_h, rec_h, f_h = ModelUtils.f_measure_framewise(y_h_i, y_h_hat_i)

            accs_b += acc_b
            precs_b += prec_b
            recs_b += rec_b
            fs_b += f_b

            accs_db += acc_db
            precs_db += prec_db
            recs_db += rec_db
            fs_db += f_db

            ave_error_rate_ibi += error_rate_ibi

            precs_macro_tn += prec_macro_tn
            recs_macro_tn += rec_macro_tn
            fs_macro_tn += f1_macro_tn
            precs_weighted_tn += prec_weighted_tn
            recs_weighted_tn += rec_weighted_tn
            fs_weighted_tn += f1_weighted_tn

            precs_macro_td += prec_macro_td
            recs_macro_td += rec_macro_td
            fs_macro_td += f1_macro_td
            precs_weighted_td += prec_weighted_td
            recs_weighted_td += rec_weighted_td
            fs_weighted_td += f1_weighted_td

            precs_macro_key += prec_macro_key
            recs_macro_key += rec_macro_key
            fs_macro_key += f1_macro_key
            precs_weighted_key += prec_weighted_key
            recs_weighted_key += rec_weighted_key
            fs_weighted_key += f1_weighted_key

            if y_o_i.shape[0] > 0:
                precs_macro_o += prec_macro_o
                recs_macro_o += rec_macro_o
                fs_macro_o += f1_macro_o
                precs_weighted_o += prec_weighted_o
                recs_weighted_o += rec_weighted_o
                fs_weighted_o += f1_weighted_o
                count_o += 1

            if y_v_i.shape[0] > 0:
                precs_macro_v += prec_macro_v
                recs_macro_v += rec_macro_v
                fs_macro_v += f1_macro_v
                precs_weighted_v += prec_weighted_v
                recs_weighted_v += rec_weighted_v
                fs_weighted_v += f1_weighted_v
                count_v += 1

            if y_h_i.shape[0] > 0:
                accs_h += acc_h
                precs_h += prec_h
                recs_h += rec_h
                fs_h += f_h
                count_h += 1

        # log
        logs = {
            'val_loss': loss,
            'val_loss_b': loss_b,
            'val_loss_db': loss_db,
            'val_loss_ibi': loss_ibi,
            'val_loss_tn': loss_tn,
            'val_loss_td': loss_td,
            'val_loss_key': loss_key,
            'val_loss_o': loss_o,
            'val_loss_v': loss_v,
            'val_loss_h': loss_h,
            'val_acc_beat': accs_b / x.shape[0],
            'val_p_beat': precs_b / x.shape[0],
            'val_r_beat': recs_b / x.shape[0],
            'val_f_beat': fs_b / x.shape[0],
            'val_acc_db': accs_db / x.shape[0],
            'val_p_db': precs_db / x.shape[0],
            'val_r_db': recs_db / x.shape[0],
            'val_f_db': fs_db / x.shape[0],
            'val_ave_error_rate_ibi': ave_error_rate_ibi / x.shape[0],
            'val_p_macro_tn': precs_macro_tn / x.shape[0],
            'val_r_macro_tn': recs_macro_tn / x.shape[0],
            'val_f_macro_tn': fs_macro_tn / x.shape[0],
            'val_p_weighted_tn': precs_weighted_tn / x.shape[0],
            'val_r_weighted_tn': recs_weighted_tn / x.shape[0],
            'val_f_weighted_tn': fs_weighted_tn / x.shape[0],
            'val_p_macro_td': precs_macro_td / x.shape[0],
            'val_r_macro_td': recs_macro_td / x.shape[0],
            'val_f_macro_td': fs_macro_td / x.shape[0],
            'val_p_weighted_td': precs_weighted_td / x.shape[0],
            'val_r_weighted_td': recs_weighted_td / x.shape[0],
            'val_f_weighted_td': fs_weighted_td / x.shape[0],
            'val_p_macro_key': precs_macro_key / x.shape[0],
            'val_r_macro_key': recs_macro_key / x.shape[0],
            'val_f_macro_key': fs_macro_key / x.shape[0],
            'val_p_weighted_key': precs_weighted_key / x.shape[0],
            'val_r_weighted_key': recs_weighted_key / x.shape[0],
            'val_f_weighted_key': fs_weighted_key / x.shape[0],
        }
        if count_o > 0:
            logs.update({
                'val_p_macro_o': precs_macro_o / count_o,
                'val_r_macro_o': recs_macro_o / count_o,
                'val_f_macro_o': fs_macro_o / count_o,
                'val_p_weighted_o': precs_weighted_o / count_o,
                'val_r_weighted_o': recs_weighted_o / count_o,
                'val_f_weighted_o': fs_weighted_o / count_o,
            })
        if count_v > 0:
            logs.update({
                'val_p_macro_v': precs_macro_v / count_v,
                'val_r_macro_v': recs_macro_v / count_v,
                'val_f_macro_v': fs_macro_v / count_v,
                'val_p_weighted_v': precs_weighted_v / count_v,
                'val_r_weighted_v': recs_weighted_v / count_v,
                'val_f_weighted_v': fs_weighted_v / count_v,
            })
        if count_h > 0:
            logs.update({
                'val_acc_h': accs_h / count_h,
                'val_p_h': precs_h / count_h,
                'val_r_h': recs_h / count_h,
                'val_f_h': fs_h / count_h,
            })
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}



class MyModel():
    def __init__(self):
        from pm2s.features.beat import RNNJointBeatProcessor
        from pm2s.features.hand_part import RNNHandPartProcessor
        from pm2s.features.key_signature import RNNKeySignatureProcessor
        from pm2s.features.quantisation import RNNJointQuantisationProcessor
        from pm2s.features.time_signature import RNNTimeSignatureProcessor

        super().__init__()
        processor_beat = RNNJointBeatProcessor(
            "../../_model_state_dicts/beat/RNNJointBeatModel.pth"
        )
        processor_beat._model.eval()
        processor_beat._model.to("cuda")
        self.beat = processor_beat._model
        processor_quantisation = RNNJointQuantisationProcessor(
            "../../_model_state_dicts/quantisation/RNNJointQuantisationModel.pth",
            "../../_model_state_dicts/beat/RNNJointBeatModel.pth",
        )
        processor_quantisation._model.eval()
        processor_quantisation._model.to("cuda")
        self.quant = processor_quantisation._model

        processor_hand = RNNHandPartProcessor(
            "../../_model_state_dicts/hand_part/RNNHandPartModel.pth"
        )
        processor_hand._model.eval()
        processor_hand._model.to("cuda")
        self.hand = processor_hand._model

        processor_time_sig = RNNTimeSignatureProcessor(
            "../../_model_state_dicts/time_signature/RNNTimeSignatureModel.pth"
        )
        processor_time_sig._model.eval()
        processor_time_sig._model.to("cuda")
        self.time_sig = processor_time_sig._model
        processor_key_sig = RNNKeySignatureProcessor(
            "../../_model_state_dicts/key_signature/RNNKeySignatureModel.pth"
        )
        processor_key_sig._model.eval()
        processor_key_sig._model.to("cuda")
        self.key = processor_key_sig._model
    def forward(self, x):
        y_b, y_db, y_tempo = self.beat(x)
        y_b, y_db, y_onset, y_value = self.quant(x)
        y_hands = self.hand(x)
        y_time_nume, y_time_deno = self.time_sig(x)
        y_key = self.key(x)

        return y_b, y_db, y_tempo, y_time_nume, y_time_deno, y_key, y_onset, y_value, y_hands



class ConvBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_size // 4,
                kernel_size=(kernel_size, in_features),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_size // 4,
                out_channels=hidden_size // 2,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_size // 2,
                out_channels=hidden_size,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)

        x = x.unsqueeze(1)   # (batch_size, 1, sequence_length, in_features)
        x = self.conv(x)   # (batch_size, hidden_size, sequence_length, 1)
        x = x.squeeze(3).transpose(1, 2)  # (batch_size, sequence_length, hidden_size)

        return x


class GRUBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.grus_beat = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        x, _ = self.grus_beat(x)  # (batch_size, sequence_length, hidden_size*2)
        x = self.linear(x)  # (batch_size, sequence_length, hidden_size)

        return x


class LinearOutput(nn.Module):
    def __init__(self, in_features, out_features, activation_type='sigmoid'):
        super().__init__()

        self.activation_type = activation_type

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, out_features)
        if activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'softmax':
            self.activation = nn.LogSoftmax(dim=2)
        elif activation_type == 'softplus':
            self.activation = nn.Softplus()

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)

        x = self.dropout(x)  # (batch_size, sequence_length, in_features)
        x = self.linear(x)  # (batch_size, sequence_length, out_features)
        x = self.activation(x)  # (batch_size, sequence_length, out_features)

        return x