import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from BaseModel import BaseModel


# Sequence to sequence (direct correspondence between input and output) module
# A very similar one can be implemented with LSTM layers instead of GRU
class GRUseq2seq(BaseModel):
    def __init__(self, n_features, hid_dim, n_layers, dropout, learning_rate, batch_size, results_directory,
                 gpu_id=None, bidirectional=False, task='classification', num_classes=1, checkpoints_directory=None):
        super(GRUseq2seq, self).__init__(model_name="GRUseq2seq", checkpoints_directory=checkpoints_directory)
        self.n_features = n_features
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.results_directory = results_directory
        self.gpu_id = gpu_id
        self.bidirectional = bidirectional
        self.task = task  # classification or regression
        self.num_classes = num_classes  # only used if the task is classification. if it is a binary classification: 1
        self.checkpoints_directory = checkpoints_directory

        self.gru = nn.GRU(input_size=n_features, hidden_size=hid_dim, num_layers=n_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.d = 2 if bidirectional else 1
        self.fc_out = nn.Linear(hid_dim * self.d, num_classes if task == 'classification' else n_features)

        # Set loss function based on task_type
        if task == 'classification':
            self.criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, x, lengths):
        # Pack the padded sequence (expects inputs in shape [batch_size, seq_len, input_size])
        # print(f"x before packing:{x}")
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # print(f"packed x:{packed_x}")

        # Pass through GRU layers
        packed_output, _ = self.gru(packed_x)

        # Unpack to apply the fully connected
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # FC
        output = self.fc_out(output)

        return output

    def training_step(self, batch, batch_idx):
        X, Y, lengths = batch
        # lengths = [len(x) for x in X]
        output = self(X, lengths)
        loss = self.criterion(output, Y)
        self.log('train_loss', loss, prog_bar=True)
        # if task=='classification':
        # accuracy = ....
        # self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, lengths = batch
        output = self(X, lengths)
        loss = self.criterion(output, Y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler] if scheduler else [optimizer]


class GRUseq2one(BaseModel):
    def __init__(self, n_features, hid_dim, n_layers, dropout, learning_rate, batch_size, results_directory,
                 gpu_id=None, bidirectional=False, task='classification', num_classes=1, checkpoints_directory=None):
        super(GRUseq2one, self).__init__(model_name="GRUseq2one", checkpoints_directory=checkpoints_directory)
        self.n_features = n_features
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.results_directory = results_directory
        self.gpu_id = gpu_id
        self.bidirectional = bidirectional
        self.task = task  # classification or regression
        self.num_classes = num_classes  # only used if the task is classification.
        # if it is a binary classification, num_classes should be 1

        self.gru = nn.GRU(input_size=n_features, hidden_size=hid_dim, num_layers=n_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.d = 2 if bidirectional else 1
        # For seq2one, the FC layer outputs one value or class per sample (not per timestep)
        self.fc_out = nn.Linear(hid_dim * self.d, num_classes if task == 'classification' else 1)

        # Set loss function based on task_type
        if task == 'classification':
            self.criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, x, lengths):
        # Pack the padded sequence (expects inputs in shape [batch_size, seq_len, input_size])
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through GRU layers
        packed_output, _ = self.gru(packed_x)

        # Unpack to get the final hidden state of the last timestep
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Take only the output of the last time step for each sequence
        last_outputs = output[torch.arange(output.size(0)), lengths - 1]  # Select the last hidden state

        # Pass through fully connected layer (for classification or regression)
        output = self.fc_out(last_outputs)

        return output

    def training_step(self, batch, batch_idx):
        X, Y, lengths = batch
        output = self(X, lengths)
        loss = self.criterion(output, Y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, lengths = batch
        output = self(X, lengths)
        loss = self.criterion(output, Y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        return optimizer


class GRUEncoderDecoder(BaseModel):
    def __init__(self, n_features, enc_hid_dim, dec_hid_dim, enc_layers, dec_layers, dropout, learning_rate,
                 batch_size, results_directory, gpu_id=None, bidirectional=False, checkpoints_directory=None):
        """
        GRU Encoder-Decoder model for sequence-to-sequence regression tasks.

        :param n_features: Number of input features per time step.
        :param enc_hid_dim: Hidden dimension for the encoder.
        :param dec_hid_dim: Hidden dimension for the decoder.
        :param enc_layers: Number of layers in the encoder.
        :param dec_layers: Number of layers in the decoder.
        :param dropout: Dropout value.
        :param learning_rate: Learning rate for the optimizer.
        :param batch_size: Batch size.
        :param results_directory: Directory to save the results.
        :param gpu_id: GPU ID for training, if available.
        :param bidirectional: Whether the encoder is bidirectional.
        """
        super(GRUEncoderDecoder, self).__init__(model_name="GRUEncoderDecoder",
                                                checkpoints_directory=checkpoints_directory)
        self.n_features = n_features
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.results_directory = results_directory
        self.gpu_id = gpu_id
        self.bidirectional = bidirectional
        self.d = 2 if bidirectional else 1  # Double the hidden size if the encoder is bidirectional

        # Encoder GRU
        self.encoder = nn.GRU(input_size=n_features, hidden_size=enc_hid_dim, num_layers=enc_layers,
                              bidirectional=bidirectional, batch_first=True, dropout=dropout)

        # Decoder GRU
        self.decoder = nn.GRU(input_size=n_features, hidden_size=dec_hid_dim, num_layers=dec_layers,
                              batch_first=True, dropout=dropout)

        # Fully connected output layer to map hidden states to output features
        self.fc_out = nn.Linear(dec_hid_dim, n_features)

        self.criterion = nn.MSELoss()  # For regression tasks

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, src, tgt, src_lengths, tgt_lengths):
        """
        Forward pass through the GRU Encoder-Decoder model.
        :param src: Input sequence of shape [batch_size, src_seq_len, n_features].
        :param tgt: Target sequence of shape [batch_size, tgt_seq_len, n_features].
        :param src_lengths: Lengths of each sequence in the batch (for packing).
        :param tgt_lengths: Lengths of each target sequence (optional, could be ignored).
        :return: Reconstructed sequence of shape [batch_size, tgt_seq_len, n_features].
        """

        # Pack the source sequence for the encoder
        packed_src = pack_padded_sequence(src, src_lengths, batch_first=True, enforce_sorted=False)

        # Pass through the encoder
        packed_output, hidden = self.encoder(packed_src)

        # For the decoder, we need the last hidden state from the encoder
        # If encoder is bidirectional, we need to concatenate the forward and backward hidden states
        if self.bidirectional:
            hidden = self._concat_bidirectional_hidden(hidden)

        # Unpack the packed sequence (to apply dropout)
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # We can now run the decoder, starting with the hidden state from the encoder
        # Assuming `tgt` has been provided for teacher forcing
        packed_tgt = pack_padded_sequence(tgt, tgt_lengths, batch_first=True, enforce_sorted=False)

        # Pass through the decoder using the encoder's last hidden state
        packed_dec_output, _ = self.decoder(packed_tgt, hidden)

        # Unpack the decoder outputs
        dec_output, _ = pad_packed_sequence(packed_dec_output, batch_first=True)

        # Final fully connected layer to map decoder hidden states to predicted output
        output = self.fc_out(dec_output)

        return output

    @staticmethod
    def _concat_bidirectional_hidden(hidden):
        """Concatenate forward and backward hidden states for bidirectional GRU."""
        # The hidden state has shape [num_layers * num_directions, batch_size, hidden_dim]
        forward_hidden = hidden[0:hidden.size(0):2]  # Extract forward hidden states
        backward_hidden = hidden[1:hidden.size(0):2]  # Extract backward hidden states
        return torch.cat((forward_hidden, backward_hidden), dim=2)  # Concatenate along hidden_dim

    def training_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        output = self(src, tgt, src_lengths, tgt_lengths)
        loss = self.criterion(output, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        output = self(src, tgt, src_lengths, tgt_lengths)
        loss = self.criterion(output, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)


class TransformerSeq2Seq(BaseModel):
    def __init__(self, n_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                 learning_rate, batch_size, results_directory, checkpoints_directory=None):
        super(TransformerSeq2Seq, self).__init__(model_name="TransformerSeq2Seq",
                                                 checkpoints_directory=checkpoints_directory)
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.results_directory = results_directory

        # Transformer encoder and decoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Fully connected layer to map output of the transformer to feature space
        self.fc_out = nn.Linear(d_model, n_features)

        # Loss function for sequence to sequence
        self.criterion = nn.MSELoss()  # Assuming regression; change for classification tasks

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, src, tgt):
        # src and tgt shapes: [batch_size, seq_len, n_features]

        # Transform input and target to [seq_len, batch_size, d_model]
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Encode the source sequence
        memory = self.encoder(src)

        # Decode the target sequence
        output = self.decoder(tgt, memory)

        # Project to the output feature space
        output = self.fc_out(output)

        # Revert back to [batch_size, seq_len, n_features]
        return output.permute(1, 0, 2)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.criterion(output, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.criterion(output, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TransformerSeq2One(BaseModel):  # Encoder-only Transformer
    def __init__(self, n_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, learning_rate,
                 batch_size, results_directory, num_classes=1, checkpoints_directory=None):
        super(TransformerSeq2One, self).__init__(model_name="TransformerSeq2One",
                                                 checkpoints_directory=checkpoints_directory)
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.results_directory = results_directory
        self.num_classes = num_classes

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Fully connected layer for classification or regression
        self.fc_out = nn.Linear(d_model, num_classes)

        # Set loss function
        if num_classes == 1:
            self.criterion = nn.MSELoss()  # For regression
        else:
            self.criterion = nn.CrossEntropyLoss()  # For classification

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, src):
        # src shape: [batch_size, seq_len, n_features]

        # Transform input to [seq_len, batch_size, d_model]
        src = src.permute(1, 0, 2)

        # Encode the source sequence
        memory = self.encoder(src)

        # Take the last hidden state for sequence-to-one
        output = self.fc_out(memory[-1])  # Last time step

        return output

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src)
        loss = self.criterion(output, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src)
        loss = self.criterion(output, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TransformerEncoderDecoder(BaseModel):
    def __init__(self, n_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                 learning_rate, batch_size, results_directory, checkpoints_directory=None):
        super(TransformerEncoderDecoder, self).__init__(model_name="TransformerEncoderDecoder",
                                                        checkpoints_directory=checkpoints_directory)
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.results_directory = results_directory

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Fully connected layer to map output of the transformer to feature space
        self.fc_out = nn.Linear(d_model, n_features)

        # Loss function for encoder-decoder
        self.criterion = nn.MSELoss()  # Assuming regression; change for classification tasks

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, src, tgt):
        # src and tgt shapes: [batch_size, seq_len, n_features]

        # Transform input and target to [seq_len, batch_size, d_model]
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Encode the source sequence
        memory = self.encoder(src)

        # Decode the target sequence
        output = self.decoder(tgt, memory)

        # Project to the output feature space
        output = self.fc_out(output)

        # Revert back to [batch_size, seq_len, n_features]
        return output.permute(1, 0, 2)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.criterion(output, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.criterion(output, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
