# ============================================================
# models.py
# ============================================================
# Assignment: Implement neural architectures for
# Voice Activity Detection (VAD) using PyTorch.
#
# Tasks:
#   1. Implement a Xavier initializer
#   2. Implement the LSTM-based VAD model
#   3. Implement the BiLSTM-based VAD model
#   4. Implement the CNN + LSTM hybrid VAD model
#   5. Complete the model builder function
# ============================================================

import torch
import torch.nn as nn


NUM_CLASSES = 2  # 0 = non-speech, 1 = speech


# ------------------------------------------------------------
# 1. Xavier Uniform Initialization Utility
# ------------------------------------------------------------
def init_weights_xavier(m):
    """
    Apply Xavier uniform initialization to all suitable layers.
    """
    # TODO: implement the Xavier initializer

    #apply xavier normalization to m, which a layer in the network.
    #xavier uniform can be used on linear and convolutional layers.
    #lstm layers have multiple weight matrices, so iterate through them.
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters(): #from docs, returns dictionary of parameter names and tensors
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data) #param.data is the tensor inside the object param
            elif "bias" in name:
                nn.init.zeros_(param.data)


# ------------------------------------------------------------
# 2. LSTM-based VAD model
# ------------------------------------------------------------
class LSTMVad(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # TODO: define the LSTM-based architecture
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        # Batch-frist = true because the input is in the form [B,T,F]
        self.fc = nn.Linear(hidden_size, NUM_CLASSES) #fully connected linear activation that produces 2 classes

    def forward(self, x):
        """
        x: [B, T, F]
        returns: logits [B, T, 2]
        """
        # TODO: implement the forward pass
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        #logits = raw unnormalized scores, crossentropyloss function performs softmax + loss computations for you
        return logits


# ------------------------------------------------------------
# 3. Bi-directional LSTM-based VAD model
# ------------------------------------------------------------
class BiLSTMVad(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # TODO: define the BiLSTM-based architecture
        # 2 layer lstm with 128 unit
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True) # x = (batch, seq, feature)
        self.classifier = nn.Linear(hidden_size * 2, NUM_CLASSES) #bidirectional doubles the hidden size

    def forward(self, x):
        """
        x: [B, T, F]
        returns: logits [B, T, 2]
        """
        # TODO: implement the forward pass
        lstm_out = self.lstm(x)[0] #self.lstm provides [output, (h_n=final hidden state, c_n final cell state)]
        logits = self.classifier(lstm_out)
        #logits = raw unnormalized scores, crossentropyloss function performs softmax + loss computations for you
        return logits



# ------------------------------------------------------------
# 4. CNN + (Bi)LSTM hybrid VAD model
# ------------------------------------------------------------
class CNNLSTMVad(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        conv1_out: int = 16,
        conv2_out: int = 32,
        pool_kernel: int = 2,
    ):
        super().__init__()
        # TODO: define CNN feature extractor and related parameters
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # • CNN layers for feature extraction:
        #– see https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        #– Conv2D layer with 16 filters, kernel size 3, stride 1, padding 1.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_out, kernel_size=3, stride=1, padding=1)
        #– ReLU activation and MaxPooling.
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(pool_kernel, 1))
        #– Second Conv2D layer with 32 filters, kernel size 3.
        self.conv2 = nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=3, stride=1, padding=1)

        features_per_step = conv2_out * n_features
        self._build_recurrent(features_per_step)

    def _build_recurrent(self, features_per_step: int):
        """
        Build LSTM and FC layers once CNN feature dimensions are known.
        """
        # TODO: implement recurrent layer creation

        # LSTM takes the CNN features as input_size, output size depends on if lstm is bidirectional
        self.lstm = nn.LSTM(input_size=features_per_step, hidden_size=self.hidden_size, num_layers=self.lstm_layers, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        lstm_output_size = self.hidden_size
        if self.bidirectional:
            lstm_output_size *= 2

        # Final classification layer (per time step)
        self.fc = nn.Linear(lstm_output_size, NUM_CLASSES)

    def forward(self, x):
        """
        x: [B, T, F]
        returns: logits [B, T', 2]
        """
        # TODO: implement the forward pass
        
        #CNN expects a 4D input [B, C, H, W], so add channel dimension
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        #The output is [B, C, T', F]; reshape it to [B, T', C*F] for LSTM input
        B, C, T_prime, F = x.shape
        x = x.permute(0, 2, 1, 3) # rearrance to [B,T',C,F]
        x = x.reshape(B, T_prime, C * F) # [B, T', features]

        # --- LSTM ---
        lstm_out = self.lstm(x)[0]
        logits = self.fc(lstm_out)
        return logits



# ------------------------------------------------------------
# Model builder
# ------------------------------------------------------------
def build_model(model_type, n_features, hidden_size=128, num_layers=2, **kwargs):
    model_type = model_type.lower()

    if model_type == "lstm":
        return LSTMVad(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )

    elif model_type == "bilstm":
        return BiLSTMVad(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )

    elif model_type == "cnnlstm":
        return CNNLSTMVad(
            n_features=n_features,
            hidden_size=hidden_size,
            lstm_layers=num_layers,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from ['lstm', 'bilstm', 'cnnlstm']."
        )
