import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, input_dim, channels, num_lstm_layers, hidden_dim, projection_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        self.lstm_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Dynamically add LSTM layers
        for _ in range(num_lstm_layers):
            lstm = nn.LSTM(channels, hidden_dim, batch_first=True, bidirectional=True)
            self.lstm_layers.append(lstm)
            # Projection and batch normalization layers
            projection = nn.Linear(hidden_dim * 2, projection_dim)  # *2 for bidirectional
            bn = nn.BatchNorm1d(projection_dim)
            self.projection_layers.append(projection)
            self.bn_layers.append(bn)
            channels = projection_dim  # Update the input size for the next layer

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # Adjust for LSTM input format

        final_hidden = []
        final_cell = []

        for lstm, projection, bn in zip(self.lstm_layers, self.projection_layers, self.bn_layers):
            x, (h, c) = lstm(x)
            x = projection(x)
            x = bn(x.permute(0, 2, 1)).permute(0, 2, 1)
            final_hidden.append(h)
            final_cell.append(c)
        # Depending on the use case, you might only need the last layer's states
        # Concatenate or handle differently depending on the network design requirements
        return x, final_hidden[-1], final_cell[-1]


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.query_dim_transform = nn.Linear(512, 256)

    def forward(self, input, hidden, cell, encoder_outputs):
        # hidden = self.query_dim_transform(hidden)
        # cell = self.query_dim_transform(cell)
        query = hidden.unsqueeze(0)  # Making sure it matches expected attention dims
        query = hidden.permute(1,0,2)
        #encoder_outputs = encoder_outputs.permute(1,0,2)
        attn_output, attn_weights = self.attention(query, encoder_outputs, encoder_outputs)
        lstm_input = attn_output.squeeze(0)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.classifier(output)
        return output, hidden, cell


class LASModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_lstm_layers, projection_dim, channels, learning_rate):
        super().__init__()
        self.encoder = Encoder(input_dim, channels, num_lstm_layers, hidden_dim, projection_dim)
        self.decoder = Decoder(vocab_size, projection_dim)
        self.learning_rate = learning_rate

    def forward(self, x, targets):
        encoder_outputs, hidden_n, cell_n = self.encoder(x)
        outputs = []
        hidden_n = torch.cat((hidden_n[0], hidden_n[1]), dim=1)  # Concatenate along hidden_size
        cell_n = torch.cat((cell_n[0], cell_n[1]), dim=1)
        hidden, cell = hidden_n, cell_n  # These should be properly formatted as per Decoder's expectation
        input = torch.zeros((targets.size(0), 1), dtype=torch.long, device=self.device)  # SOS tokens
        for t in range(targets.size(1)):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs.append(output)
            input = targets[:, t].unsqueeze(1)  # Next input is the actual target token (teacher forcing)
        return torch.cat(outputs, dim=1)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        loss = F.cross_entropy(self(x, targets), targets.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        loss = F.cross_entropy(self(x, targets), targets.view(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, targets = batch
        loss = F.cross_entropy(self(x, targets), targets.view(-1))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


