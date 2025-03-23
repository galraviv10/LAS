import torch
from torch import nn
import pytorch_lightning as pl

class Listener(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output

class Speller(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        return self.fc(output), hidden

class Attender(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(2*hidden_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # This is a simplistic attention mechanism.
        attn_weights = torch.softmax(self.attention(encoder_outputs), dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        return context.squeeze(1)

class LASModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()
        self.listener = Listener(input_dim, hidden_dim)
        self.attender = Attender(hidden_dim)
        self.speller = Speller(vocab_size, hidden_dim)

    def forward(self, x, targets):
        encoder_outputs = self.listener(x)
        hidden = None
        loss = 0
        for t in range(targets.size(1)):
            if t == 0:
                input = torch.zeros((targets.size(0), 1), dtype=torch.long, device=self.device)
            else:
                input = targets[:, t-1].unsqueeze(1)
            context = self.attender(encoder_outputs, hidden)
            output, hidden = self.speller(input, hidden)
            loss += nn.CrossEntropyLoss()(output.squeeze(1), targets[:, t])
        return loss

    def training_step(self, batch, batch_idx):
        x, targets = batch
        x = x.transpose(1, 2)
        loss = self.forward(x, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        x = x.transpose(1, 2)
        loss = self.forward(x, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, targets = batch
        x = x.transpose(1, 2)
        loss = self.forward(x, targets)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
