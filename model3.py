import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
from transformers import BertModel, BertTokenizer
import jiwer

class Encoder(nn.Module):
    def __init__(self, input_dim, channels, num_lstm_layers, hidden_dim, projection_dim):
        super().__init__()
        self.num_directions = 2
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),  # First layer reducing each dimension by half
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),  # Second layer, further reducing time dimension by half
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.lstm_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        #update channels:
        channels = channels * 20
        # Dynamically add LSTM layers
        for _ in range(num_lstm_layers):
            lstm = nn.LSTM(channels, hidden_dim, dropout=0.3 ,batch_first=True, bidirectional=True)
            self.lstm_layers.append(lstm)
            # Projection and batch normalization layers
            projection = nn.Linear(hidden_dim * self.num_directions, projection_dim)  # *2 for bidirectional
            bn = nn.BatchNorm1d(projection_dim)
            self.projection_layers.append(projection)
            self.bn_layers.append(bn)
            channels = hidden_dim
        # self.lstm = nn.LSTM(channels, hidden_dim, num_layers = self.num_lstm_layers, batch_first=True, bidirectional=True)
        self.state_projection = nn.Linear(hidden_dim*2, projection_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        # Calculate the necessary dimensions for the states
        batch_size = x.size(0)
        reduced_time = x.size(3)
        x = x.permute(0, 3, 1, 2)  # Permute to [batch, reduced_time, channels, reduced_frequency]
        x = x.reshape(batch_size, reduced_time, -1)  # Flatten the channel and reduced_frequency dimensions

        final_hidden = []
        final_cell = []

        # Initialize hidden and cell states
        h = torch.zeros(self.num_directions , batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(self.num_directions , batch_size, self.hidden_dim, device=x.device)

        for lstm, projection, bn in zip(self.lstm_layers, self.projection_layers, self.bn_layers):
            x, (h, c) = lstm(x, (h, c))
            x = projection(x)
            x = bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x, (h, c) = self.lstm(x, (h, c))
        final_hidden.append(h)
        final_cell.append(c)

        # Concatenate the bidirectional states from the last LSTM layer
        final_layer_hidden = final_hidden[-1]
        final_layer_cell = final_cell[-1]         ## hidden_dim: ( direction=2, Batch_size=512, Hout=256)
        final_layer_hidden = torch.cat((final_layer_hidden[-2, :, :], final_layer_hidden[-1, :, :]), dim=1)
        final_layer_cell = torch.cat((final_layer_cell[-2, :, :], final_layer_cell[-1, :, :]), dim=1)
        final_layer_hidden = self.state_projection(final_layer_hidden).unsqueeze(0)
        final_layer_cell = self.state_projection(final_layer_cell).unsqueeze(0)
        # x = self.state_projection(x)
        ## hidden_dim: ## hidden_dim: ( direction=1, Batch_size=512, Hout=256)
        return x, final_layer_hidden, final_layer_cell


class Decoder(nn.Module):
    def __init__(self, input_dim,vocab_size, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim+hidden_dim, hidden_dim, dropout=0.3, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs):
       ## Making sure it matches expected attention dims: encoder_outputs=(512,17,256) and hidden_dim (direction=1,batch=512, feature=256)
        query = hidden.squeeze(0).unsqueeze(1) ##query_dim (512,1, 256)
        attn_output, attn_weights = self.attention(query, encoder_outputs, encoder_outputs) #shape: [batch,seq_length,hidden_dim]
       # Concatenate attention output and embedded input token
        lstm_input = torch.cat((input, attn_output), dim=2)  # shape: [batch_size, seq_legnth(=1), hidden_dim(=256) +token_dim (=768)]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.classifier(output)
        return output, hidden, cell


class LASModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_lstm_layers, projection_dim, channels, learning_rate, teacher_forcing):
        super().__init__()
        self.encoder = Encoder(input_dim, channels, num_lstm_layers, hidden_dim, projection_dim)
        self.decoder = Decoder(768, vocab_size, projection_dim) ##768 no teacher forcing
        self.teacher_forcing = teacher_forcing
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x, targets):
        encoder_outputs, hidden, cell = self.encoder(x)
        outputs = []
        start_token_idx = self.tokenizer.cls_token_id
        start_token = torch.full((x.size(0), 1), start_token_idx, device=x.device)
        for t in range(targets.size(1)):  ## targets= [batch=512, seq_legnth=512]
            if t == 0:
                input = start_token  # Use the start token as input (batch=512,1,1)
            else:
                if self.teacher_forcing:
                    input = targets[:, t-1].unsqueeze(1)  # Use the target as input (batch=512,1,1)
                else:
                    input = torch.argmax(outputs[-1], dim=-1)  # Use the last predicted output as input (batch=512,1,1)
            input = self.bert.embeddings(input) # bert embedding (batch=512,1,1)
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs.append(output)  ##output_dim=(batch=512,seq_len=1,vocab=30522)
        return torch.cat(outputs, dim=1)

    def training_step(self, batch, batch_idx): ##teacher_forcing
        x, targets = batch
        logits = self(x, targets) ##forward pass logits_dim=(batch=512,seq_len=512,vocab=30522)
        logits = logits.view(-1,logits.size(-1)) ## logits new dim = (batch=512*seq_len=512,vocab=30522)
        loss = F.cross_entropy(logits, targets.view(-1)) ##targets=(batch=512,seq_len=512)->targets.view(-1)=(batch=512*seq_len=512)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        logits = self(x, targets)  ##forward pass logits_dim=(batch=512,seq_len=512,vocab=30522)
        logits_loss = logits.view(-1, logits.size(-1))  ## logits new dim = (batch=512*seq_len=512,vocab=30522)
        loss = F.cross_entropy(logits_loss, targets.view(
            -1))  ##targets=(batch=512,seq_len=512)->targets.view(-1)=(batch=512*seq_len=512)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)

        ##Sanity checks:
        # Decode all tokens in the batch
        # Reshape logits for decoding: [batch, seq_len, vocab_size]
        tokens = logits.argmax(dim=-1)  # Get token indices: [batch, seq_len]
        decoded_texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in tokens]
        ground_truth_texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in targets]

        # Calculate WER for each pair of prediction and ground truth
        wer_scores = [jiwer.wer(gt, pred) for gt, pred in zip(ground_truth_texts, decoded_texts)]
        average_wer = sum(wer_scores) / len(wer_scores)

        # Log the average WER
        self.log('avg_wer', average_wer, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #Optionally print first decoded example in the batch for verification
        # if batch_idx == 0:  # Print only for the first batch
        #     print(f"  Ground Truth: {ground_truth_texts[0]}")
        #     print(f"  Predicted   : {decoded_texts[0]}")

        return loss

    def test_step(self, batch, batch_idx):
        x, targets = batch
        logits = self(x, targets) ##forward pass logits_dim=(batch=1,seq_len=512,vocab=30522)
        logits = logits.view(-1,logits.size(-1)) ## logits new dim = (batch=1*seq_len=512,vocab=30522)
        loss = F.cross_entropy(logits, targets.view(-1)) ##targets=(batch=1,seq_len=512)->targets.view(-1)=(batch=1*seq_len=512)
        self.log('test_loss', loss, sync_dist=True)

        # Decode all tokens in the batch
        logits = logits.view(-1, logits.size(-2), logits.size(-1))
        tokens = logits.argmax(dim=-1)  # Get token indices: [batch, seq_len]
        decoded_texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in tokens]
        ground_truth_texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in targets]

        # Calculate WER for each pair of prediction and ground truth
        wer_scores = [jiwer.wer(gt, pred) for gt, pred in zip(ground_truth_texts, decoded_texts)]
        average_wer = sum(wer_scores) / len(wer_scores)

        # Log the average WER
        self.log('avg_wer', average_wer, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Optionally print first decoded example in the batch for verification
        if batch_idx == 0 or 5 or 10 or 15:  # Print only for the first batch
            print(f"  Ground Truth: {ground_truth_texts[0]}")
            print(f"  Predicted   : {decoded_texts[0]}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
        #                                                             verbose=True),
        #     'monitor': 'val_loss'  # Name of the metric to monitor
        # }
        return [optimizer] #[scheduler]
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


