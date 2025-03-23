import torch
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    mel_specs, token_ids = zip(*batch)
    mel_specs = torch.stack(mel_specs)  # Assuming all mel spectrograms have the same shape
    token_ids_padded = pad_sequence(token_ids, batch_first=True, max_length=256)
    return mel_specs, token_ids_padded