import os
import torch
import torchaudio
import librosa
import numpy as np
#from scipy.special import additional
from torch.utils.data import Dataset, DataLoader
from random import randrange, random
from scaler_gan.scalergan_utils.scalergan_utils import norm_audio_like_hifi_gan, sample_segment, mel_spectrogram, crop_mel,load_audio_to_np
from transformers import BertTokenizer
import random
from torch.nn.functional import pad
import glob


class AudioTextDataset(Dataset):
    def __init__(self, audio_paths_file, config, train, augment=True, transform=None, save_mel=False):
        self.config = config
        self.augment = augment
        self.train = train
        self.sample_aug = random.random() < 0.2 and train
        self.transform = transform
        self.augmentation_method = "specaugment" #self.scalergan
        # Load a pre-trained BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #for testing puposes
        self.save_mel = save_mel

        # Read audio paths from the specified text file
        with open(audio_paths_file, 'r') as file:
            self.audio_paths = [line.strip() for line in file if line.strip()]

        if not self.audio_paths:
            raise FileNotFoundError("No audio paths found in the specified file.")

        print("Loaded audio paths:", len(self.audio_paths))

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        text_path = audio_path.replace('.wav', '.txt')
        mel = None

        if self.augment and self.sample_aug:
            try:
                mel = self.specaugment(audio_path, self.config['aug_output_dir'],self.config['additional_output_dir'])
                if isinstance(mel, np.ndarray):
                    mel = torch.tensor(mel, dtype=torch.float32)
            except Exception as e:
                print(f"DEBUG: Exception in try block: {e}")
                print("No Augmentation found")
        else:
            # Load audio file
            waveform, sr = load_audio_to_np(audio_path)
            waveform = norm_audio_like_hifi_gan(waveform)
            waveform = torch.FloatTensor(waveform)
            waveform = sample_segment(waveform.unsqueeze(0), self.config['segment_size'])

            mel = mel_spectrogram(
                waveform,
                n_fft=self.config['n_fft'],
                num_mels=self.config['num_mels'],
                sampling_rate=sr,
                hop_size=self.config['hop_size'],
                win_size=self.config['win_size'],
                fmin=self.config['fmin'],
                fmax=self.config['fmax']
            )

        mel = crop_mel(mel, self.config['must_divide'])
        mel = mel.squeeze(0)


        mel = self.pad_mel(mel,target_length=self.config['target_length'])
        # print(mel.shape)
        # if self.save_mel:
        #     save_dir = self.config.get('mel_save_dir', 'mel_spectrograms_scalerGan')  # Output directory
        #     os.makedirs(save_dir, exist_ok=True)
        #     mel_filename = os.path.join(save_dir, os.path.basename(audio_path).replace('.wav', '.npy'))
        #     np.save(mel_filename, mel)  # Convert to NumPy and save


        # Load corresponding text file
        with open(text_path, 'r') as file:
            text = file.read().strip()
        # tokenize
        token_ids_tensor = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.config['max_len_tokens'],
            return_tensors="pt",
            truncation=True
        )['input_ids'].squeeze(0)
        return mel, token_ids_tensor

    def scalergan(self, waveform):
        # Placeholder for augmentation logic
        pass

    def specaugment(self, file_path,output_directory,additional_output_directory):
        filename = os.path.basename(file_path)  # Get "file.wav"
        filename_no_ext = os.path.splitext(filename)[0]  # Get "file"
        augmented_filepath = os.path.join(output_directory, filename_no_ext + "_cropped_*.*.npy")
        additional_augmented_filepath = os.path.join(additional_output_directory, filename_no_ext + "_cropped_*.*.npy")
        #augmented_filename = filename_no_ext + "_augmented.wav"  # "file_augment.npy"
        # augmented_filepath = os.path.join(output_directory, augmented_filename)
        matching_file= glob.glob(augmented_filepath )
        if matching_file == []:
            matching_file = glob.glob(additional_augmented_filepath)
        if matching_file:
            return np.load(matching_file[0])
        # if os.path.exists(augmented_filepath):
        #     return np.load(augmented_filepath)
        else:
            print(f"File not found: {augmented_filepath}")
            return None

    def pad_mel(self, mel, target_length):
        """Pad or truncate mel spectrogram to a fixed length."""
        current_length = mel.shape[-1]  # Last dimension is time frames
        if current_length < target_length:
            pad_amount = target_length - current_length
            mel = pad(mel, (0, pad_amount), "constant", 0)  # Pad on the right
        else:
            mel = mel[:, :target_length]  # Truncate

        return mel
