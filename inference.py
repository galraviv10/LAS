import torch
import yaml
from scipy.stats.tests.test_mstats_extras import test_rsh
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
from model3 import LASModel
from dataloader import AudioTextDataset


class LASInference:
    def __init__(self, checkpoint_path, config_path):
        """
        Initializes the model, tokenizer, and dataset.
        :param checkpoint_path: Path to trained checkpoint (.ckpt)
        :param config_path: Path to config.yml
        """
        # Load config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Load model
        self.model = LASModel(
            input_dim=self.config['training']["input_dim"],
            hidden_dim=self.config['training']["hidden_dim"],
            vocab_size=self.config['training']["vocab_size"],
            num_lstm_layers=self.config['training']["num_lstm_layers"],
            projection_dim=self.config['training']["projection_dim"],
            channels=self.config['training']["channels"],
            learning_rate=self.config['training']["learning_rate"],
            teacher_forcing=self.config['training']["teacher_forcing"]
        )

        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint['state_dict'],strict=True)
        self.model.eval()  # Set model to evaluation mode

        # Load tokenizer (BERT-based)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Load test dataset from config
        self.test_loader = self.get_test_loader()

    def get_test_loader(self):
        """
        Creates a DataLoader for the test dataset.
        """
        test_dataset = AudioTextDataset(
            audio_paths_file=self.config['test']['data_path'],
            config=self.config['audio_processing'],
            train=False
        )
        return DataLoader(
            test_dataset,
            batch_size=self.config['test']['batch_size'],  # Match training batch size
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )
    def infer(self, audio_tensor, ground_truth_tokens=None):
        """
        Runs inference on an audio sample.
        :param audio_tensor: Processed audio input tensor (batch=1, feature_dim, time_steps)
        :param ground_truth_tokens: Optional, tokenized reference text (tensor)
        :return: Decoded text, loss (if ground truth is provided)
        """
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.model.encoder(audio_tensor)

            # Start decoding with [CLS] token
            start_token = torch.tensor(
                [[self.tokenizer.cls_token_id]],
                device=encoder_outputs.device,
                dtype=torch.long
            ).unsqueeze(2)  # Shape: [1,1,1]

            predicted_logits = []
            predicted_tokens = []

            for _ in range(self.config['audio_processing']['max_len_tokens']):  # Limit length
                output, hidden, cell = self.model.decoder(start_token, hidden, cell, encoder_outputs)

                predicted_logits.append(output)  # Collect all logits
                token = output.argmax(dim=-1).item()  # Get token with highest probability
                print(f"Generated token: {token}, Token text: {self.tokenizer.decode([token])}")

                # if token == self.tokenizer.sep_token_id:  # Stop if [SEP] token is generated
                #     break

                predicted_tokens.append(token)  # Store token prediction
                start_token = torch.tensor([[[token]]], device=encoder_outputs.device,
                                           dtype=torch.long)  # Maintain shape [1,1,1]

            loss = None
            if ground_truth_tokens is not None:
                # Convert collected logits into tensor
                predicted_logits = torch.cat(predicted_logits, dim=1)   # Shape [1, seq_len, vocab_size]
                loss = F.cross_entropy(
                    predicted_logits.view(-1, predicted_logits.size(-1)),  # Ensure shape [batch * seq_len, vocab_size]
                    ground_truth_tokens.view(-1)  # Ensure shape [batch * seq_len]
                ).item()
                #Decode the predicted tokens
                if predicted_tokens:
                    decoded_text = self.tokenizer.decode(predicted_tokens, skip_special_tokens=True)
                else:
                    decoded_text = "[NO OUTPUT]"

                decoded_text = self.tokenizer.decode(predicted_tokens, skip_special_tokens=True)
                ground_truth_text = self.tokenizer.decode(ground_truth_tokens.squeeze(0).tolist(), skip_special_tokens=True)

        return decoded_text, ground_truth_text, loss

    def run_inference(self):
        """
        Runs inference on the entire test dataset.
        """
        for idx, (audio, ground_truth) in enumerate(self.test_loader):
            decoded_text, ground_truth_text, loss = self.infer(audio, ground_truth)
            print(f"Sample {idx + 1}:")
            print(f"  Ground Truth: {ground_truth_text}")
            print(f"  Predicted   : {decoded_text}")
            if loss is not None:
                print(f"  Loss        : {loss:.4f}")
            print("-" * 50)


# Example Usage
if __name__ == "__main__":
    config_path = "config.yml"

    # Load checkpoint path from config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model_path = config['inference']['checkpoint']  # Change to actual checkpoint path

    # Initialize inference
    inferencer = LASInference(model_path, config_path)

    # Run inference on the test dataset
    inferencer.run_inference()
