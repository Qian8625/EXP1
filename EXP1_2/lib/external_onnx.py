"""VSE model (Optimized)"""
import os
import numpy as np
import torch
import torch.nn.init
import onnxruntime
from torch import nn

def get_text_external_encoder(embed_size, no_txtnorm=False):
    return Clip_TextModel(embed_size, no_txtnorm=no_txtnorm)

class Clip_TextModel(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super().__init__()
        self.onnx_path = os.path.join('onnx/clip-vit-base-patch32_text_encoder.onnx')
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

        self.providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=self.providers)

        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.MLP = nn.Linear(in_features=512, out_features=embed_size)
        self.max_seq_length = 77
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the MLP."""
        torch.nn.init.xavier_uniform_(self.MLP.weight)
        if self.MLP.bias is not None:
            torch.nn.init.constant_(self.MLP.bias, 0)

    def forward(self, input_ids, lengths):
        if isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.cpu().numpy()
        else:
            input_ids_np = np.array(input_ids)

        batch_size, max_len = input_ids_np.shape

        if max_len > self.max_seq_length:
            input_ids_np = input_ids_np[:, :self.max_seq_length]
            max_len = self.max_seq_length
            # Clamp lengths as well
            if isinstance(lengths, torch.Tensor):
                lengths = torch.clamp(lengths, max=self.max_seq_length)
            else:
                lengths = np.clip(lengths, 0, self.max_seq_length)

        if isinstance(lengths, torch.Tensor):
            lengths_np = lengths.cpu().numpy()
        else:
            lengths_np = np.array(lengths)

        # Create a range array [0, 1, 2, ..., max_len-1]
        aranged_len = np.arange(max_len)
        attention_mask = (aranged_len[None, :] < lengths_np[:, None]).astype(np.int64)

        # Prepare inputs for ONNX inference
        feed_dict = {
            'input_ids': input_ids_np.astype(np.int64),
            'attention_mask': attention_mask
        }

        outputs = self.session.run(None, feed_dict)
        last_hidden_state = outputs[0]

        # Normalize the embeddings
        if not self.no_txtnorm:
            norms = np.linalg.norm(last_hidden_state, axis=1, keepdims=True)
            # Use a small epsilon for numerical stability
            text_embeds_np = last_hidden_state / (norms + 1e-8)
        else:
            text_embeds_np = last_hidden_state

        # Convert back to a PyTorch tensor and move to the correct device
        device = self.MLP.weight.device
        text_embeds_tensor = torch.from_numpy(text_embeds_np).to(device)

        # Project to the target embedding space
        final_embeds = self.MLP(text_embeds_tensor)

        return final_embeds