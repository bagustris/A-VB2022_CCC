import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor
import torch
import torchaudio
import os

# wav_path = '/data/A-VB/audio/wav/'
# files = os.listdir(wav_path)

# save_dir = "/data/A-VB/features/w2v2-lr-300"

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir, exist_ok=True)

# load model
processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust")

# file = files[-1]
# array, fs = torchaudio.load(os.path.join(wav_path, file))
array, fs = torchaudio.load("/data/A-VB/audio/wav/00001.wav")
input = processor(array.squeeze(), sampling_rate=fs, return_tensors="pt")

with torch.no_grad():
    outputs = model(**input)

last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).numpy()
print(f"last_hidden_states.shape: {last_hidden_states.shape}")