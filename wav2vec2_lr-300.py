
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio

# load model
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-large-robust-ft-swbd-300h")
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-robust-ft-swbd-300h")

# audio file is decoded on the fly
array, fs = torchaudio.load("/data/A-VB/audio/wav/00001.wav")
input = processor(array.squeeze(), sampling_rate=fs, return_tensors="pt")

with torch.no_grad():
    outputs = model(**input)

last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).numpy()
# change to list to print
print(f"Hidden state shape: {last_hidden_states.shape}")
