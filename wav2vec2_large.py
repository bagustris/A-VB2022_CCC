
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch

# load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# audio file is decoded on the fly
array, fs = torchaudio.load("/data/A-VB/audio/wav/00001.wav")
input = processor(array.squeeze(), sampling_rate=fs, return_tensors="pt")

# apply the model to the input array from wav
with torch.no_grad():
    outputs = model(**input)

# extract last hidden state, compute average, convert to numpy
last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).numpy()

# change to list to print
print(f"Hidden state shape: {last_hidden_states.shape}")
# Hidden state shape: (768,)
