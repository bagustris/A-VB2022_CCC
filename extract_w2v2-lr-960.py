import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
import torch
import torchaudio
import os

wav_path = '/data/A-VB/audio/wav/'
files = os.listdir(wav_path)

save_dir = "/data/A-VB/features/w2v2-lr-960"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")

# head = [str(i) for i in range(768)]

# audio file is decoded on the fly
for file in files:
    print(f"Processing {file}")

    array, fs = torchaudio.load(os.path.join(wav_path, file))
    input = processor(array.squeeze(), sampling_rate=fs, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**input)

    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).numpy()
    np.savetxt(f"{save_dir}/{str(file[:-3])}csv", last_hidden_states.reshape(1, -1), delimiter=',')
