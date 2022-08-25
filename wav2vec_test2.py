from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector
from datasets import load_dataset
import torch

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")

# audio file is decoded on the fly
inputs = feature_extractor(
    [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
)
with torch.no_grad():
    embeddings = model(**inputs).embeddings

embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

# the resulting embeddings can be used for cosine similarity-based retrieval
cosine_sim = torch.nn.CosineSimilarity(dim=-1)
similarity = cosine_sim(embeddings[0], embeddings[1])
threshold = 0.7  # the optimal threshold is dataset-dependent
if similarity < threshold:
    print("Speakers are not the same!")

print(f"Similarity between speakers is {round(similarity.item(), 2)}")