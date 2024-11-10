import nltk
from nltk.corpus import wordnet as wn
import clip
import torch
import numpy as np

nouns = torch.load('nouns.pt')
nouns = np.array(nouns)
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)
with open('imagenet_class_clean.npy', 'rb') as f:
    imagenet_cls = np.load(f)

text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_cls]).to(device)
percentile = 50
with torch.no_grad():
    text_features = model.encode_text(text).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    scores = []
    for word in nouns:
        text = torch.cat([clip.tokenize(f"the "+word)]).to(device)
        sample_features = model.encode_text(text).float()
        sample_features /= sample_features.norm(dim=-1, keepdim=True)
        similarity = -(sample_features @ text_features.T)[0]
        value, index = torch.kthvalue(similarity, percentile)
        scores.append(-value.cpu().numpy())

my_list = scores
sorted_indices = sorted(range(len(my_list)), key=lambda x: my_list[x])
torch.save(sorted_indices, 'sorted_nouns.pt')
