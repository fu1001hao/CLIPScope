import os
from imagenetv2_pytorch import ImageNetV2Dataset
from scipy.special import softmax
import nltk
from nltk.corpus import wordnet as wn
import clip
import torch
from torchvision.datasets import Places365, ImageNet, INaturalist
import inflect
import torchvision.datasets as dset
import spacy
import numpy as np
import json
from sklearn import metrics
from random import shuffle

def get_features(text, model, normalize = True):
    with torch.no_grad():
        text_features = model.encode_text(text).float()
        if normalize:
            text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

nouns = torch.load('sorted_nouns.pt')

m = torch.nn.Softmax()
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)
with open('imagenet_class_clean.npy', 'rb') as f:
    imagenet_cls = np.load(f)

def run(IDdata, OODdata,  IDclass, OODclass, model, preprocess,  L = 200, height = 256, normalize = True):

    with torch.no_grad():
        text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in IDclass]).to(device)
        ID_features = get_features(text, model, normalize = normalize)
        text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in OODclass[:len(OODclass)//2]]).to(device)
        OOD_features1 = get_features(text, model, normalize = normalize)
        text = torch.cat([clip.tokenize(f"a photo of {c}") for c in OODclass[len(OODclass)//2:]]).to(device)
        OOD_features2 = get_features(text, model, normalize = normalize)
        text_features = torch.cat([ID_features, OOD_features1, OOD_features2], 0)

    scores = []
    L1 = min(L, len(IDdata))
    xID, yID = np.arange(L1), np.zeros(L1)
    L = min(L, len(OODdata))
    xOOD, yOOD = np.arange(L), np.ones(L)
    x = np.concatenate((xID, xOOD),0)
    y = np.concatenate((yID, yOOD), 0)
    indices = np.random.permutation(len(y))
    count = np.ones(len(IDclass))
    countin = np.ones(len(IDclass))
    zero = 1e-30
    for i in indices:
        j, idx = x[i], y[i]
        if idx==0:
            j = int(j)
            if isinstance(IDdata[j], dict):
                image, _ = IDdata[j].values()
            else:
                image, _ = IDdata[j]
        else:
            image, _ = OODdata[j]
        with torch.no_grad():
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input).float()
            if normalize:
                image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T)
            score = count/count.sum()
            probs = (100*similarity[:, :len(IDclass)]).softmax(dim=-1).float()[0] 
            pred = probs.argmax()
            p = (100*similarity).softmax(dim=-1).float()[0, :len(IDclass)].sum()
            confidence = probs[pred]*p/score[pred]
            count[pred] += 1

            print(confidence, p)
            tmp = confidence
            scores.append([tmp.cpu(), idx, pred.cpu()])

    scores = np.array(scores)

    return scores

L = 10000                    # change the number as you need
imagenet = dset.ImageFolder(root="../data/imagenet/tests")
texture = dset.ImageFolder(root="../data/dtd/images")
inature = dset.ImageFolder(root="../data/iNaturalist")
places = dset.ImageFolder(root="../data/Places")
sun = dset.ImageFolder(root="../data/SUN")

imagenet_labels = imagenet_cls.tolist()

k = 5000             # change the number as you need
Totscores = run(imagenet, inature, imagenet_labels,nouns[:k]+nouns[-k:],  model, preprocess,  L=L)
scores, labels = Totscores[:,0], Totscores[:,1]
fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
AUROC = metrics.auc(fpr, tpr)
print("inature")
print('AUROC', AUROC)
idx = np.argmax(tpr>0.95)
print('FPR', fpr[idx], 'TPR', tpr[idx])

Totscores = run(imagenet, sun, imagenet_labels,nouns[:k]+ nouns[-k:], model, preprocess,  L=L)
scores, labels = Totscores[:,0], Totscores[:,1]
fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
AUROC = metrics.auc(fpr, tpr)
print("sun")
print('AUROC', AUROC)
idx = np.argmax(tpr>0.95)
print('FPR', fpr[idx], 'TPR', tpr[idx])

Totscores = run(imagenet, places, imagenet_labels, nouns[:k]+ nouns[-k:],  model, preprocess,  L=L)
scores, labels = Totscores[:,0], Totscores[:,1]
fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
AUROC = metrics.auc(fpr, tpr)
print("places")
print('AUROC', AUROC)
idx = np.argmax(tpr>0.95)
print('FPR', fpr[idx], 'TPR', tpr[idx])

Totscores = run(imagenet, texture, imagenet_labels, nouns[:k]+ nouns[-k:], model, preprocess,  L=L)
scores, labels = Totscores[:,0], Totscores[:,1]
fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
AUROC = metrics.auc(fpr, tpr)
print("texture")
print('AUROC', AUROC)
idx = np.argmax(tpr>0.95)
print('FPR', fpr[idx], 'TPR', tpr[idx])
