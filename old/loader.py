import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

class InstrumentDataset(object):
	def __init__(self, datadir, normalize=False):
		super(InstrumentDataset, self).__init__()
		self.datadir = datadir
		self.files = os.listdir(datadir)
		self.len = len(self.files)
		self.normalize = normalize
		self.ylen = 157

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		S = np.load(os.path.join(self.datadir, self.files[idx]))
		# y, sr = librosa.load(os.path.join(self.datadir, self.files[idx]), duration=5, sr=16000)
		# D = np.abs(librosa.stft(y))**2
		# S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=64, fmax=8000)
		x = torch.tensor(S.T, dtype=torch.float)
		if self.normalize:
			mu = x.mean(axis=0, keepdims=True)
			sig = x.std(axis=0, keepdims=True)
			x = (x-mu)/sig
		return x
