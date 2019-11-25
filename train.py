import torch
from model import Encoder, Decoder, Seq2Seq
import torch.nn as nn
import torch.optim as optim
from loader import InstrumentDataset
from torch.utils.data import Dataset, DataLoader
import os

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = 'cuda:0'

enc = Encoder()
dec1 = Decoder() # Piano
dec2 = Decoder() # Tabla
model1 = Seq2Seq(enc, dec1, device).to(device)
model2 = Seq2Seq(enc, dec2, device).to(device)

model1.apply(init_weights)
model2.apply(init_weights)

print(f'The model has {count_parameters(model1):,} trainable parameters')

optimizer1 = optim.Adam(model1.parameters())#, lr=0.03, momentum=0.8)
optimizer2 = optim.Adam(model2.parameters())
criterion = nn.MSELoss()

trainsets = [InstrumentDataset('Preprocessed/Piano/train/', True), InstrumentDataset('Preprocessed/Tabla/train/', True)]
trainloaders = [DataLoader(x, batch_size=75, shuffle=True, num_workers=8) for x in trainsets]

model1.train()
model2.train()
	
print('Training Started')
for epoch in range(1):
	print('='*20, 'Training Piano', '='*20)
	for i,x in enumerate(trainloaders[0]):
		x = x.transpose(0,1)
		x = x.to(device)
		optimizer1.zero_grad()
		output = model1(x, x)   
		loss = criterion(output, x)        
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model1.parameters(), 100)
		optimizer1.step()
		print(loss.item())

	print('='*20, 'Training Tabla', '='*20)
	for i,x in enumerate(trainloaders[1]):
		x = x.transpose(0,1)
		x = x.to(device)
		optimizer2.zero_grad()
		output = model2(x, x)   
		loss = criterion(output, x)        
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model2.parameters(), 100)
		optimizer2.step()
		print(loss.item())


testsets = [InstrumentDataset('Preprocessed/Piano/test/'), InstrumentDataset('Preprocessed/Tabla/test/')]
testloaders = [DataLoader(x, batch_size=75, shuffle=False, num_workers=8) for x in testsets]

model1.eval()
model2.eval()

import librosa
import numpy as np
import soundfile as sf
os.makedirs('results/piano', exist_ok=True)
with torch.no_grad():
	for i,x in enumerate(testloaders[0]):
		mu = x.mean(axis=1, keepdims=True)
		sig = x.std(axis=1, keepdims=True)
		x = (x-mu)/sig
		x = x.transpose(0,1)
		x = x.to(device)
		output = model2(x, x, 0).transpose(0,1)
		x = x.transpose(0,1).detach().cpu().numpy()
		# x and output are batch_size x seq_len x filterbanks
		print(output.shape, sig.shape, mu.shape, x.shape)
		sig = sig.detach().cpu().numpy()
		mu = mu.detach().cpu().numpy()
		output = output.detach().cpu().numpy()*sig + mu
		for j in range(output.shape[0]):
			S_dB = librosa.power_to_db(output[j,:,:].T, ref=np.max)
			y_hat = librosa.feature.inverse.mel_to_audio(output[j,:,:].T, sr=16000)
			sf.write('results/piano/piano2tabla_'+str(i*output.shape[1]+j)+'.wav', y_hat, 16000)
			print('Saved: results/piano/piano2tabla_'+str(i*output.shape[1]+j)+'.wav')
			S_dB = librosa.power_to_db(x[j,:,:].T, ref=np.max)
			y_hat = librosa.feature.inverse.mel_to_audio(x[j,:,:].T, sr=16000)
			sf.write('results/piano/piano_'+str(i*output.shape[1]+j)+'.wav', y_hat, 16000)
			print('Saved: results/piano/piano_'+str(i*output.shape[1]+j)+'.wav')
