# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import random

y, sr = librosa.load('1727.wav', duration=5, sr=16000)
sf.write('1727_0.wav', y, sr)
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=64, fmax=8000)
plt.subplot(121)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
# plt.show()

y_hat = librosa.feature.inverse.mel_to_audio(S, sr)
sf.write('1727_1.wav', y_hat, sr)

import torch
import torch.nn as nn
import torch.optim as optim

device = 'cpu'

class Encoder(nn.Module):
    def __init__(self, input_dim=64, hid_dim=32, n_layers=4, dropout=0.2):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        
    def forward(self, x):
    	# x = [seq_len, batch_size, input_size]
        outputs, (hidden, cell) = self.rnn(x)
        
        #outputs = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim=64, hid_dim=32, n_layers=4, dropout=0.2):
        super(Decoder, self).__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
                
        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout = dropout)
        self.out = nn.Linear(hid_dim, output_dim)
        
    def forward(self, input, hidden, cell):        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
                 
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device='cpu'):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else output
        
        return outputs

enc = Encoder()
dec = Decoder()
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())#, lr=0.03, momentum=0.8)
criterion = nn.MSELoss()

x = torch.tensor(np.expand_dims(S.T, 1), dtype=torch.float)
mu = x.mean(axis=0, keepdims=True)
sig = x.std(axis=0, keepdims=True)
x = (x-mu)/sig

x = x.to(device)

model.train()
        
for i in range(1000):
    optimizer.zero_grad()
    output = model(x, x)   
    loss = criterion(output, x)        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    optimizer.step()
    if i%10 == 9:
        # print(torch.abs(x[1]-output[1]))
        print(loss.item())

model.eval()
with torch.no_grad():
    output = model(x, x, 0) #turn off teacher forcing
    loss = criterion(x, x)

plt.subplot(122)
output = output.detach().cpu()*sig + mu
S_dB = librosa.power_to_db(output[:,0,:].T, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()

y_hat = librosa.feature.inverse.mel_to_audio(output[:,0,:].T, sr)
sf.write('1727_2.wav', y_hat, sr)
