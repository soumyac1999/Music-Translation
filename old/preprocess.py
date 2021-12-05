import numpy as np
import librosa
import soundfile as sf
import shutil, os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--src', help="Src root", required=True)
parser.add_argument('--dst', help="Dest root", required=True)

if __name__ == '__main__':
	args = parser.parse_args()
	src_root = args.src
	dest_root = args.dst
	for instrument_name in os.listdir(args.src): # instrument wise
		print(instrument_name)
		for data_split in os.listdir(os.path.join(src_root, instrument_name)): # split dir
			for filename in os.listdir(os.path.join(src_root, instrument_name, data_split)):
				# do  something about it -> process
				# now load and save npy files
				srcpath = os.path.join(src_root, instrument_name, data_split, filename)
				y, sr = librosa.load(srcpath, sr=16000)
				D = np.abs(librosa.stft(y))**2
				S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=64, fmax=8000)
				dstpath = os.path.join(dest_root, instrument_name, data_split, filename.split('.')[0])
				os.makedirs(os.path.join(dest_root, instrument_name, data_split), exist_ok=True)
				np.save(dstpath, S)
