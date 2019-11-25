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
				srcpath = os.path.join(src_root, instrument_name, data_split, filename)
				duration = 5
				dur = int(librosa.get_duration(filename=srcpath))
				for i, offset in enumerate(range(0, int(dur/duration)*duration, duration)):
					y, sr = librosa.load(srcpath, offset=offset, duration=duration, sr=16000)
					os.makedirs(os.path.join(dest_root, instrument_name, data_split), exist_ok=True)
					dstpath = os.path.join(dest_root, instrument_name, data_split, filename.split('.')[0]+'_'+str(i)+'.wav')
					print(dstpath)
					sf.write(dstpath, y, sr)

