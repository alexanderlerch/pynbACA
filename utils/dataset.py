import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pyACA

# TODO license issue for download scripts: https://www.upf.edu/web/mtg/tonas
#TODO AL: it might be easiest to just use the files that I am always using from the slides: https://github.com/alexanderlerch/ACA-Slides/tree/2nd_edition/audio, particularly sax_example.mp3 

class PitchDataset(Dataset):
    def __init__(self, root='asset/pitch'):
        super(PitchDataset, self).__init__()
        self.wavList =  [os.path.join(root, fn) for fn in os.listdir(root) if fn.endswith('.wav')]
    
    def __getitem__(self, index):
        sr, wav = pyACA.ToolReadAudio(self.wavList[index])
        with open(self.wavList[index].replace('.wav', '.f0.Corrected.txt')) as f:
            # time, _, _, fz
            data = f.read().split('\n')
            time = np.stack([float(d.split()[0]) for d in data if d])
            freq = np.stack([float(d.split()[2]) for d in data if d])
            q_freq = np.stack([float(d.split()[3]) for d in data if d])
        return wav, sr, time, freq, q_freq
        
    def __len__(self):
        return len(self.wavList)


if __name__ == '__main__':
    os.chdir('..')
    dataset = PitchDataset()
    wav, time, freq  = dataset[0]
    print(wav.shape, time.shape, freq.shape)