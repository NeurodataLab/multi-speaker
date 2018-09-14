"""
NeurodataLab LLC 02.03.2018
Written by Andrey Belyaev & Eva Kazimirova
"""
import argparse
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import os.path as osp
import scipy.signal
from scipy.io import wavfile
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm


class SpecCreator:
    def __init__(self, out_path, sec_size, overlap):
        self.sec_size, self.overlap = sec_size, overlap
        self.out_path = out_path
        if not osp.exists(self.out_path):
            os.mkdir(self.out_path)

    def create_spectrogram(self, audio_path):
        """
        Creates spectrograms from audio file
        :param audio_path: path to audio
        :param sec_size: num pixel equals to one second
        :param overlap: sliding windows overlap
        """
        audio_name = audio_path.split("/")[-1].replace(".wav", "")
        fs, w = wavfile.read(audio_path)
        if len(w.shape) == 2:
            w = w[:, 0]
        dur = len(w) / fs

        cmap = plt.cm.get_cmap('Greys')
        cmap.set_under('w')
        f, t, sxx = scipy.signal.spectrogram(w, fs=fs, window='hann', nperseg=int(fs / 12.32),
                                             noverlap=int(self.overlap * (fs / 12.32)), mode='psd', nfft=16000)
        sxx_db = 10 * np.log10(abs(sxx[:1500, :]) / 2 * 10e-5)

        dpi = 50
        fig = plt.figure(figsize=(dur * self.sec_size // dpi, self.sec_size * 2 // dpi), dpi=dpi, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        extent = (0, dur * self.sec_size // dpi, 0, self.sec_size * 2 // dpi)
        plt.imshow(sxx_db[::-1, :], cmap=cmap, extent=extent, norm=mpl.colors.Normalize(vmin=-50, vmax=0, clip=False))
        plt.savefig(osp.join(self.out_path, '%s.jpeg' % audio_name), dpi=dpi, frameon=False)

        # Resize saved image in case of bad matplotlib result
        img = imread(osp.join(self.out_path, '%s.jpeg' % audio_name))
        img = resize(img, (dur * self.sec_size, self.sec_size * 2)[::-1])
        imsave(osp.join(self.out_path, '%s.jpeg' % audio_name), img)


def parse_args():
    home = osp.dirname(osp.abspath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--audio_path', type=str, default=None, help='Path to audio file')
    parser.add_argument('-d', '--dir', type=str, default=None, help='Directory with audio files')
    parser.add_argument('-o', '--out', type=str, default=osp.join(home, 'specs'), help='Path to save spectrograms')
    parser.add_argument('--sec-size', type=float, default=250, help='num pixel equals to one second')
    parser.add_argument('--overlap', type=float, default=0.8, help='sliding windows overlap')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.audio_path is not None or args.dir is not None, 'No audio files to process'

    if not osp.exists(args.out):
        os.mkdir(args.out)

    print 'Start processing'
    spec_creator = SpecCreator(args.out, args.sec_size, args.overlap)

    if args.audio_path is not None:
        spec_creator.create_spectrogram(args.audio_path)

    if args.dir is not None:
        for p in tqdm(os.listdir(args.dir)):
            if p.endswith('wav'):
                spec_creator.create_spectrogram(osp.join(args.dir, p))
