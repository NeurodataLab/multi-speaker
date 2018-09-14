"""
NeurodataLab LLC 02.03.2018
Written by Andrey Belyaev & Eva Kazimirova
"""
import argparse
import json
import mxnet as mx
import numpy as np
import os
import os.path as osp
from scipy import misc
from skimage import transform
from skimage.exposure import equalize_hist, adjust_sigmoid
from tqdm import tqdm

home = osp.dirname(osp.abspath(__file__))


class VoiceCounter:
    def __init__(self, model_path, model_epoch, batch_size, image_shape, sec_size,
                 interval_sec, interval_step_sec, context, out_dir):
        """
        Count voices on given spectrogram
        :param model_path: path to mxnet model
        :param model_epoch: mxnet model epoch
        :param batch_size: batch size
        :param image_shape: model's input shape
        :param sec_size: num pixel equals to one second
        :param interval_sec: sliding window size
        :param interval_step_sec: sliding window step
        :param context: device to use, -1 to use CPU, or N=0,1,... to use GPU_N
        :param out_dir: path to save results
        """
        # Initialize params
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_epoch)
        self.batch_size, self.image_shape = batch_size, tuple(image_shape)
        data_names, label_names = ['data'], ['separate_voices_label']
        data_shapes = [(self.batch_size, 3) + self.image_shape]

        self.second_size, self.interval_sec, self.interval_step_sec = sec_size, interval_sec, interval_step_sec
        self.interval_size = int(self.second_size * self.interval_sec)
        self.step_size = int(self.second_size * self.interval_step_sec)

        self.ctx = mx.gpu(context) if context >= 0 else mx.cpu(0)

        self.out_dir = out_dir
        if not osp.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # Initialize model
        self.model = mx.mod.Module(sym, data_names=data_names, label_names=label_names, context=self.ctx)
        self.model.bind(data_shapes=zip(data_names, data_shapes), for_training=False)
        self.model.init_params(arg_params=arg_params, aux_params=aux_params)

        # Zero forward
        self.model.forward(mx.io.DataBatch(data=[mx.nd.zeros(data_shapes[0])]), is_train=False)
        _ = self.model.get_outputs()[0].asnumpy()
        print 'Model init ok'

    def prepare_image(self, image):
        image = (image - image.min()) / (image.max() - image.min())
        image_hist = equalize_hist(image)
        image_sigmoid = adjust_sigmoid(image)
        return np.concatenate([image[np.newaxis], image_hist[np.newaxis], image_sigmoid[np.newaxis]])

    def process_batch(self, batch, seconds, info):
        self.model.forward(mx.io.DataBatch(data=[mx.nd.array(batch)]))
        predictions = self.model.get_outputs()[0].asnumpy()
        for p, s in zip(predictions, seconds):
            if s is not None:
                info[s] = float(p)

    def process_spectrogram(self, img_path):
        img_info = {}
        img = misc.imread(img_path).mean(axis=-1)
        W, H = img.shape[:2]
        batch, seconds = [], []
        for i in tqdm(range(0, H - self.interval_size, self.step_size)):
            batch_img = img[:, i: i + self.interval_size]
            batch_img = transform.resize(batch_img, self.image_shape)
            batch_img = self.prepare_image(batch_img)
            batch.append(batch_img)
            seconds.append(float(i + self.interval_size / 2.) / self.second_size)
            if len(batch) >= self.batch_size:
                self.process_batch(batch, seconds, img_info)
                batch, seconds = [], []

        if len(batch) > 0:
            batch.extend([np.zeros_like(batch[0]) for _ in range(len(batch), self.batch_size)])
            seconds.extend([None for _ in range(len(batch), self.batch_size)])
            self.process_batch(batch, seconds, img_info)

        im_name = '.'.join(img_path.split('/')[-1].split('.')[:-1])
        with open(osp.join(self.out_dir, '%s.json' % im_name), 'w') as f:
            json.dump(img_info, f, indent=2)

    def process_folder(self, folder):
        im_paths = os.listdir(folder)
        for n, im_path in enumerate(im_paths):
            print 'Start image %d/%d: %s' % (n + 1, len(im_paths), im_path)
            if im_path.endswith(('jpg', 'jpeg')):
                self.process_spectrogram(osp.join(folder, im_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--spec-path', type=str, default=None, help='Path to spectrogram')
    parser.add_argument('-d', '--dir', type=str, default=None, help='Path to directory with spectrograms')
    parser.add_argument('--model-path', type=str, default=osp.join(home, 'model/voice_counter'),
                        help='Path to mxnet model')
    parser.add_argument('--model-epoch', type=int, default=0, help='Model epoch')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--image-shape', type=int, nargs=2, default=(80, 40), help='model input shape')
    parser.add_argument('--sec-size', type=int, default=250, help='num pixel equals to one second')
    parser.add_argument('--interval', type=float, default=0.4, help='sliding window size')
    parser.add_argument('--step', type=float, default=0.01, help='sliding window step')
    parser.add_argument('--device', type=int, default=-1,
                        help='device to use, -1 to use CPU, or N=0,1,... to use GPU_N')
    parser.add_argument('--out-dir', type=str, default=osp.join(home, 'results'), help='path to save results')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.spec_path is not None or args.dir is not None, 'No spectrograms to process'

    voice_counter = VoiceCounter(model_path=args.model_path, model_epoch=args.model_epoch, batch_size=args.batch_size,
                                 image_shape=args.image_shape, sec_size=args.sec_size, interval_sec=args.interval,
                                 interval_step_sec=args.step, context=args.device, out_dir=args.out_dir)

    if args.spec_path is not None:
        voice_counter.process_spectrogram(args.spec_path)
    if args.dir is not None:
        voice_counter.process_folder(args.dir)
