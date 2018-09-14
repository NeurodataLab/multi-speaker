## Automatic detection of multi-speaker fragments with high time resolution

This is python implementation of the project, described in 
[Automatic detection of multi-speaker fragments with high time resolution](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1878.pdf).
The aim of the project is to detect fragments in audio files where there are more than one speakers.

### How to use

First step: compute spectrogram from input audio --> specs/SPEC.jpeg\
`python2 SpecCreator.py [--audio-path=AUDIO_PATH] [--dir=DIRECTORY_WITH_AUDIOS]`

Second step: process spectrogram by CNN --> results/RESULTS.json\
`pythn2 VoiceCounter.py [--spec-path=SPECTROGRAM_PATH] [--dir=DIRECTORY_WITH_SPECTROGRAMS]`

Output will be in json format with probabilities of more than one speakers talking in the each frame.

For full information about params see \
`python2 SpecCreator.py --help` and `python2 VoiceCounter.py --help`

### Requirements

* Linux
* python2
* numpy, scipy, scikit-image, matplotlib, [mxnet](https://mxnet.incubator.apache.org/install), tqdm

### Future updates

The output json file may need different processing depending on current aims. The code to obtain the results from the main paper, section 2.3, will be provided later (write the authors, if needed).

### Authors and citation

#### When using this code, please cite this [paper](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1878.pdf)

Authors: Belyaev Andrey, Kazimirova Evdokia

Support: Neurodatalab LLC, USA

Contact: e.kazimirova@neurodatalab.com
