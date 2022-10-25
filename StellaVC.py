import sys
import os
import torch
import librosa
from scipy.io.wavfile import write
from pydub import AudioSegment
import logging
import utils
import shutil
import numpy as np

from mel_processing import spectrogram_torch
from models import SynthesizerTrn
from hubert import load_hubert

# global variable
source_path = ''
export_path = 'sovits_cache/wavefile.wav'
flag_upload = False # 等待输入source path
flag_convert = False # 是否等待启动convert
flag_terminate = False # 是否结束线程
flag_mode = False # vc模式。False为hubert，True为flow
sid_source = None # 实际使用时，只有sid_target会用上
sid_target = None

class Sovits():
    def __init__(self, hubert_path, vits_path, hps):
        self.hubert_path = hubert_path
        self.vits_path = vits_path
        self.save_path = 'sovits_cache/temp.wav'
        self.play_path = 'sovits_cache/play.wav'
        self.download_path = 'sovits_cache/download.wav'
        self.hps = hps

        self._load_models()

    def _load_models(self):
        try:
            hubert = load_hubert(self.hubert_path)
            self.hubert = hubert
        except:
            print('Failed to load hubert model!')

        vits = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model)
        _ = vits.eval()

        try:
            _ = utils.load_checkpoint(self.vits_path, vits)
            self.vits = vits
        except:
            print('Failed to load vits model!')

    def inferene_hubert(self, source, sid=None, noise_scale=0.667, noise_scale_w=0.8, length_scale=0.6):
        with torch.inference_mode():
            # extract speech units
            unit = self.hubert.units(source)
            unit = torch.FloatTensor(unit)
            unit_lengths = torch.LongTensor([unit.size(1)])
            # convert voice
            # single speaker
            if sid is None:
                converted = self.vits.infer(
                    unit,
                    unit_lengths,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale)[0][0,0].data.float().numpy()
            # multi-speaker
            else:
                converted = self.vits.infer(
                    unit,
                    unit_lengths,
                    sid,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale)[0][0, 0].data.float().numpy()

        # write converted audio to cache
        write(self.save_path, 22050, converted)
        shutil.copy(self.save_path, self.play_path)
        shutil.copy(self.save_path, self.download_path)

    def inference_flow(self, source, sid_src, sid_tgt):
        spec = spectrogram_torch(source, self.hps.data.filter_length,
                                 self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                 center=False)
        spec_lengths = torch.LongTensor([spec.size(-1)])

        with torch.inference_mode():
            converted = self.vits.voice_conversion(
                spec, spec_lengths,
                sid_src=sid_src,
                sid_tgt=sid_tgt)[0][0, 0].data.float().numpy()

        # write converted audio to cache
        write(self.save_path, 22050, converted)
        shutil.copy(self.save_path, self.play_path)
        shutil.copy(self.save_path, self.download_path)

def get_logger(filename='test.log'):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(lineno)s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def ask_if_continue():
    while True:
        res = input('Contiue? (y/n): ')
        if res == 'y':
            break
        elif res == 'n':
            sys.exit(0)

def wait_upload():
    global flag_upload
    while True:
        if flag_upload:
            break

def load_audio(audio_path):
    global source_path
    global flag_upload
    if audio_path:
        source_path = audio_path
        flag_upload = True

def wait_convert():
    global flag_convert
    while True:
        if flag_convert:
            break


def convert_audio():
    global flag_upload, flag_convert, source_path
    if source_path:
        flag_upload = True
        flag_convert = True


def change_mode():
    global flag_mode
    if flag_mode:
        flag_mode = False
    else:
        flag_mode = True


def revise_path(origin_path):
    origin_path = origin_path.replace('\\', '/').replace('\n', '/n').replace('\r', '/r')
    revised_path = origin_path.replace('\t', '/t').replace('\a', '/a').replace('\b', '/b')

    return revised_path


def load_wav(audio_path: str):
    global source_path, source_path
    if audio_path.endswith('wav'):
        pass
    else:
        if audio_path.endswith('mp3'):
            audio = AudioSegment.from_mp3(audio_path)

        elif audio_path.endswith('ogg'):
            audio = AudioSegment.from_ogg(audio_path)

        elif audio_path.endswith('flv'):
            audio = AudioSegment.from_flv(audio_path)
        else:
            raise ValueError('Not supported audio format!')

        audio.export(export_path, format='wav')
        source_path = export_path

    source_hubert, sr = librosa.load(source_path)
    source_hubert = librosa.resample(source_hubert, sr, 22050)
    source_hubert = librosa.to_mono(source_hubert)
    source_hubert = torch.from_numpy(source_hubert).unsqueeze(0).unsqueeze(1)

    # source_flow = torch.FloatTensor(source.astype(np.float32))
    # source_flow = source_flow / 32768.0
    # source_flow = source_flow.unsqueeze(0)

    source_flow, _ = utils.load_wav_to_torch(source_path)
    source_flow = source_flow / 32768.0
    source_flow = source_flow.unsqueeze(0)

    return source_hubert, source_flow

def select_speaker(speaker_id):
    global sid_target
    sid_target = torch.LongTensor([speaker_id])


def terminate_vc():
    global flag_upload, flag_terminate
    flag_terminate = True
    flag_upload = True

def voice_conversion(hubert_path, vits_path, config_path):
    print('Loading models...')
    hps = utils.get_hparams_from_file(config_path)
    sovits = Sovits(hubert_path, vits_path, hps)
    print('Successfully loaded models!')

    while True:
        global flag_upload, flag_convert, flag_terminate, source_path, sid_source, sid_target
        flag_upload = flag_convert = flag_terminate = False

        print('Please input some audio...')
        wait_upload()

        if flag_terminate:
            print('Terminating...')
            break

        source_path = revise_path(source_path)
        print(f'Successfully loaded audio from {source_path}')

        source_hubert, source_flow = load_wav(source_path)

        wait_convert()
        print('Converting...')
        if not flag_mode:
            print('Hubert mode...')
            sovits.inferene_hubert(source_hubert, sid_target)
        else:
            print('Flow mode...')
            # vits的vc不是any-to-many，但可以通过自己转自己的方式提升效果
            sovits.inference_flow(source_flow, sid_target, sid_target)
        print('Successfully converted the source audio!')

    print('Terminated!')

# debug use
# if __name__ == "__main__":
    # hubert_path = '../models/hubert-soft.pt'
    # vits_path = '../models/sovits-nat-1.pth'
    # config_path = '../models/config-sovits-nat-1.json'
    # voice_conversion(hubert_path, vits_path, config_path)
