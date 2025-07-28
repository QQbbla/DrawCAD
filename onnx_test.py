import soundfile as sf
import numpy as np
import torch
import onnxruntime
from modelscope.models.audio.ans.zipenhancer import mag_pha_stft, mag_pha_istft
from modelscope.utils.audio.audio_utils import audio_norm

# onnx模型路径

onnx_model_path = "onnx_model.onnx"

audio_path = '001.wav'
output_path = 'output.wav'

is_verbose = True


class OnnxModel:
    def __init__(self, onnx_filepath, providers=None):
        self.onnx_model = onnxruntime.InferenceSession(onnx_filepath, providers=providers)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def __call__(self, noisy_wav):
        n_fft = 400
        hop_size = 100
        win_size = 400

        norm_factor = torch.sqrt(noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0))
        if is_verbose:
            print(f"norm_factor {norm_factor}")

        noisy_audio = (noisy_wav * norm_factor)

        noisy_amp, noisy_pha, _ = mag_pha_stft(
            noisy_audio,
            n_fft,
            hop_size,
            win_size,
            compress_factor=0.3,
            center=True)

        ort_inputs = {self.onnx_model.get_inputs()[0].name: self.to_numpy(noisy_amp),
                      self.onnx_model.get_inputs()[1].name: self.to_numpy(noisy_pha),
                      }
        ort_outs = self.onnx_model.run(None, ort_inputs)

        amp_g = torch.from_numpy(ort_outs[0])
        pha_g = torch.from_numpy(ort_outs[1])

        if is_verbose:
            print(f"Enhanced amplitude mean and std: {torch.mean(amp_g)} {torch.std(amp_g)}")
            print(f"Enhanced phase mean and std: {torch.mean(pha_g)} {torch.std(pha_g)}")

        wav = mag_pha_istft(
            amp_g,
            pha_g,
            n_fft,
            hop_size,
            win_size,
            compress_factor=0.3,
            center=True)

        wav = wav / norm_factor

        wav = self.to_numpy(wav)

        return wav


onnx_model = OnnxModel(onnx_model_path)


wav, fs = sf.read(audio_path)
wav = audio_norm(wav).astype(np.float32)
noisy_wav = torch.from_numpy(np.reshape(wav, [1, wav.shape[0]]))

if is_verbose:
    print(f"wav {wav}")
    print(f"noisy_wav {noisy_wav}")

enhanced_wav = onnx_model(noisy_wav)

if is_verbose:
    print(f"enhanced_wav {enhanced_wav}")

sf.write(output_path, (enhanced_wav[0] * 32768).astype(np.int16), fs)