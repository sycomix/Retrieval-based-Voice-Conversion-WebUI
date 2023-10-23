import os, sys, traceback

# device=sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 5:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from fairseq import checkpoint_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

f = open(f"{exp_dir}/extract_f0_feature.log", "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(sys.argv)
model_path = "hubert_base.pt"

printt(exp_dir)
wavPath = f"{exp_dir}/1_16k_wavs"
outPath = (
    f"{exp_dir}/3_feature256" if version == "v1" else f"{exp_dir}/3_feature768"
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt(f"load model(s) from {model_path}")
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
printt(f"move model to {device}")
if device not in ["mps", "cpu"]:
    model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt(f"all-feature-{len(todo)}")
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = f"{wavPath}/{file}"
                out_path = f'{outPath}/{file.replace("wav", "npy")}'

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": feats.half().to(device)
                    if device not in ["mps", "cpu"]
                    else feats.to(device),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,  # layer 9
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt(f"{file}-contains nan")
                if idx % n == 0:
                    printt(f"now-{len(todo)},all-{idx},{file},{feats.shape}")
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
