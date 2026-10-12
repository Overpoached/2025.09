"""Reference CLIP preprocessing for one image, and a diff against the C++ output.

usage: python preprocess_ref.py image.jpg ref.bin [--compare pixel_values_ours.bin]
"""
import argparse
import numpy as np
from PIL import Image
from clip.clip import _transform   # the exact transform clip.load() returns; no model download

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("out")
p.add_argument("--compare")
a = p.parse_args()

ref = _transform(224)(Image.open(a.image)).numpy()          # [3, 224, 224] float32
ref.astype("<f4").tofile(a.out)
print("wrote", a.out, ref.shape)

if a.compare:
    ours = np.fromfile(a.compare, dtype="<f4")[: ref.size].reshape(ref.shape)
    d = np.abs(ours - ref)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    levels = np.rint(d * std * 255).astype(int)               # diff in uint8 pixel steps
    worst = np.unravel_index(d.argmax(), d.shape)
    print(f"max_abs {d.max():.3e}  exact {np.mean(d == 0):.4%}  "
          f"pixels off by >=1 level: {np.count_nonzero(levels)}  worst (c,y,x)={worst}")