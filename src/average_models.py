"""Average the weights of fine-tuned models that share a base model and head initialisation.

The result is one ordinary model (same architecture, size and speed as each input).
Averaging is done in fp32 and saved in the first model's dtype.

    python src/average_models.py <out_dir> <model_dir> <model_dir> [...]
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    out, dirs = Path(sys.argv[1]), [Path(d) for d in sys.argv[2:]]
    sds = [load_file(d / "model.safetensors") for d in dirs]
    assert all(sd.keys() == sds[0].keys() for sd in sds), "different architectures"
    avg = {}
    for k, v in sds[0].items():
        avg[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(v.dtype) if v.is_floating_point() else v
    out.mkdir(parents=True, exist_ok=True)
    save_file(avg, out / "model.safetensors", metadata={"format": "pt"})
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy(dirs[0] / name, out / name)
    a, b = (torch.cat([sd[k].double().flatten() for k in sd if sd[k].is_floating_point()]) for sd in sds[:2])
    print(f"averaged {len(sds)} models -> {out}; first two: weight cosine {float(a @ b / (a.norm() * b.norm())):.6f}, "
          f"relative distance {float((a - b).norm() / a.norm()):.4f}")


if __name__ == "__main__":
    main()
