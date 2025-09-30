# AgeTransformer-VLM-based-Augmentation

AgeTransformer: A VLM-enhanced Model for Facial Age Transformation extends the original AgeTransgan architecture with vision-language guidance and mixture-of-experts (MoE) relabeling to generate ten age-conditioned renderings for every input face.

## Table of Contents
- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [AgeTransformer CLI](#agetransformer-cli)
  - [Streamlit Demo](#streamlit-demo)
  - [MoE Estimator CLI](#moe-estimator-cli)
  - [Training](#training-coming-soon)
- [Data and Checkpoints](#data-and-checkpoints)
- [MoE Age Estimator](#moe-age-estimator)
- [Contributing](#contributing)
- [Citation](#citation)
- [Contact](#contact)

## Highlights
- Ten discrete age anchors learned by the AgeTransformer backbone.
- Vision-language guidance keeps identity stable while adjusting semantic age cues.
- MoE age estimator recalibrates FFHQ-Aging with ensemble-based uncertainty handling.
- CAF dataset receives DDColor restoration and ESRGAN super-resolution for sharper supervision.
- End-to-end recipes for inference demos and evaluation are included; full training release is planned.
- Ready-to-run AgeTransformer CLI (`agt_infer.py`), Streamlit demo (`app.py`), and MoE ensemble launcher (`moe_estimator.py`).

## Repository Layout
```
AgeTransformer-VLM-based-Augmentation/
|-- agt_infer.py            # CLI inference script (batched age translation)
|-- app.py                  # Streamlit demo UI
|-- models/                 # Place pretrained AgeTransformer + MoE checkpoints here
|-- module/                 # Core model, dataset, and loss implementations
|-- moe_age_estimator/      # MoE ensemble utilities (Janus-Pro, MiVOLO, ResNet50, VGG16)
|-- requirements.txt        # Python dependencies
|-- sample/                 # Auto-created during training for qualitative dumps
`-- train.py                # Placeholder entrypoint for future training release
```

## Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) with matching PyTorch build
- Conda or venv for isolated environments

Install PyTorch that matches your CUDA driver from [pytorch.org](https://pytorch.org/get-started/locally/) before installing the remaining packages.

## Quick Start

### 1. Clone and Environment Setup
```bash
git clone https://github.com/coconutbee/AgeTransformer-VLM-based-Augmentation.git
cd AgeTransformer-VLM-based-Augmentation

conda create -n atf python=3.10
conda activate atf
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick the right CUDA/CPU wheel
pip install -r requirements.txt
```

### AgeTransformer CLI
Download the required checkpoints (see [Data and Checkpoints](#data-and-checkpoints)) and place them under `models/`. Then run:
```bash
python agt_infer.py \
  --ckpt models/agetransformer.pt \
  --input path/to/face_or_directory \
  --out outputs/ \
  --targets 0 3 5 9
```
Key flags:
- `--size` must match the training resolution (default 128).
- `--targets` accepts any subset of the 10 anchor classes.
- `--device`, `--fp16`, and `--bf16` let you control acceleration.

### Streamlit Demo
For an interactive UI:
```bash
streamlit run app.py
```
Upload images, select target age IDs, and preview/download the generated grids directly from the browser.

### MoE Estimator CLI
`moe_estimator.py` batches Janus-Pro, MiVOLO, VGG16, and ResNet50 predictions, then fuses them with the dynamic MoE head to produce relabeled ages.
```bash
python moe_estimator.py
```
Before running, download and build the Janus-Pro 7B model (see [huggingface.co/deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)) and update `CONFIG['janus_model_path']` to point at the local checkpoint directory.
Defaults are controlled by the `CONFIG` dictionary near the top of the script:
- `image_folder`: root directory of input face crops (recurses into subfolders).
- `*_model_path`: local checkpoints for each expert; download links are listed below.
- The script expects the helper environments defined in the repo docs (`Janus`, `mivolo`, `IPCGAN`). Adjust the `conda run` commands if your setup differs.

Artifacts generated:
- `predictions_Janus.csv`: raw Janus outputs.
- `predictions.csv`: consolidated table containing expert scores and `MoE_Predicted_Age`.


## Data and Checkpoints
| Artifact | Description | Location / Link | Destination |
| --- | --- | --- | --- |
| MoE Age Estimator | Ensemble scripts + weights (Janus-Pro, MiVOLO, ResNet50, VGG16 fusion) | [mega](https://mega.nz/file/2U8lxRKJ#Z2KczVkP72AnvNawfK8tAGeNZknqrack3VGjbZZC6zM) | `models/age_estimator.pth` |
| CAF-Enhanced Dataset | DDColor + ESRGAN enhanced faces | [mega](https://mega.nz/folder/j1BWXa4T#DwHyfwBY9t84QxmS39ItIA) | `data/CAF_enhanced` |
| AgeTransformer Generator | Pretrained generator weights | [mega](https://mega.nz/file/XEk3HSTQ#X24EA0t0YlIJI8sIt6dN3iPHLu4rVagVfnI7q0uQ3iA) | `models/agetransformer.pt` |
| Expert Backbones | MiVOLO detector + checkpoint | [detector](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view), [MiVOLO](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) | `models/` |
| Expert Backbones | ResNet50, VGG16, Janus-Pro | [ResNet50](https://mega.nz/file/eVty2bSY#byRidRMlh5G26mZ-23-Y9gXwxtKSFuRSx-7J43aVK24), [VGG16](https://mega.nz/file/HclwGZrR#qNEedMY5N2rBIui3CRTa59SNM9oMXvjt3Pr3_qMZix0), [Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-7B) | `models/` |

Example `models/` directory after downloading:
```
models/
├── 79999_iter.pth
├── age_estimator.pth
├── agetransformer.pt
├── best_dynamic_moe_model.pth
├── model_best_loss
├── model_imdb_cross_person_4.22_99.46.pth.tar
└── yolov8x_person_face.pt
```

## MoE Age Estimator
- Experts: Janus-Pro, MiVOLO, ResNet50, and VGG16 (legacy models).
- Gating network produces logits `g_i`; mixture weights use `w_i = softmax(g_i)`.
- Final age prediction aggregates expert outputs: `age = sum_i(w_i * age_i)`.
- `moe_age_estimator/` contains ensemble inference code and utilities for dynamic expert fusion.

## Contributing
We welcome bug reports, feature suggestions, and pull requests once the public release is live. For large changes, open an issue first so we can align on the roadmap.

## Citation
If you leverage AgeTransformer-VLM-based-Augmentation or its datasets, please cite:
```
@misc{AgeTransformer2025,
  title  = {AgeTransformer: A VLM-enhanced Model for Facial Age Transformation},
  author = {Paul, ???},
  year   = {2025},
  note   = {Technical Report},
  url    = {https://github.com/USERNAME/AgeTransformer-VLM-based-Augmentation}
}
```

## Contact
- Email: <your_email@example.com>
- Issues: use the GitHub issue tracker once the repository is public.
- Release notes and announcements will be posted on the project page.
