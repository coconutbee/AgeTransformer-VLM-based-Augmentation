# AgeTransformer-VLM-based-Augmentation

> zh: AgeTransformer-VLM-based-Augmentation 以我提出的 AgeTransformer 架構為核心，整合視覺語言模型與資料增強流程，建立可一次輸出十種年齡轉換結果的年齡變換器。
> en: AgeTransformer-VLM-based-Augmentation builds on the proposed AgeTransformer architecture to produce ten age-conditioned renderings per input face, pairing MoE-based relabeling with enhanced training data.

## Overview
- Supplies ready-to-train AgeTransformer code plus curated datasets so researchers can reproduce and improve age translation.
- Offers pretrained AgeTransformer checkpoints to validate performance out-of-the-box and a pipeline to fine-tune on custom data.

## Highlights
- AgeTransformer backbone delivers ten discrete age anchors (configurable default: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95 years).
- Vision-language guidance stabilizes identity preservation while steering semantic age cues.
- Mixture-of-Experts age estimator recalibrates FFHQ-Aging with ensemble uncertainty handling.
- CAF dataset is color- and texture-enhanced via DDColor + ESRGAN for sharper supervision.
- End-to-end recipes for inference demos, evaluation, and full training are included.

## Repository Roadmap
- [x] Project README and high-level documentation.
- [ ] Release MoE age estimator checkpoints and inference scripts.
- [ ] Publish FFHQ-Aging relabeled metadata and statistics.
- [ ] Publish CAF enhanced dataset splits and metadata.
- [ ] Release AgeTransformer training/inference source code.
- [ ] Upload pretrained AgeTransformer weights and demo notebooks.

## Pipeline at a Glance
```
Raw FFHQ-Aging --> MoE Age Estimator --> Clean age labels -->
                                                   |
CAF dataset --> DDColor --> ESRGAN --> Enhanced faces -->
                                                   |
                        AgeTransformer training + VLM-guided augmentation --> Ten age renders
```

## Provided Resources
| Artifact | Description | Availability |
| --- | --- | --- |
| MoE Age Estimator | Ensemble scripts + weights (Janus-Pro, MiVOLO, ResNet50, VGG16 fusion). | Releases (coming soon) |
| FFHQ-Aging Relabeled | Metadata `.json` / `.csv`, reliability scores, quality flags. | Releases (coming soon) |
| CAF-Enhanced Dataset | DDColor + ESRGAN processed images paired with original metadata. | Releases (coming soon) |
| AgeTransformer Training Code | Data loaders, trainer, configs, VLM guidance modules. | `src/` directory (coming soon) |
| Pretrained AgeTransformer | Checkpoints ready for inference + demo notebooks. | Releases (coming soon) |

## Getting Started
1. Clone the repository.
```bash
git clone https://github.com/coconutbee/AgeTransformer-VLM-based-Augmentation.git
```
2. Create a Python 3.10+ environment and install PyTorch with CUDA support that matches your GPU.
```bash
conda create -n atf python=3.10
conda activate atf
pip3 install torch torchvision # install the correct version from pytorch.org
```
3. Install project dependencies.
```bash
pip install -r requirements.txt
```
4. Download the MoE estimator, relabeled FFHQ-Aging metadata, CAF-enhanced data, and pretrained checkpoints from the Releases page.
5. 
### Environment Setup Example
```bash
conda create -n atf python=3.10
conda activate atf
pip3 install torch torchvision # install the correct version from pytorch.org
pip install -r requirements.txt  # file will be added with the code release
```

### Directory Layout (expected)
```
AgeTransformer-VLM-based-Augmentation/
|-- data/               # Place downloaded datasets here (see instructions below)
|-- moe_age_estimator/  # Ensemble inference utilities and checkpoints
|-- module/             # AgeTransformer functions
`-- README.md
```

## Data Preparation
### FFHQ-Aging Relabeling
- Place our balanced & relabeled FFHQ-Aging [training_data](https://mega.nz/folder/SUM1GADC#4APMLfB6qQFPbDbK4kXgCw) and [validation_data](https://mega.nz/folder/2I8kUJID#oyv5ckiiJV3knq_ktZqIPg) images under `data/`.

### CAF Enhancement
- The raw CAF dataset can download from our [cloud](http://www.vision.caltech.edu/datasets/caf/).
- Apply [DDColor](https://github.com/piddnad/DDColor.git) for color restoration followed by [ESRGAN](https://github.com/TencentARC/GFPGAN.git) for super-resolution; If you want to download the enhanced data directly you can find it by the [link](http://www.vision.caltech.edu/datasets/caf/).

### Checkpoints
- Download the MoE age estimator[(link)](https://mega.nz/file/2U8lxRKJ#Z2KczVkP72AnvNawfK8tAGeNZknqrack3VGjbZZC6zM) and AgeTransformer pretrained weights from the Releases page and place them under `models/` directory.

## MoE Age Estimator
- Experts: Janus-Pro (VLM-augmented), MiVOLO, ResNet50[(our previous work)](https://link.springer.com/chapter/10.1007/978-3-030-89131-2_27), VGG16[(our previous work)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720573.pdf).
- Weighted aggregation uses gating scores with softmax normalization `w_i = softmax(g_i)`.
- Final prediction is `age = sum_i(w_i * age_i)`.
- Provides both CLI (`python moe_age_estimator/infer.py --input ...`) and batch API integration for dataset relabeling.

## AgeTransformer Training
1. Launch training, for example `python train.py`.

## Inference & Evaluation
- Use `src/infer.py` to generate ten age-shifted images from a single input; specify `--ages` to override default anchors.
- Identity preservation can be evaluated with ArcFace or similar embeddings (scripts will be located in `tools/eval_identity.py`).
- Age accuracy is verified by the MoE estimator in evaluation-only mode for each generated output.
- Quantitative reports (FID, LPIPS, age MAE) and qualitative grids are written to `outputs/experiments/<run_name>/`.

## Contributing
- Bug reports, feature requests, and pull requests are welcome once the initial code release is published.
- Please open an issue before submitting major changes so we can coordinate roadmap alignment.

## Citation
If you use AgeTransformer-VLM-based-Augmentation or the accompanying datasets in your research, please cite the paper:
```
@misc{AgeTransformer2025,
  title  = {AgeTransformer-VLM-based-Augmentation},
  author = {Paul, ???},
  year   = {2025},
  note   = {Technical Report},
  url    = {https://github.com/USERNAME/AgeTransformer-VLM-based-Augmentation}
}
```

## Contact
- Email: <your_email@example.com>
- Issues: please use the GitHub issue tracker once the repository is public.
- Updates will be announced on the project page and release notes.
