# AgeTransformer-VLM-based-Augmentation

> zh: AgeTransformer-VLM-based-Augmentation 以我提出的 AgeTransformer 架構為核心，整合視覺語言模型與資料增強流程，建立可一次輸出十種年齡轉換結果的年齡變換器。
> en: AgeTransformer-VLM-based-Augmentation builds on the proposed AgeTransformer architecture to produce ten age-conditioned renderings per input face, pairing MoE-based relabeling with enhanced training data.

## 概述 | Overview
- zh: 針對 FFHQ-Aging 與 CAF 進行嚴謹的前處理，提供更可信賴的訓練標籤與乾淨樣本。
- zh: 設計 MoE 年齡預估器融合 Janus-Pro、MiVOLO、ResNet50、VGG16，以提升年齡標註一致性。
- en: Supplies ready-to-train AgeTransformer code plus curated datasets so researchers can reproduce and improve age translation.
- en: Offers pretrained AgeTransformer checkpoints to validate performance out-of-the-box and a pipeline to fine-tune on custom data.

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
1. Clone the repository and pull the latest release assets.
2. Create a Python 3.10+ environment (Conda or venv) and install PyTorch with CUDA support that matches your GPU.
3. Install project dependencies (a consolidated `requirements.txt` / `environment.yml` will ship with the code release).
4. Download the MoE estimator, relabeled FFHQ-Aging metadata, CAF-enhanced data, and pretrained checkpoints from the Releases page.
5. Configure dataset paths inside `configs/project.yaml` (to be provided) before running training or inference.

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
|-- configs/            # YAML configs for MoE, data pipelines, and training
|-- data/               # Place downloaded datasets here (see instructions below)
|-- moe_age_estimator/  # Ensemble inference utilities and checkpoints
|-- src/                # AgeTransformer models, trainers, evaluators
|-- tools/              # Helper scripts for preprocessing and visualization
`-- README.md
```

## Data Preparation
### FFHQ-Aging Relabeling
- Place original FFHQ-Aging images under `data/raw/ffhq_aging/`.
- Run the MoE estimator to produce a new `ffhq_aging_moe_labels.json` containing mean age, per-expert predictions, and confidence scores.
- Filter subjects using the provided reliability thresholds (for example, discard samples where expert variance exceeds configurable bounds).
- Export summary statistics (`stats/ffhq_aging_moe_report.csv`) to track age distribution balance.

### CAF Enhancement
- Place the raw CAF dataset under `data/raw/caf/`.
- Apply DDColor for color restoration followed by ESRGAN for super-resolution; scripts will output to `data/processed/caf_enhanced/`.
- Keep sidecar metadata linking enhanced images back to originals for auditing and ablation studies.

## MoE Age Estimator
- Experts: Janus-Pro (VLM-augmented), MiVOLO, ResNet50, VGG16.
- Weighted aggregation uses temperature-scaled confidences `w_i = softmax((c_i - mu)/tau)`.
- Final prediction is `age = sum_i(w_i * age_i)`; variance is tracked for filtering and curriculum scheduling.
- Provides both CLI (`python moe_age_estimator/infer.py --input ...`) and batch API integration for dataset relabeling.
- Calibration metrics (MAE, EMD, reliability diagrams) are logged for transparency.

## AgeTransformer Training
1. Build training manifest that pairs each face with its MoE age label and optional CAF-enhanced counterpart.
2. Configure age anchors in `configs/age_anchors.yaml` to control the ten output bins.
3. Launch training with distributed support, for example `python -m torch.distributed.run --nproc_per_node=4 src/train.py --config configs/age_transformer_base.yaml`.
4. Enable VLM guidance (Janus-Pro captions / embeddings) by pointing to the caption cache inside the config.
5. Monitor training via TensorBoard and validation scripts (`src/eval_age_consistency.py`).

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
