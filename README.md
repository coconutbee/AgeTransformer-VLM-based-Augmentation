# AgeTransformer-VLM-based-Augmentation

> AgeTransformer-VLM-based-Augmentation builds on the proposed AgeTransformer architecture to produce ten age-conditioned renderings per input face, pairing MoE-based relabeling with enhanced training data.

## Overview
- Supplies ready-to-train AgeTransformer code plus curated datasets so researchers can reproduce and improve age translation.
- Offers pretrained AgeTransformer checkpoints to validate performance out-of-the-box and a pipeline to fine-tune on custom data.

## Highlights
- AgeTransformer backbone delivers ten discrete age anchors.
- Vision-language guidance stabilizes identity preservation while steering semantic age cues.
- Mixture-of-Experts age estimator recalibrates FFHQ-Aging with ensemble uncertainty handling.
- CAF dataset is color- and texture-enhanced via DDColor + ESRGAN for sharper supervision.
- End-to-end recipes for inference demos, evaluation, and full training are included.


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
| FFHQ-Aging Relabeled | FFHQ-Aging relabeled training data | Finished |
| CAF-Enhanced Dataset | DDColor + ESRGAN processed images paired with original metadata. | Releases (coming soon) |
| Pretrained AgeTransformer | Checkpoints ready for inference | Releases (coming soon) |

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
- The enhanced CAF dataset can download from our [cloud](https://mega.nz/folder/j1BWXa4T#DwHyfwBY9t84QxmS39ItIA).
- The enhanced CAF dataset made from CAF dataset. We applied [DDColor](https://github.com/piddnad/DDColor.git) for color restoration followed by [ESRGAN](https://github.com/TencentARC/GFPGAN.git) for super-resolution.

### Checkpoints
- Download the MoE age estimator[(link)](https://mega.nz/file/2U8lxRKJ#Z2KczVkP72AnvNawfK8tAGeNZknqrack3VGjbZZC6zM) and AgeTransformer pretrained weights from the Releases page and place them under `models/` directory.
- Download the experts' weights and place them under `models/`. MiVOLO[(face detector model)](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view)&[(mivolo checkpoint)](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view). ResNet50[(link)](https://mega.nz/file/eVty2bSY#byRidRMlh5G26mZ-23-Y9gXwxtKSFuRSx-7J43aVK24). VGG16[(link)](https://mega.nz/file/HclwGZrR#qNEedMY5N2rBIui3CRTa59SNM9oMXvjt3Pr3_qMZix0). Janus-Pro[(link)](https://huggingface.co/deepseek-ai/Janus-Pro-7B).
- Download the agetransformer [(pretrained weights)](https://mega.nz/file/XEk3HSTQ#X24EA0t0YlIJI8sIt6dN3iPHLu4rVagVfnI7q0uQ3iA) from the Releases page and place them under `models/` directory.

models/
├── 79999_iter.pth
├── age_estimator.pth
├── agetransformer.pt
├── best_dynamic_moe_model.pth
├── model_best_loss
├── model_imdb_cross_person_4.22_99.46.pth.tar
└── yolov8x_person_face.pt

## MoE Age Estimator
- Experts: Janus-Pro, [MiVOLO](https://github.com/WildChlamydia/MiVOLO.git), ResNet50(our previous work), VGG16(our previous work).
- Weighted aggregation uses gating scores with softmax normalization `w_i = softmax(g_i)`.
- Final prediction is `age = sum_i(w_i * age_i)`.

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
