<div align='center'>
<h1 align="center">[ECCV 2026] Anchoring on Reality: Breaking the Pseudo-Target Ceiling in Makeup Transfer</h1>
    Bo Wei<sup> 1*</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=wLTXeNwAAAAJ&hl=en&oi=ao' target='_blank'>Xianhui Lin</a><sup> 2†</sup>&emsp;
    Yi Dong<sup> 2</sup>&emsp;
    Zhongzhong Li<sup> 2</sup>&emsp;
    Zonghui Li<sup> 2</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=en&user=BhmLztgAAAAJ' target='_blank'>Zirui Wang</a><sup> 2</sup>&emsp;
</div>

<div align='center'>
    <a href='https://scholar.google.com/citations?hl=en&user=12MzNVkAAAAJ' target='_blank'>Jiachen Yang</a><sup> 2</sup>&emsp;
    Xing Liu<sup> 2</sup>&emsp;
    Hong Gu<sup> 2</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=en&user=tmT_voUAAAAJ' target='_blank'>Xiaoming Li</a><sup> 3✉</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=en&user=rUOpCEYAAAAJ' target='_blank'>Wangmeng Zuo</a><sup> 1✉</sup>
</div>

<div align='center'>
    <sup>1 </sup>Harbin Institute of Technology&emsp;
    <sup>2 </sup>vivo BlueImage Lab&emsp;
    <sup>3 </sup>Nanjing University
</div>

<div align='center'>
    <small><sup>*</sup> Work done during an internship at vivo.</small>&emsp;
    <small><sup>†</sup> Project lead</small>
</div>

<div align="center">
  <p>
    <a href="https://arxiv.org/abs/2606.31089" target="_blank"><img src="https://img.shields.io/badge/arXiv-ART-red" alt="arXiv link"></a>&nbsp;
    <a href="https://csbowei.github.io/ART/" target="_blank"><img src="https://img.shields.io/badge/Project-Homepage-green" alt="project homepage"></a>&nbsp;
    <a href="https://huggingface.co/csbowei/ART" target="_blank"><img src="https://img.shields.io/badge/🤗 Hugging Face-Models-blue" alt="HF Models"></a>&nbsp;
    <a href="https://huggingface.co/spaces/csbowei/ART-Makeup-Transfer" target="_blank"><img src="https://img.shields.io/badge/🤗 Hugging Face-Space-orange" alt="HF Space"></a>&nbsp;
    <a href="https://drive.google.com/drive/folders/1UfUUAk86fWWzYsZcGOzpTmAWa32IwK21?usp=drive_link" target="_blank"><img src="https://img.shields.io/badge/Dataset-MF2K-purple?logo=googledrive&logoColor=white" alt="MF2K Dataset"></a>&nbsp;
    <!-- <a href="" target="_blank"><img src="https://img.shields.io/github/stars/csbowei/ART?style=social" alt="GitHub stars"></a> -->
  </p>
</div>

## 🔍 Overview

<div align="center">
<img src="./assets/example_gif.gif" width="100%">
</div>

<div align="center">
<img src="./assets/teaser.png" width="100%">
</div>

**Figure 1.** ART enables high-fidelity transfer of diverse makeup styles (simple to complex) at up to 2K resolution, while preserving identity, geometry, and makeup placement under challenging expressions, occlusions, and cross-gender scenarios.

<p align="center">
  <img src="./assets/pipeline.png" width="100%" alt="ART pipeline overview">
</p>

**Figure 2. Method overview.**
ART is a two-stage framework that shifts supervision from the synthetic pseudo-target to the real reference.
**Stage I** performs pseudo-target initialization by training the transfer model for global makeup placement and an auxiliary makeup remover for bare-skin extraction.
**Stage II** predicts a differentiable makeup carrier ẑ at noise level σ<sub>tr</sub>, and then reconstructs the real reference from its bare-skin counterpart conditioned on ẑ. Since ẑ remains differentiable, the gradient of L<sub>refine</sub> back-propagates into the transfer prediction, helping recover fine-grained makeup cues and suppress synthetic artifacts, while L<sub>bottleneck</sub> preserves the global structure.


## ✨ Highlights

⚓ **Reality-Anchored Refinement.** We propose ART, a two-stage DiT framework that breaks the *pseudo-target ceiling*. By anchoring a refinement cycle to the real reference, ART overrides pseudo-target artifacts while stabilizing training through a controlled-noise bottleneck.

💄 **A 2K-Resolution In-the-Wild Makeup Dataset.** We introduce MakeupFaces2K (MF2K), the first 2K-resolution makeup portrait dataset with 8,573 images, spanning diverse makeup intensities and underrepresented demographics to support high-fidelity makeup transfer and related tasks.

🏆 **State-of-the-Art Performance.** ART achieves superior makeup fidelity across diverse styles, consistently preserving fine-grained details and source identity even under occlusions and extreme expressions.


## 🗓️ Plan & Updates
Higher-resolution model weights and training code will be released **as soon as possible** after internal review.

**Plan：**

- [x] Inference code
- [x] 512×512 LoRA weights
- [x] 1024×1024 LoRA weights
- [ ] Higher-resolution weights
- [ ] Training code
- [x] MF2K dataset


**Updates：**
- **`2026/07/09`**: We launched the [Hugging Face Space demo](https://huggingface.co/spaces/csbowei/ART-Makeup-Transfer). Try ART online!
- **`2026/07/09`**: We released an interactive Gradio app and added more example images.
- **`2026/07/07`**: We released the MF2K dataset.
- **`2026/07/07`**: We released the 1024×1024 LoRA weights.
- **`2026/07/01`**: We released the 512×512 LoRA weights.
- **`2026/06/30`**: We released the inference code. Continuous updates, stay tuned!

## 🚀 Getting Started

### 🛠️ 1. Environment Setup

```bash
# Create and activate environment
conda create -n art python=3.10
conda activate art

# Install dependencies
pip install -r ./requirements.txt
```

### 📦 2. Download Pretrained Weights

[Download](https://huggingface.co/csbowei/ART) the ART LoRA checkpoints and place them under `./checkpoints/`, or use the following command:

```bash
# pip install -U "huggingface_hub[cli]"

# If Hugging Face is not directly accessible, you can use hf-mirror:
# export HF_ENDPOINT=https://hf-mirror.com

hf download csbowei/ART --local-dir ./checkpoints --include "*.safetensors"
```

| Model | Link |
| :--- | :--- |
| ART LoRA | [csbowei/ART](https://huggingface.co/csbowei/ART) |
| FLUX.1-Kontext-dev | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) |

### ▶️ 3. Usage

```bash
# Makeup transfer: apply the reference's makeup onto the input face
python infer.py \
    --task        mt \
    --input       ./examples/source/src_001.jpg \
    --ref         ./examples/ref/ref_001.jpg \
    --lora_path   ./checkpoints/art_transfer_lora_1024.safetensors \
    --resolution  1024 \
    --output_dir  ./examples/outputs \
    --save_concat
```

```bash
# Makeup removal: strip cosmetics from the input face
python infer.py \
    --task        demakeup \
    --input       ./examples/ref/ref_002.jpg \
    --lora_path   ./checkpoints/art_demakeup_lora_1024.safetensors \
    --resolution  1024 \
    --output_dir  ./examples/outputs \
    --save_concat
```

The generated results should look similar to the following:

<div align="center">
<table>
  <tr>
    <td align="center"><img src="./assets/example_transfer.png" width="100%"></td>
    <td align="center"><img src="./assets/example_demakeup.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Makeup Transfer</b> (<code>--task mt</code>)</td>
    <td align="center"><b>Makeup Removal</b> (<code>--task demakeup</code>)</td>
  </tr>
</table>
</div>

> The `--resolution` parameter should match the resolution of the LoRA model. `--save_concat` creates a side-by-side concatenated image of the input and output.

Or you can launch the interactive Gradio app:

```bash
python app.py
```

After launching, open the local URL shown in the terminal. The app automatically lists examples from `./examples/source/` and `./examples/ref/`.


## 📊 Datasets

### 💄 MF2K

<div align="center">
<img src="./assets/mf2k.png" width="100%">
</div>

**Figure 3. MF2K, the first 2K-resolution in-the-wild makeup dataset.**
MF2K contains 8,573 images spanning four makeup intensities from bare skin to artistic styles, with broad demographic diversity.
Compared with existing datasets, it captures fine-grained makeup textures and thus serves as a demanding benchmark for high-fidelity transfer.

#### Download

Due to redistribution constraints of source images, MF2K is released as metadata and reconstruction scripts. Please download the original images from the URLs provided in the metadata, rename them according to `metadata.origin_path`, and organize them under `images/origin/`.

- [Google Drive](https://drive.google.com/drive/folders/1UfUUAk86fWWzYsZcGOzpTmAWa32IwK21?usp=drive_link)
- [Baidu Netdisk](https://pan.baidu.com/s/1p1Gizr4lW5II-YnNcGsK4w?pwd=8y6r)

#### File Overview

```text
MF2K/
├── metas/
│   ├── train.jsonl          # 8,373 training images, one JSON object per line
│   └── test.json            # 100 source/reference test pairs
├── make_crop.py             # Reconstruct aligned crops from original images
└── images/
    ├── origin/              # User-downloaded original images
    ├── crop/                # Reconstructed aligned face crops
    └── crop_resize2048/     # 2048x2048 resized crops generated by the script
```

Metadata entry format:

```text
{
  "id": "<image_id>",                    // Image id
  "split": "train | test",
  "makeup_category": "bare | light | heavy | artistic",
  "gender": "male | female",
  "metadata": {
    "photo_url": "...",                  // Original image URL
    "author": "...",                     // Image author or provider
    "license": "...",                    // Source image license
    "origin_size": [<width>, <height>],  // Original image size
    "origin_path": "images/origin/...",  // Path for the downloaded original image
    "crop_path": "images/crop/...",      // Output path for the aligned crop
    "crop_resize2048_path": "images/crop_resize2048/...",
    "origin_md5": "...",                 // MD5 checksum of the original image
    "crop_md5": "...",                   // MD5 checksum of the crop image
    "face": {
      "crop_size": <crop_size>,          // Aligned crop resolution before resizing
      "affine_matrix": [[...], [...]]    // Affine transform from original image to crop
    }
  }
}
```

#### Data Splits

The training split contains 8,373 images across four makeup categories: bare, light, heavy, and artistic. All training face crops are at least 2048x2048. 

The test split contains 100 source/reference pairs, where each source is bare-faced and each reference uses artistic makeup. Some test crops are below 2048x2048 but remain approximately 1K or higher. All evaluations reported in our paper are conducted at 512x512 resolution.

#### Reconstruct Crops

After downloading the original images, place each file according to `metadata.origin_path`, for example:

```text
MF2K/images/origin/00001.png
MF2K/images/origin/08374.jpg
```

Then generate aligned crops and 2048x2048 resized crops:

```bash
python MF2K/make_crop.py \
    --dataset_root ./MF2K \
    --workers 16
```

Outputs are written to `images/crop/` and `images/crop_resize2048/`. Use `--no_resize2048` if only aligned crops are needed.


### 🧪 Evaluation Datasets

Our evaluation covers four datasets with complementary challenges:

- **MT**: A standard makeup-transfer dataset with mostly aligned frontal portraits under controlled settings. BeautyGAN

- **MT-Wild**: An in-the-wild makeup transfer dataset with varied poses, expressions, and backgrounds under unconstrained settings. [PSGAN](https://github.com/wtjiang98/PSGAN), [Google Drive](https://drive.google.com/drive/folders/1ubqJ49ev16NbgJjjTt-Q75mNzvZ7sEEn)

- **LADN**: A makeup transfer dataset with diverse and extreme makeup styles involving large color and spatial variations. [LADN](https://github.com/wangguanzhi/LADN), [Google Drive](https://drive.google.com/file/d/1Y0AlgKSVZNsjUG4u0YUHzoxtsG0qgexZ/view)

- **MF2K**: The artistic subset split of the MF2K dataset, emphasizing high-frequency cosmetic details for stress-testing transfer fidelity.


## 🙏 Acknowledgements

We sincerely thank the authors of [FLUX.1-Kontext](https://github.com/black-forest-labs/flux) for releasing the foundational generative model that our work builds upon. We also acknowledge [Diffusion-4k](https://github.com/zhang0jhon/diffusion-4k) for its inspiring wavelet loss design.

We are also grateful to previous makeup transfer works and datasets, including [PSGAN](https://github.com/wtjiang98/PSGAN), [EleGANt](https://github.com/chenyu-yang-2000/elegant), [MAD](https://basiclab.github.io/MAD/), [SHMT](https://github.com/Snowfallingplum/SHMT), [Stable-Makeup](https://github.com/Xiaojiu-z/Stable-Makeup), BeautyGAN, and [LADN](https://github.com/wangguanzhi/LADN). Their valuable contributions and released resources have inspired our research and enabled fair comparison.


## 📜 Citation

If you find our work or code useful for your research, please cite:

```bibtex
@article{wei2026art,
  title={Anchoring on Reality: Breaking the Pseudo-Target Ceiling in Makeup Transfer},
  author={Wei, Bo and Lin, Xianhui and Dong, Yi and Li, Zhongzhong and Li, Zonghui and Wang, Zirui and Yang, Jiachen and Liu, Xing and Gu, Hong and Li, Xiaoming and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2606.31089},
  year={2026}
}
```
