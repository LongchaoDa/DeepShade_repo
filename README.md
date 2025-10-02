# DeepShade: Enable Shade Simulation by Text-Conditioned Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-2507.12103-b31b1b.svg)](https://arxiv.org/abs/2507.12103)

This is the official repository for **DeepShade**, a diffusion-based framework for shade simulation and generation.  
It accompanies our paper:

> **DeepShade: Enable shade simulation by text-conditioned image generation**  
> Longchao Da, Xiangrui Liu, Mithun Shivakoti, Thirulogasankar Pranav Kutralingam, Yezhou Yang, Hua Wei  
> *arXiv preprint arXiv:2507.12103, 2025*  

---

## 🌍 Background

Heatwaves are intensifying due to global warming, posing serious risks to public health. While shade is critical for reducing heat exposure, **current routing and mapping systems ignore shade information**, largely because:

- Shadows are **hard to estimate from noisy satellite imagery**, and  
- Generative models often lack sufficient **high-quality training data**.  

**DeepShade** addresses these challenges by combining **3D simulation**, **edge-aware diffusion modeling**, and **text-conditioning** to synthesize realistic shade patterns at different times of day and across urban layouts.

---

## ✨ Contributions

1. **Dataset Construction**  
   - Built an extensive dataset across diverse geographies, building densities, and urban forms.  
   - Used Blender-based 3D simulations + building outlines to capture realistic shadows under varying solar zenith angles and times of day.  
   - Aligned simulated shadows with satellite imagery to enable robust training.

2. **DeepShade Model**  
   - A diffusion model that **jointly leverages RGB and Canny edge layers**, highlighting edge features critical for shadow formation.  
   - Introduces **contrastive learning** to better model temporal shade changes.  
   - Supports **textual conditioning** (e.g., time of day, solar angles) to generate controllable shade images.

3. **Application**  
   - Demonstrated shade-aware route planning in **Tempe, Arizona**, computing shade ratios for real pedestrian paths.  
   - Offers insights for **urban planning**, **environmental design**, and **public health** under extreme heat.

---

## 📦 Repository Structure

This repo provides a **minimal baseline** using ControlNet for shade simulation.

- `ControlNet/` – training and inference code  
- `dataset/` – placeholder for toy and full datasets  
- `a_inference/analyze.ipynb` – notebook for analyzing shade image features and RGB distributions  

---

## 📂 Dataset

- **Toy Dataset** (quick start):  
  [Google Drive link](https://drive.google.com/file/d/1tkSzr3WZfflo4fQDpYdr4FXSiiEbi1Pg/view?usp=sharing)  
  Unzip into:
  ```bash
  DeepShade_repo/dataset/

- **Full Dataset** (research use):  
  [Hugging Face – DARL-ASU/DeepShade](https://huggingface.co/datasets/DARL-ASU/DeepShade)

Each dataset includes:
- A **JSON file** managing source, target, and prompt metadata.  
- Shade-augmented imagery aligned with RGB data.  

⚠️ Remember to **update the JSON paths** to match your environment (replace `YOURNAME` with your own username and path).

---

## 🚀 Usage

### Train the model
```bash
python DeepShade_repo/ControlNet/run_vanillaControlnet_train_dlc.py
```

### Test on a single image
```bash
python DeepShade_repo/ControlNet/a_inference/infer_single.py
```

### Analyze shade features
Open the notebook:
```bash
DeepShade_repo/ControlNet/a_inference/analyze.ipynb
```



## 📖 Citation

If you find our work useful, please cite:

```bibtex
@article{da2025deepshade,
  title={Deepshade: Enable shade simulation by text-conditioned image generation},
  author={Da, Longchao and Liu, Xiangrui and Shivakoti, Mithun and Kutralingam, Thirulogasankar Pranav and Yang, Yezhou and Wei, Hua},
  journal={arXiv preprint arXiv:2507.12103},
  year={2025}
}