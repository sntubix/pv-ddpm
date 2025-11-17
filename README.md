# Bridging the Data Gap: Spatially Conditioned Diffusion Model for Anomaly Generation in Photovoltaic Electroluminescence Images
<img width="1268" height="712" alt="Screenshot from 2025-10-23 08-39-26" src="https://github.com/user-attachments/assets/000cb463-5da8-43f3-af2f-b36cd21cdd14" />

## Abstract
Reliable anomaly detection in photovoltaic (PV) modules is critical for maintaining solar energy efficiency. However, developing robust computer vision models for PV inspection is constrained by the scarcity of large-scale, diverse, and balanced datasets. This study introduces PV-DDPM, a spatially conditioned denoising diffusion probabilistic model that generates anomalous electroluminescence (EL) images across four PV cell types: multi-crystalline silicon (multi-c-Si), mono-crystalline silicon (mono-c-Si), half-cut multi-c-Si, and interdigitated back contact (IBC) with dogbone interconnect. PV-DDPM enables controlled synthesis of single-defect and multi-defect scenarios by conditioning on binary masks representing structural features and defect positions. To the best of our knowledge, this is the first framework that jointly models multiple PV cell types while supporting simultaneous generation of diverse anomaly types. We also introduce E-SCDD, an enhanced version of the SCDD dataset, comprising 1,000 pixel-wise annotated EL images spanning 30 semantic classes, and 1,768 unlabeled synthetic samples. Quantitative evaluation shows our generated images achieve a Fr´echet Inception Distance (FID) of 4.10 and Kernel Inception Distance (KID) of 0.0023 ± 0.0007 across all categories. Training the vision–language anomaly detection model AA-CLIP on E-SCDD, compared to the SCDD dataset, improves pixel-level AUC and average precision by 1.70 and 8.34 points, respectively.

## Dataset
E-SCDD dataset available at: https://huggingface.co/datasets/shivahanifi/extended-scdd

## Result
<img width="1366" height="801" alt="Screenshot from 2025-11-11 14-47-29" src="https://github.com/user-attachments/assets/73fc3bd5-3d49-4d0a-8f36-f7b24e8796a9" />

## Getting Started
### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate pv-diffusion
```
### 2. Download Pretrained Weights 

```bash
mkdir models
cd models
gdown --id 1xBTqsSK193FjicpVUAs-1n04C2nt6Qmz

```

### 3. Training
```bash
python train.py --dataset_path <path to dataset> --resume_weight <path tp pretrained weights>
```

## Citation

If you use PV-DDPM or E-SCDD in your research, please cite these work:
```
@misc{hanifi2025bridgingdatagapspatially,
      title={Bridging the Data Gap: Spatially Conditioned Diffusion Model for Anomaly Generation in Photovoltaic Electroluminescence Images}, 
      author={Shiva Hanifi and Sasan Jafarnejad and Marc Köntges and Andrej Wentnagel and Andreas Kokkas and Raphael Frank},
      year={2025},
      eprint={2511.09604},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2511.09604}, 
}

@article{pratt2023benchmark,
      title={A benchmark dataset for defect detection and classification in electroluminescence images of PV modules using semantic segmentation},
      author={Pratt, Lawrence and Mattheus, Jana and Klein, Richard},
      journal={Systems and Soft Computing},
      pages={200048},
      year={2023},
      publisher={Elsevier} }

```
