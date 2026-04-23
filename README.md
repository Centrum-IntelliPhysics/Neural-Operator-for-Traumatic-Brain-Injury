# Multimodal Neural Operators for Real-Time Biomechanical Modelling of Traumatic Brain Injury

**Authors:**  
[Anusha Agarwal](https://scholar.google.com/citations?user=VLIkDnkAAAAJ&hl=en), [Dibakar Roy Sarkar](https://scholar.google.com/citations?user=Sz4nHdYAAAAJ&hl=en), and [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en)

📄 **Published in:** *Computer Methods and Programs in Biomedicine* (2026)  
🔗 **Paper:** [https://www.sciencedirect.com/science/article/pii/S0169260726001586](https://www.sciencedirect.com/science/article/pii/S0169260726001586)  
🔗 **DOI:** [10.1016/j.cmpb.2026.109398](https://doi.org/10.1016/j.cmpb.2026.109398)

---

## Abstract

Traumatic brain injury (TBI) remains a major public health concern, with over 69 million cases annually worldwide. Accurate patient-specific biomechanical modeling is critical for injury risk assessment, but it requires integrating heterogeneous data sources — volumetric neuroimaging, scalar demographic parameters, and acquisition metadata. Conventional finite element solvers remain too computationally expensive for time-sensitive clinical settings, while existing neural operator formulations have not been systematically evaluated for fusing high-dimensional medical imaging with heterogeneous patient-specific metadata.

This study presents a systematic investigation of **multimodal neural operator architectures** for brain biomechanics. We reformulate TBI modeling as a multimodal operator learning problem and propose two fusion strategies: **field projection** for Fourier Neural Operator (FNO) based architectures (broadcasting scalars onto spatial grids) and **branch decomposition** for Deep Operator Networks (DeepONet) (separate encoding with multiplicative fusion). Four architectures — FNO, Factorized FNO (F-FNO), Multi-Grid FNO (MG-FNO), and DeepONet — are extended with multimodal fusion mechanisms and evaluated on 249 *in vivo* Magnetic Resonance Elastography (MRE) datasets across physiologically relevant frequencies (20–90 Hz).

---

## Key Contributions

- **Systematic investigation** of multimodal neural operator architectures for brain biomechanics prediction
- **Two fusion strategies** tailored to heterogeneous biomedical data:
  - *Field projection* for FNO-based architectures
  - *Branch decomposition* with multiplicative fusion for DeepONet
- **Patient-specific prediction framework** mapping volumetric MRI, scalar demographics, and acquisition parameters to full-field brain displacement
- **Comprehensive benchmark** across four neural operator architectures with detailed accuracy, efficiency, and failure-mode analysis
- **Real-time capability** with inference orders of magnitude faster than finite element solvers

---

## Results Summary

Experiments demonstrate significant computational speedups while preserving predictive accuracy across physiologically relevant frequency ranges (20–90 Hz). No single architecture dominates across all criteria — each offers distinct trade-offs between accuracy, spatial fidelity, and computational cost.

### Real Displacement Fields

| **Model**  | **MAE**              | **MSE**              | **RMSE**             | **Accuracy**         |
|------------|----------------------|----------------------|----------------------|----------------------|
| FNO        | 0.0438 ± 0.0335      | 0.0051 ± 0.0068      | 0.0567 ± 0.0439      | 0.8813 ± 0.1582      |
| F-FNO      | 0.0427 ± 0.0329      | 0.0053 ± 0.0069      | 0.0572 ± 0.0451      | 0.8853 ± 0.1481      |
| MG-FNO     | 0.0397 ± 0.0315      | 0.0044 ± 0.0061      | 0.0515 ± 0.0417      | 0.8938 ± 0.1481      |
| **DeepONet** | **0.0350 ± 0.0314** | **0.0039 ± 0.0057** | **0.0463 ± 0.0421** | **0.9000 ± 0.1465** |

### Imaginary Displacement Fields

| **Model**  | **MAE**              | **MSE**              | **RMSE**             | **Accuracy**         |
|------------|----------------------|----------------------|----------------------|----------------------|
| FNO        | 0.0468 ± 0.0472      | 0.0074 ± 0.0153      | 0.0608 ± 0.0606      | 0.8628 ± 0.1832      |
| F-FNO      | 0.0543 ± 0.0460      | 0.0090 ± 0.0187      | 0.0720 ± 0.0620      | 0.8471 ± 0.1703      |
| **MG-FNO** | **0.0410 ± 0.0446** | **0.0058 ± 0.0126** | **0.0523 ± 0.0554** | **0.8825 ± 0.1866** |
| DeepONet   | 0.0475 ± 0.0390      | 0.0064 ± 0.0106      | 0.0612 ± 0.0512      | 0.8669 ± 0.1739      |

### Computational Efficiency

| **Model**  | **Iter/sec ↑** | **Parameters ↓** | **Train Time (min) ↓** | **GPU Memory (GB) ↓** |
|------------|----------------|------------------|------------------------|-----------------------|
| FNO        | 0.65           | 1.42B            | **57.1**               | 40.71                 |
| F-FNO      | 0.52           | 6.95M            | 244.7                  | 49.98                 |
| MG-FNO     | 0.08           | 353.95M          | 180.5                  | 7.12                  |
| **DeepONet** | **3.83**     | **2.09M**        | 60.9                   | **4.11**              |

### Key Findings

- **DeepONet** achieved the highest accuracy on real displacement fields (MSE = 0.0039, 90.0% accuracy) with the fastest inference (3.83 it/s) and fewest parameters (2.09M)
- **MG-FNO** achieved the best performance on imaginary fields (MSE = 0.0058, 88.3% accuracy) with the lowest GPU memory among FNO variants (7.12 GB)
- **F-FNO** offered over 200× parameter reduction vs. baseline FNO but did not improve accuracy in the moderate-resolution setting
- **No single architecture dominates** — selection depends on the relative importance of accuracy, spatial fidelity, and compute budget

---

## Architecture Overview

### DeepONet — Branch Decomposition Fusion
Branch-trunk architecture with **six branch networks** (one CNN for T1 MRI, five FNNs for scalar features: scan direction, vibration frequency, sex, brain volume, age) and one trunk network for spatial coordinates. Multimodal fusion via **elementwise multiplication** of branch embeddings. Best accuracy on real fields with the most compact footprint (2.09M parameters).

### Multi-Grid FNO (MG-FNO) — Field Projection Fusion
Partitions the input domain into non-overlapping patches [20, 20, 22], each augmented with downsampled global T1 context to capture both fine-grained local features and long-range anatomical context. Best performance on imaginary fields.

### Factorized FNO (F-FNO) — Field Projection Fusion
Decomposes 3D spectral convolution into three separate 1D transforms along each spatial axis, reducing parameter count by ~200× relative to the baseline FNO.

### Standard FNO — Field Projection Fusion
Applies joint 3D spectral convolution in the Fourier domain over the full volume. Scalar metadata is broadcast onto spatial grids at the input, enforcing uniform spatial conditioning from the first layer.

---

## Dataset

### Brain Biomechanics Imaging Repository (BBIR)

- **Source:** Washington University in St. Louis (WUSTL) MRE dataset
- **Subjects:** 52 healthy adults (14–80 years, balanced sex representation)
- **Total samples:** 249 (after preprocessing)
- **Modalities:** T1-weighted MRI, MRE displacement fields, demographic data
- **Frequency range:** 20–90 Hz mechanical excitations
- **Resolution:** 3mm isotropic voxel resolution
- **Subject-level split** (prevents data leakage):
  - Training: 36 subjects (171 samples)
  - Validation: 5 subjects (25 samples)
  - Testing: 11 subjects (53 samples)

📂 **Data Source:** [https://www.nitrc.org/projects/bbir](https://www.nitrc.org/projects/bbir)

### Data Processing Pipeline

- **Normalization:** Per-voxel min-max scaling for T1 volumes; global min-max scaling for displacement fields
- **Brain masking:** Binary masks constrain loss and evaluation to anatomically relevant regions
- **Demographic encoding:** Age, sex, brain volume, scan direction, excitation frequency
- **Positional encoding:** 3D spatial coordinates [x, y, z] provide location-aware features

<img width="900" height="600" alt="Dataset visualization" src="https://github.com/user-attachments/assets/62b833bc-d4f3-47e7-9812-89d02805e4f0" />

### Dataset Citation

```bibtex
@article{bayly2021mr,
  title={MR imaging of human brain mechanics in vivo: new measurements to facilitate the development of computational models of brain injury},
  author={Bayly, Philip V and Alshareef, Ahmed and Knutsen, Andrew K and Upadhyay, Kshitiz and Okamoto, Ruth J and Carass, Aaron and Butman, John A and Pham, Dzung L and Prince, Jerry L and Ramesh, KT and others},
  journal={Annals of Biomedical Engineering},
  volume={49},
  number={10},
  pages={2677--2692},
  year={2021},
  publisher={Springer}
}
```

---

## Installation

```bash
git clone https://github.com/Centrum-IntelliPhysics/Neural-Operator-for-Traumatic-Brain-Injury.git
cd Neural-Operator-for-Traumatic-Brain-Injury
pip install -r requirements.txt
```

---

## Citation

If you use this code or find this work useful for your research, please cite our paper:

```bibtex
@article{AGARWAL2026109398,
  title   = {Multimodal neural operators for real-time biomechanical modelling of traumatic brain injury},
  journal = {Computer Methods and Programs in Biomedicine},
  pages   = {109398},
  year    = {2026},
  issn    = {0169-2607},
  doi     = {10.1016/j.cmpb.2026.109398},
  url     = {https://www.sciencedirect.com/science/article/pii/S0169260726001586},
  author  = {Anusha Agarwal and Dibakar Roy Sarkar and Somdatta Goswami},
  keywords = {Multimodal neural operators, Heterogeneous data fusion, Fourier Neural Operator (FNO), Deep Operator Network (DeepONet), Traumatic brain injury, Magnetic Resonance Elastography (MRE), Patient-specific modeling, Real-time prediction, Digital twins}
}
```

---

## Acknowledgments

The authors acknowledge computing support from the Advanced Research Computing at Hopkins (ARCH) core facility at Johns Hopkins University and the Rockfish cluster, supported by NSF grant OAC1920103. The research efforts of DRS and SG are supported by NSF Grant No. 2436738. The Brain Biomechanics Imaging Repository is supported under grants U01 NS112120 and R56 NS055951.
