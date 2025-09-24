# Real-Time Brain Biomechanics Prediction with Neural Operators: Toward Clinically Deployable Traumatic Brain Injury Models

**Authors:**  
Anusha Agarwal, [Dibakar Roy Sarkar](https://scholar.google.com/citations?user=Sz4nHdYAAAAJ&hl=en), and [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en)
 
---

## Abstract

We benchmark various neural operator frameworks for traumatic brain injury (TBI) modeling, addressing the critical need for rapid, personalized brain biomechanics predictions. Our study evaluates four state-of-the-art architectures: DeepONet, Fourier Neural Operator (FNO), Factorized Fourier Neural Operator (F-FNO), and Multi-Grid Fourier Neural Operator (MG-FNO) on predicting brain displacement fields from MRI-derived anatomical geometry and MRE-derived stiffness maps.

---

## Key Contributions

- **First systematic benchmark** of neural operator variants for TBI biomechanics modeling  
- **Patient-specific prediction framework** mapping MRI geometry and demographic data to brain displacement fields  
- **Comprehensive evaluation** across four neural operator architectures with detailed performance analysis  
- **Real-time capability** reducing computation time from hours to milliseconds while maintaining high accuracy  

---

## Results Summary

Our experiments demonstrate significant computational speedups while preserving predictive accuracy across physiologically relevant frequency ranges (20–90 Hz):

| **Model** | **Real Displacement** | **Imaginary Displacement** | **Iterations/Second** | **Parameters** |
|-----------|----------------------|----------------------------|----------------------|----------------|
| **MG-FNO** | **MSE: 0.0023 (94.3% Acc)** | **MSE: 0.0045 (90.4% Acc)** | 8.08 | 353M |
| F-FNO | MSE: 0.0025 (93.9% Acc) | MSE: 0.0101 (87.5% Acc) | **1.44** | **6.9M** |
| FNO | MSE: 0.0041 (91.2% Acc) | MSE: 0.0430 (89.0% Acc) | 1.98 | 1.4B |
| DeepONet | MSE: 0.0064 (83.5% Acc) | MSE: 0.0068 (92.0% Acc) | **14.54** | **2.1M** |

**Key Findings:**
- **MG-FNO achieved the highest overall accuracy** with 43.9% and 89.5% error reduction versus baseline  
- **F-FNO provided the best parameter efficiency** with 203.8× fewer parameters than traditional FNO  
- **DeepONet offered fastest inference** but reduced fidelity in high-gradient regions  
- **All models showed strong dependence on spatial information** such as voxel position  

---

## Architecture Overview

### Multi-Grid FNO (MG-FNO)
Best-performing architecture. Partitions the input domain into non-overlapping patches [20, 20, 22] with global context integration, enabling both local fine-grained and global long-range dependency modeling.  

### Factorized FNO (F-FNO)
Factorizes spectral convolution into one-dimensional operations along each spatial axis, reducing computational cost by over two orders of magnitude while retaining accuracy.  

### Standard FNO
Uses spectral convolution layers in the Fourier domain to capture global dependencies with resolution-invariant learning.  

### DeepONet
Employs branch-trunk architecture with separate networks for anatomical features and spatial coordinates, enabling efficient functional mapping.  

---

## Dataset

Full Dataset is available [here](http://www.nitrc.org/projects/bbir). The preprocessed data used in the paper is available [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/droysar1_jh_edu/EgexIiglwpJNuC1-f5aZlM4BDI6e7GXewHoi7d9VBfA6hQ?e=j1pZ7q)

### Brain Biomechanics Imaging Repository
- **Source**: Washington University in St. Louis (WUSTL) MRE dataset  
- **Subjects**: 52 healthy adults (14–80 years, balanced sex representation)  
- **Total samples**: 249 (after preprocessing)  
- **Modalities**: T1-weighted MRI, MRE displacement fields, demographic data  
- **Frequency range**: 20–90 Hz mechanical excitations  
- **Resolution**: 3mm isotropic voxel resolution  
- **Split**: 70% training, 10% validation, 20% testing  

### Data Processing Pipeline
- **Normalization**: Min-max scaling for T1 volumes and displacement fields  
- **Brain masking**: Binary masks to focus on anatomically relevant regions  
- **Demographic encoding**: Age, sex, brain volume, scan direction, frequency  
- **Positional encoding**: 3D spatial coordinates [x, y, z] for spatial awareness  

<img width="900" height="600" alt="Image" src="https://github.com/user-attachments/assets/62b833bc-d4f3-47e7-9812-89d02805e4f0" />

---

## Installation

```bash
git clone https://github.com/[username]/neural-operators-tbi-benchmark.git
cd neural-operators-tbi-benchmark
pip install -r requirements.txt

```

## Citation

If you use this code or dataset for your research, please cite our paper:

```bash
@article{agarwal2025benchmarking,
  title={Benchmarking Neural Operators for Biomechanical Modeling of Traumatic Brain Injury},
  author={Agarwal, Anusha and Sarkar, Dibakar Roy and Goswami, Somdatta},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

