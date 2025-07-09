# TML25_A3_37
Trustworthy Machine Learning — Assignment 3  
**Robustness Against Adversarial Attacks**

---

## Objective

Train a robust image classification model that performs well on both clean and adversarial examples. The adversarial inputs are crafted using:

- **FGSM**: Fast Gradient Sign Method (1-step)
- **PGD**: Projected Gradient Descent (multi-step)

The model must generalize well on unperturbed data while remaining resilient to white-box adversarial perturbations bounded in the ℓ∞-norm.

---

## Methodology

We implemented two curriculum-based adversarial training pipelines using a `resnet18` model, incorporating both PGD-only and FGSM→PGD attack schedules. Exponential Moving Average (EMA) was used to stabilize training, and a clean fine-tuning phase was applied at the end.

### Approach 1: PGD-Only Curriculum Training

- PGD used throughout training with gradually increasing strength.
- 3-phase curriculum:
  - Epochs 0–9: PGD(1-step), ε = 2/255
  - Epochs 10–29: PGD(2-step), ε = 4/255
  - Epochs 30–49: PGD(3-step), ε = 6/255
- Final 10 epochs fine-tune on clean inputs only.

### Approach 2: FGSM→PGD Curriculum Training

- Starts with FGSM for warm-up, then transitions to PGD.
- 4-stage curriculum:
  - Epochs 0–6: FGSM, ε = 8/255
  - Epochs 7–19: PGD(1-step), ε = 2/255
  - Epochs 20–39: PGD(2-step), ε = 4/255
  - Epochs 40–54: PGD(3-step), ε = 6/255
- Includes clean fine-tuning at the end.

---

## Files

| File                          | Description                                |
|-------------------------------|--------------------------------------------|
| `train_pgdonly.py`            | PGD-only curriculum training script        |
| `train_fgsm_pgd.py`           | FGSM→PGD curriculum training script        |
| `Train.pt`                    | Input Dataset                              |
| `robust_curriculum_ema.pt`    | Final model from Approach 1                |
| `fgsm_pgd_curriculum_ema.pt`  | Final model from Approach 2(for submission)|         
| `submit.py`                   | Submission to the evaluation server        |
| `utils.py`                    | Data loading and pre-processing            |

---

## Results

| Approach   | Clean Accuracy (%) | FGSM Accuracy (%) | PGD Accuracy (%) |
|------------|--------------------|-------------------|------------------|
| PGD-Only   | 54.47              | 20.57             | 10.63            |
| FGSM→PGD   | 58.67              | 28.67             | 15.10            |

---

## Key Concepts Used

- Adversarial Training ([1], [2])
- Curriculum Scheduling
- Label Smoothing
- Exponential Moving Average (EMA)
- FGSM and PGD Attack Generation
- Clean Fine-Tuning

---

## References

[1] Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015  
[2] Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018  
[3] Wong et al., *Fast is Better than Free: Revisiting Adversarial Training*, ICLR 2020  
[4] Boenisch & Dziedzic, *Lecture 06: Adversarial Machine Learning*, TML SS2025

---

## How to Run

```bash
# Approach 1
python train_pgdonly.py

# Approach 2
python train_fgsm_pgd.py

## Dependencies

pip install torch torchvision tqdm




