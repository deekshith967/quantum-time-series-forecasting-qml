# Quantum Time Series Forecasting using Hybrid QML

## Overview

This project implements a hybrid **Quantum-Classical Machine Learning model** for financial time-series forecasting using PennyLane.

The model combines:

* Quantum Autoencoder (feature compression)
* Variational Quantum Circuit (prediction)

Applied on financial datasets like:

* S&P 500
* NIFTY 50
* WTI Oil Prices

---

## Methodology

### 1. Data Preprocessing

* OHLCV features (Open, High, Low, Close, Volume)
* 10-day sliding window
* MinMax normalization

### 2. Quantum Feature Extraction

* Quantum Autoencoder reduces 10 → 4 latent features

### 3. Prediction Model

* Variational Quantum Circuit (VQC)
* Acts as regression model

---

##  Results

Results are based on experimental evaluation from the project report.

| Metric   | Value        |
| -------- | ------------ |
| MSE      | Refer Report |
| MAE      | Refer Report |
| R² Score | Refer Report |

 Detailed results available in:
`docs/project_report.pdf`

---

## Project Structure

```
src/        → Source code
docs/       → Report & diagrams
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python src/main.py --mode train
python src/main.py --mode test
```

---

## Dataset

Datasets are not included due to size.

You can download from:

* Yahoo Finance
* Kaggle

Place datasets inside:

```
datasets/
```

---

## Technologies Used

* PennyLane
* PyTorch
* NumPy
* Pandas
* Scikit-learn

---

## Future Work

* Improve R² score using deeper circuits
* Hybrid classical + quantum ensemble
* Real-time forecasting system

---

##  Author

N,Sai Deekshith B.Tech CSE (AI) (IDD) Student
Interested in Quantum Machine Learning & AI Research
Connect With Me
 LinkedIn: https://www.linkedin.com/in/sai-deekshith-nagireddi-bbab932b1/
 GitHub: https://github.com/deekshith967
 Portfolio: https://portfolio-phi-five-42.vercel.app
