# 🧬 Lauki Finance: Algorithmic Credit Risk Engine

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

### 🚀 **Live Dashboard:** [Click Here to Launch App]https://ml-project-credit-risk-modelgit-rvfr6bvvxuj8rxf3qyqgaj.streamlit.app/

---

## 📖 Executive Summary
**Lauki Finance AI** is a production-grade machine learning system designed to predict loan default probabilities and assign FICO-like credit scores (300-900).

Unlike standard "black box" ML models, this system implements a **Hybrid Decision Engine** that combines:
1.  **Statistical Inference:** A Logistic Regression model trained on historical lending data.
2.  **Business Logic Layer:** A "Safety Net" algorithm that overrides model outputs for edge cases (e.g., high-income earners with reckless credit utilization).
3.  **Score Calibration:** Mathematical transformation of raw log-odds into user-friendly credit tiers.

---

## 🧠 The Engineering Logic
The core challenge was preventing the **"Zero Risk Fallacy"** (where models predict 0% probability) and the **"Binary Trap"** (where scores swing wildly between 300 and 900).

### 1. The Model Architecture
* **Algorithm:** Logistic Regression (Selected for interpretability and regulatory compliance).
* **Key Features:**
    * `Loan-to-Income Ratio (LTI)`: The strongest predictor of default.
    * `Credit Utilization`: Heavily penalized if > 30%.
    * `Delinquency History`: Immediate disqualifier for "Good Behavior" bonuses.

### 2. The "Safety Net" Logic
Raw ML models often fail on nuance. This system injects logic to handle "Grey Zone" applicants:
* **The "Good Payer" Bonus:** If `Delinquency == 0`, risk is mathematically dampened, preventing instant rejection for average borrowers.
* **The "Trap" Detection:** High-income applicants with >90% Credit Utilization are flagged as High Risk, overriding their income advantage.

### 3. Score Calibration Formula
Raw probabilities ($p$) are converted to scores ($S$) using a custom linear mapping:
$$S = 300 + (1 - p) \times 600$$
* *Floor:* 300 (High Risk)
* *Ceiling:* 900 (Low Risk)

---

## 🛠️ Tech Stack & Directory Structure
The project is structured for Cloud Deployment (Streamlit Community Cloud).

```bash
Credit Risk Modelling/
├── app/
│   ├── main.py                # Frontend (Streamlit + Plotly)
│   ├── prediction_helper.py   # Inference Engine (Math & Logic)
│   └── artifacts/             # Serialized Model (.joblib)
├── requirements.txt           # Production Dependencies
├── .gitignore                 # DevOps Configuration
└── README.md                  # Documentation

💻 Local Installation
To run this engine locally:

Clone the repository:

Bash

git clone [https://github.com/savera1226/ml-project-credit-risk-model.git](https://github.com/savera1226/ml-project-credit-risk-model.git)
Install Dependencies:

Bash

pip install -r requirements.txt
Run the Dashboard:

Bash

streamlit run app/main.py
📊 Visuals
The dashboard features:

Real-time Gauge Charts: Visualizing credit health.

Dynamic Risk Assessment: Calculating "Safety Scores" and projected interest rates.

Reactive Inputs: Instant recalculation upon changing financial parameters.

Author: Krish Engineered with Precision & Python.


### STEP 2: PUSH IT AGAIN
Since it vanished, you need to commit it again.

Run this in your PyCharm Terminal:

```bash
git add README.md
git commit -m "Restored Documentation"
git push
