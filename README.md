# 📊 Customer Churn Intelligence Engine

An end-to-end Data Science & Machine Learning application built to analyze customer retention dynamics, visualize churn factors, and predict prospective customer attrition in real time. 

🚀 **Live Interactive App:** [Customer Churn Intelligence Engine](https://customer-churn-intelligence-engine-l5uegzuvgf2mt5vnepqrje.streamlit.app/)

---

## 📌 Project Overview

Customer retention is crucial for business growth and stability. The **Customer Churn Intelligence Engine** empowers data-driven teams to identify high-risk customers, understand underlying behavioral patterns driving churn, and proactively take retention steps.

By leveraging advanced machine learning algorithms alongside dynamic feature engineering and model explainability (SHAP), this app transforms raw customer metrics into actionable strategic insights.

---

## ✨ Key Features

* **Interactive Analytics Dashboard:** Perform Exploratory Data Analysis (EDA) on customer demographics, payment behavior, and usage metrics.
* **Real-time Churn Prediction:** Input custom customer attributes to obtain instant predictions on churn probability.
* **Model Interpretability (SHAP Integration):** Deep-dive into *why* a customer is predicted to churn, providing complete transparency into the algorithm's decisions.
* **Clean & Responsive UI:** Designed with Streamlit for a smooth and intuitive user interface.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Web Framework:** Streamlit
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-Learn, XGBoost, SHAP

---

## 📁 Repository Structure

```text
├── .devcontainer/                 # Dev container setup configuration
├── Churn.ipynb                    # Jupyter Notebook for EDA & model training
├── app.py                         # Main Streamlit web application
├── main.py                        # Core logic / secondary execution script
├── requirements.txt               # Python project dependencies
├── synthetic_customer_churn_100k.csv  # Dataset used for training/analysis
└── xgb_churn_pipeline.pkl         # Pre-trained XGBoost model pipeline               # Dataset directory (if applicable)
```

# 🚀 Local Installation & Setup
If you want to run this project locally, follow these simple steps:

1. ## Clone the repository:

```Bash
git clone [https://github.com/mudassirkhan1249/Customer-Churn-Intelligence-Engine.git](https://github.com/mudassirkhan1249/Customer-Churn-Intelligence-Engine.git)
cd Customer-Churn-Intelligence-Engine
```

2. ## Create a virtual environment (recommended):
```Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. ## Install the dependencies:
```Bash
pip install -r requirements.txt
```

4. ## Run the Streamlit application:

```Bash
streamlit run app.py
```
## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.


# Author
##### Mudassir Khan
##### Data Scientist/ ML Engineer

##### GitHub: [mudassirkhan1249](https://github.com/mudassirkhan1249)
