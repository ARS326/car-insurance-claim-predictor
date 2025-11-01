# Machine Learning-Based Car Insurance Claim Prediction App

This repository contains a **Streamlit** web application for predicting whether a car insurance claim will be made, based on customer and policy details.  
The predictive model is trained using **machine learning techniques** for classification.
It also contains the main model as **pipeline.pkl**, a file **main.py** to run the prediction in python ternimal, and a jupyter notebook **modelling_master.ipynb** that contains all the data representations and model training codes. 

You can access the web-app by clickinh [here](https://car-ins-claim-pred.streamlit.app/).

---

##  Business Problem

Insurance companies often face the challenge of estimating the likelihood of a customer making a claim.  
This project develops a **predictive model** to classify whether a customer is likely to make an insurance claim, using demographic and policy-related features.

---

##  Dataset

The dataset includes features such as:

- Demographics (e.g., age, location, etc.)
- Policy and vehicle information
- Other relevant risk factors

The **target variable** indicates whether a claim was made (`Yes` / `No`).

---

##  Data Preprocessing

1. Handled missing values  
2. Encoded categorical features using one-hot encoding  
3. Scaled numerical variables for uniformity  
4. Created a preprocessing pipeline and saved it with the model

---

##  Exploratory Data Analysis (EDA)

- Analyzed feature distributions  
- Studied relationships between customer behavior and claim outcomes  
- Visualized data using **Plotly**  

---

##  Model Development

- Trained multiple classification models such as **Logistic Regression**, **Random Forest**, and **XGBoost**
- Evaluated using metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **ROC-AUC**

After experimentation, the best-performing model was selected and saved as `pipeline.pkl`.

---

##  Deployment

The trained pipeline (model + preprocessing) is integrated into a **Streamlit** app for real-time predictions.


### To run locally:
```bash
# Clone the repository
git clone https://github.com/ARS326/car-insurance-claim-predictor.git
cd car-insurance-claim-predictor

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
