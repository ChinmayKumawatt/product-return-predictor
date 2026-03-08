# Customer Intelligence Platform

An end-to-end machine learning system that predicts product returns, identifies high value customers, and segments customers based on purchasing behavior.

This project demonstrates how to design a production-style machine learning pipeline with modular components, automated training workflows, experiment tracking, and a deployable web interface.

---

## Project Overview

Retail businesses often struggle with:

- Unexpected product returns
- Identifying high value customers
- Understanding customer segments

This platform solves these problems using three machine learning models integrated into a single system.

Users can interact with the models through a web interface to:

- Predict return risk of an order
- Detect high value customers
- Perform customer segmentation

---

## Features

### Return Prediction

Predicts whether an order is likely to be returned.

Input Features:

- Quantity
- Price
- Order Value
- Invoice Month
- Invoice Day
- Country

The system preprocesses these inputs and predicts the likelihood of a return.

---

### High Value Customer Detection

Identifies whether a customer belongs to the top revenue segment.

Uses RFM features:

- Recency
- Frequency
- Monetary
- Cluster

The cluster is automatically generated using the segmentation model.

---

### Customer Segmentation

Segments customers using K-Means clustering.

Segments represent behavioral patterns such as:

- VIP Customers
- Regular Customers
- At Risk Customers

---

## What Makes This Project Unique

Many ML projects stop at model training.  
This project implements a complete ML system architecture.

### Modular ML Pipelines

The system is broken into reusable components:

Data Ingestion  
Data Transformation  
Model Training  
Prediction Pipeline  
Web Application  

Each stage is independent and reusable.

---

### Automated Training Pipeline

The training workflow is automated through a pipeline.

Training Pipeline  
→ Data Ingestion  
→ Feature Engineering  
→ Model Training + Hyperparameter Tuning  
→ Model Artifacts  

---

### Multiple ML Models

The system trains and evaluates multiple algorithms:

- Logistic Regression
- Random Forest
- XGBoost

Hyperparameter tuning is performed using GridSearchCV.

The best model is selected using F1 Score.

---

### Feature Engineering with RFM Analysis

Customer behavior is captured using RFM metrics:

Recency – Time since last purchase  
Frequency – Number of purchases  
Monetary – Total spending  

These features power both segmentation and high value prediction.

---

### Intelligent Model Dependency

The system automatically connects models.

RFM Input  
→ Segmentation Model  
→ Cluster  
→ High Value Prediction  

This mirrors real production ML workflows.

---

### Experiment Tracking

Model performance reports are automatically stored.

model_reports/

return_model_report.json  
high_value_model_report.json  

Each report contains:

- Accuracy
- Precision
- Recall
- F1 Score

---

## Project Structure

customer-intelligence-platform/

app.py

artifacts/  
 return_model.pkl  
 high_value_model.pkl  
 segmentation_model.pkl  
 preprocessor.pkl  
 rfm_scaler.pkl  

model_reports/

src/

 components/  
  data_ingestion.py  
  data_transformation.py  
  model_trainer.py  

 pipeline/  
  training_pipeline.py  
  predict_pipeline.py  

 utils.py  
 logger.py  
 exception.py  

templates/

---

## Machine Learning Pipeline

Raw Dataset  
→ Data Ingestion  
→ Feature Engineering  
→ Data Transformation  
→ Model Training  
→ Model Artifacts  
→ Prediction Pipeline  
→ Flask Web Application  

---

## Technologies Used

Machine Learning

- Scikit-Learn
- XGBoost
- K-Means Clustering
- GridSearchCV

Data Processing

- Pandas
- NumPy

Backend

- Python
- Flask

Frontend

- HTML
- CSS

---

## How to Run the Project

Clone the repository

git clone https://github.com/YOUR_USERNAME/customer-intelligence-platform.git

---

Install dependencies

pip install -r requirements.txt

---

Train models

python -m src.pipeline.training_pipeline

---

Run the application

python app.py

---

Open in browser

http://localhost:5000

---

## Application Services

The web interface provides three services.

Return Prediction  
Predict if an order is likely to be returned.

High Value Customer Detection  
Classify customers as High Value or Regular Customer.

Customer Segmentation  
Segment customers based on purchasing patterns.

---

## Example Predictions

Example input and predictions.

Recency=5 Frequency=30 Monetary=5000 → High Value Customer  
Recency=150 Frequency=1 Monetary=20 → Regular Customer  
Quantity=5 Price=20 OrderValue=100 → Not Likely to Return  

---

## Future Improvements

Possible extensions:

- Model explainability with SHAP
- Real-time prediction APIs
- Dashboard analytics
- Automated retraining pipelines
- Cloud deployment

---

## Author

Chinmay Kumawat

Machine Learning & Data Science Enthusiast  
Focused on building end-to-end ML systems and production pipelines.

---

## License

This project is released under the MIT License.
