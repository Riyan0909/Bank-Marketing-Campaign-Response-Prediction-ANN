# Bank Marketing Campaign Response Prediction using ANN

An Artificial Neural Network (ANN) based machine learning project that predicts whether a customer is likely to respond positively to a bank marketing campaign.

## Overview

Marketing campaigns generate large amounts of customer data, but not every customer is equally likely to respond. This project applies an Artificial Neural Network to learn patterns from customer and campaign-related features and classify customers into two categories:

* **Yes** — Customer responded positively
* **No** — Customer did not respond

The objective is to demonstrate a complete ANN classification workflow, from data preprocessing and feature encoding to model training and evaluation.

## Key Features

* Data preprocessing and cleaning
* Categorical feature encoding
* Feature scaling
* Train-test data splitting
* Artificial Neural Network implementation
* ReLU activation for hidden layers
* Sigmoid activation for binary classification
* Model training and evaluation
* Accuracy measurement
* Confusion matrix visualization

## Model Architecture

The ANN model consists of the following architecture:

```text
Input Layer
     ↓
Dense Layer — 16 Neurons — ReLU
     ↓
Dense Layer — 8 Neurons — ReLU
     ↓
Output Layer — 1 Neuron — Sigmoid
```

The model is designed for **binary classification**, where the output represents the predicted customer response.

## Technologies & Libraries

* **Python**
* **Pandas** — Data manipulation and preprocessing
* **NumPy** — Numerical operations
* **Scikit-learn** — Data preprocessing and model evaluation
* **TensorFlow / Keras** — Artificial Neural Network implementation
* **Matplotlib** — Data visualization
* **Seaborn** — Confusion matrix visualization

## Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Train-Test Split
     ↓
ANN Model Construction
     ↓
Model Training
     ↓
Prediction
     ↓
Model Evaluation
```

## Results

The trained ANN model achieved an accuracy of approximately:

**89.06%**

### Confusion Matrix

```text
[[773  34]
 [ 65  33]]
```

The model demonstrates a solid classification performance on the test dataset and successfully learns patterns associated with customer responses to the marketing campaign.

## Project Structure

```text
Bank-Marketing-Campaign-Response-Prediction-ANN/
│
├── bank.py
├── bank.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Bank-Marketing-Campaign-Response-Prediction-ANN.git
```

Navigate to the project directory:

```bash
cd Bank-Marketing-Campaign-Response-Prediction-ANN
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python bank.py
```

## Learning Outcomes

Through this project, I implemented and practiced:

* Artificial Neural Networks
* Perceptrons and dense layers
* Forward propagation
* Activation functions
* Binary classification
* Data preprocessing
* Feature encoding
* Feature scaling
* Model training
* Model evaluation
* Confusion matrix analysis

## Future Improvements

Potential improvements for future versions include:

* Hyperparameter tuning
* Early stopping
* Regularization techniques
* Experimenting with different network architectures
* Additional classification metrics
* Model optimization

## Author

**Riyan Qamar**

Computer Science Student | Machine Learning & Artificial Intelligence
