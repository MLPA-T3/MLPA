# Synthetic Fraud Detection with Logistic Regression


This document demonstrates how to:

1. **Generate** a synthetic dataset simulating mobile money transactions (with about 5% fraud).  
2. **Train** a logistic regression model to classify transactions as fraudulent (`isFraud=1`) or not (`isFraud=0`).  
3. **Evaluate** the model’s performance on a test set.  
4. **Predict** the probability of fraud for a new transaction using custom feature values.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Imports](#imports)  
3. [Data Creation](#data-creation)  
4. [Feature Engineering](#feature-engineering)  
5. [Train/Test Split](#traintest-split)  
6. [Logistic Regression Model](#logistic-regression-model)  
7. [Evaluate the Model](#evaluate-the-model)  
8. [Predict Fraud Probability](#predict-fraud-probability)  
9. [Conclusion](#conclusion)

---

## Introduction

In this example, we **simulate** transactions with numeric and categorical features, introduce a small fraction of fraud (~5%), and use a **logistic regression** classifier to detect fraud. The data and labels are entirely **synthetic**, making it safe for demonstrations and educational purposes.

---

## Imports

    # We import necessary libraries for data manipulation, modeling, and evaluation.

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

---

## Data Creation

    # Set a reproducible random seed
    np.random.seed(42)

    # Parameters
    N = 2000            # Number of transactions
    fraud_ratio = 0.05  # About 5% fraud

    # STEP: Random hourly increments (0 to 744 ~ 31 days)
    step = np.random.randint(0, 744, size=N)

    # TYPE: Categorical distribution with probabilities
    transaction_types = ["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT"]
    type_col = np.random.choice(
        transaction_types, size=N, 
        p=[0.2, 0.3, 0.3, 0.15, 0.05]
    )

    # AMOUNT: Mostly small-to-medium from a lognormal distribution
    amount = np.random.lognormal(mean=3, sigma=1, size=N)

    # Scale up 10% of them to represent large transactions
    large_idx = np.random.choice(N, size=int(0.1 * N), replace=False)
    amount[large_idx] *= 50

    # oldbalanceOrg: random lognormal, often larger than 'amount'
    oldbalanceOrg = np.random.lognormal(mean=4, sigma=1.2, size=N) * 100
    oldbalanceOrg[large_idx] *= 10

    # newbalanceOrg = oldbalanceOrg - amount (capped at zero)
    newbalanceOrg = oldbalanceOrg - amount
    newbalanceOrg = np.where(newbalanceOrg < 0, 0, newbalanceOrg)

    # oldbalanceDest, newbalanceDest: random or partial simulation
    oldbalanceDest = np.random.lognormal(mean=4, sigma=1.2, size=N) * 50
    newbalanceDest = oldbalanceDest + amount

    # isFraud: label ~5% as fraud, more likely if amount is large
    isFraud = np.zeros(N, dtype=int)
    fraud_indices = np.random.choice(N, size=int(fraud_ratio * N), replace=False)
    for idx in fraud_indices:
        # If transaction is in top 25% of amounts, label as fraud
        if amount[idx] > np.percentile(amount, 75):
            isFraud[idx] = 1
        else:
            # 50% chance if not large but chosen
            isFraud[idx] = np.random.choice([0, 1], p=[0.5, 0.5])

    # Construct a DataFrame
    data = pd.DataFrame({
        "step": step,
        "type": type_col,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrg": newbalanceOrg,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFraud": isFraud
    })

    print("Synthetic dataset sample:")
    print(data.head())

---

## Feature Engineering

    # We'll one-hot encode the 'type' column, dropping the first category to avoid the dummy trap.
    df_encoded = pd.get_dummies(data, columns=["type"], drop_first=True)

    # Separate features (X) and target (y)
    X = df_encoded.drop("isFraud", axis=1)
    y = df_encoded["isFraud"]

---

## Train/Test Split

    # We split into 70% train and 30% test sets, stratifying by y to preserve fraud ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

---

## Logistic Regression Model

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

---

## Evaluate the Model

    y_pred = model.predict(X_test)

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))

---

## Predict Fraud Probability

    # Example new transaction
    new_transaction = {
        "step": 100,
        "type": "CASH_OUT",
        "amount": 2000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrg": 3000.0,
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 12000.0
    }

    # Convert to DataFrame and encode 'type' as before
    new_df = pd.DataFrame([new_transaction])
    new_df_encoded = pd.get_dummies(new_df, columns=["type"], drop_first=True)

    # Ensure all columns match training set columns
    for col in X_train.columns:
        if col not in new_df_encoded.columns:
            new_df_encoded[col] = 0

    # Reorder columns
    new_df_encoded = new_df_encoded[X_train.columns]

    # Get probability of fraud (second column is P(class=1))
    prob_fraud = model.predict_proba(new_df_encoded)[:, 1]
    print(f"\nProbability of fraud for the new transaction: {prob_fraud[0]:.4f}")

---


