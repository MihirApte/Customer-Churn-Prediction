"""Example customers shared by the try_*.py scripts.

Monthly charges are set to be consistent with the services each customer has. In this
dataset the charge is driven by the services: fibre customers with no add-ons pay about
EUR 68-72, and every fibre customer paying 90 or more has at least one add-on. A profile
that breaks this (for example fibre, no add-ons, EUR 95) never occurs in the real data, so
the model's score for it is not meaningful.
"""

BASE = {
    "SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 64.0, "TotalCharges": 768.0,
    "gender": "Male", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
}

EXAMPLES = {
    "New fibre customer, month-to-month, pays by electronic check": {
        "tenure": 3, "MonthlyCharges": 70.0, "TotalCharges": 210.0,
        "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
    },
    "Long-standing month-to-month DSL customer": {
        "tenure": 48, "MonthlyCharges": 55.0, "TotalCharges": 2640.0,
        "PaymentMethod": "Mailed check", "TechSupport": "Yes", "OnlineSecurity": "Yes",
    },
    "Senior, fibre, streaming TV and movies, one-year contract": {
        "SeniorCitizen": 1, "tenure": 20, "MonthlyCharges": 95.0, "TotalCharges": 1900.0,
        "InternetService": "Fiber optic", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "One year", "PaymentMethod": "Credit card (automatic)",
    },
    "New customer, DSL with multiple lines, paperless, electronic check": {
        "tenure": 8, "MonthlyCharges": 58.0, "TotalCharges": 464.0,
        "MultipleLines": "Yes", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    },
    "Fibre, streaming and add-ons, month-to-month, electronic check": {
        "tenure": 5, "MonthlyCharges": 98.0, "TotalCharges": 490.0,
        "InternetService": "Fiber optic", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "OnlineBackup": "Yes", "PaymentMethod": "Electronic check", "PaperlessBilling": "Yes",
    },
}
