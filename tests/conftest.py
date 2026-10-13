import pytest
from helpers import BASE_PROFILE, hashing_embedder, make_holdout

from churn.advisor.playbook import load_playbook
from churn.advisor.retriever import Retriever


@pytest.fixture
def scaler_params():
    return {
        "tenure": {"mean": 32.0, "std": 24.0},
        "MonthlyCharges": {"mean": 65.0, "std": 30.0},
        "TotalCharges": {"mean": 2300.0, "std": 2275.0},
    }


@pytest.fixture
def profile():
    return dict(BASE_PROFILE)


@pytest.fixture
def entries():
    return load_playbook()


@pytest.fixture
def retriever(entries):
    return Retriever(entries, hashing_embedder())


@pytest.fixture
def holdout():
    return make_holdout()


@pytest.fixture
def roi_params():
    return {"retention_rate": 0.30, "offer_cost": 80}
