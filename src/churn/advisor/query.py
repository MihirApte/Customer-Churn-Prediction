"""Turn a customer's top SHAP drivers into plain-English search sentences.

A SHAP driver such as "Contract_Two year" only says which column mattered. What it means
for THIS customer depends on their actual value, and the playbook only has advice for
particular values (month-to-month contracts, high bills, fibre, and so on). So every
describer returns text ONLY when the customer's value is the one the playbook addresses,
and returns None otherwise. A driver that returns None is skipped and reported as having
no playbook guidance, so an entry can never be retrieved for a customer it does not fit
(for example the month-to-month offer for a customer already on a one-year contract).

The wording matches the playbook ("month-to-month", "electronic check", "new customer"),
which is what lets the embedding match work.
"""

# Features the model uses but the advisor must not act on or search with.
# Gender is a protected characteristic: it stays in the model but never drives advice.
NON_ACTIONABLE = frozenset({"gender_Male"})

HIGH_MONTHLY_CHARGE = 80    # EUR: playbook entries on bills apply from here
LOW_TOTAL_CHARGES = 1000    # EUR: below this the customer has only recently joined


def base_field(feature: str) -> str:
    """'Contract_Two year' -> 'Contract'; 'tenure' -> 'tenure'."""
    return feature.split("_")[0]


def _tenure(p: dict):
    t = p["tenure"]
    if t <= 6:
        return f"new customer with a very short tenure of {t} months"
    if t <= 12:
        return f"short tenure of {t} months, close to the first anniversary"
    if t <= 24:
        return f"moderate tenure of {t} months"
    return f"long tenure of {t} months"


def _monthly(p: dict):
    m = p["MonthlyCharges"]
    return f"high monthly charge of EUR {m:.0f}" if m >= HIGH_MONTHLY_CHARGE else None


def _total(p: dict):
    t = p["TotalCharges"]
    if t < LOW_TOTAL_CHARGES:
        return f"low total charges of EUR {t:,.0f} so far, a recently joined customer"
    return None


def _senior(p: dict):
    return "senior citizen" if p["SeniorCitizen"] == 1 else None


def _when(field: str, value: str, text: str):
    """Describe the customer only if `field` equals `value`."""
    return lambda p: text if p[field] == value else None


def _addon(field: str, name: str):
    """Only a customer who lacks the service (and has internet) is described."""
    return lambda p: f"has no {name}" if p[field] == "No" else None


def _contract(p: dict):
    if p["Contract"] == "Month-to-month":
        return "month-to-month contract, not on a one-year or two-year plan"
    return None


def _payment(p: dict):
    return {
        "Electronic check": "pays by electronic check, a manual payment method, not automatic payment",
        "Mailed check": "pays by mailed check, a manual payment method, not automatic payment",
    }.get(p["PaymentMethod"])


DESCRIBERS = {
    "tenure": _tenure,
    "MonthlyCharges": _monthly,
    "TotalCharges": _total,
    "SeniorCitizen": _senior,
    "Partner": _when("Partner", "No", "has no partner"),
    "Dependents": _when("Dependents", "No", "has no dependents"),
    "PhoneService": _when("PhoneService", "No", "has no phone service"),
    "MultipleLines": _when("MultipleLines", "Yes", "has multiple phone lines on the account"),
    "InternetService": _when("InternetService", "Fiber optic", "fibre optic internet service"),
    "OnlineSecurity": _addon("OnlineSecurity", "online security service"),
    "OnlineBackup": _addon("OnlineBackup", "online backup service"),
    "DeviceProtection": _addon("DeviceProtection", "device protection service"),
    "TechSupport": _addon("TechSupport", "tech support service"),
    "StreamingTV": _when("StreamingTV", "Yes", "uses streaming TV"),
    "StreamingMovies": _when("StreamingMovies", "Yes", "uses streaming movies"),
    "Contract": _contract,
    "PaperlessBilling": _when("PaperlessBilling", "Yes", "paperless billing"),
    "PaymentMethod": _payment,
}


def describe_driver(feature: str, profile: dict):
    """Plain-English description of one driver for this customer, or None if it
    should not be used (non-actionable, or the customer's value is not one the
    playbook addresses)."""
    if feature in NON_ACTIONABLE:
        return None
    base = base_field(feature)
    if base == "PaymentMethod":
        # Entries target one specific method, so the driver must be the customer's own method.
        # "Mailed check = 0" on an electronic-check customer says nothing the playbook can act on.
        if profile["PaymentMethod"] != feature.split("_", 1)[1]:
            return None
    describer = DESCRIBERS.get(base)
    return describer(profile) if describer else None


def applicable_families(profile: dict) -> set:
    """Feature families (tenure, Contract, ...) whose playbook-relevant value this customer has."""
    return {base for base, describer in DESCRIBERS.items() if describer(profile) is not None}


def driver_descriptions(drivers, profile: dict) -> list:
    """(feature, description) for each usable driver, in the order given.

    `drivers` is any sequence of objects with a `.feature` attribute (the Driver
    tuples from churn.model). Drivers that map to the same profile field, such as
    Contract_One year and Contract_Two year, are described only once, under the
    first feature name.
    """
    parts, seen = [], set()
    for driver in drivers:
        base = base_field(driver.feature)
        if base in seen:
            continue
        text = describe_driver(driver.feature, profile)
        if text:
            seen.add(base)
            parts.append((driver.feature, text[0].upper() + text[1:]))
    return parts


def skipped_drivers(drivers, profile: dict) -> list:
    """Drivers with nothing the playbook can say for this customer, by feature name.

    Gender is never listed (it is deliberately ignored, not "uncovered")."""
    usable_bases = {base_field(f) for f, _ in driver_descriptions(drivers, profile)}
    return [
        d.feature for d in drivers
        if d.feature not in NON_ACTIONABLE and base_field(d.feature) not in usable_bases
    ]


FRIENDLY_NAMES = {
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charge",
    "TotalCharges": "Total charges",
    "SeniorCitizen": "Senior citizen status",
    "Partner": "Partner status",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple phone lines",
    "InternetService": "Internet service type",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract type",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}


def friendly_driver_names(features) -> list:
    """Readable, de-duplicated names for display ('PaymentMethod_Credit card (automatic)'
    becomes 'Payment method')."""
    names = []
    for feature in features:
        name = FRIENDLY_NAMES.get(base_field(feature), feature)
        if name not in names:
            names.append(name)
    return names


def build_query(drivers, profile: dict) -> str:
    """Join the descriptions of the top drivers into one search sentence."""
    parts = driver_descriptions(drivers, profile)
    if not parts:
        raise ValueError("No usable churn drivers to build a query from")
    return ". ".join(text for _, text in parts) + "."
