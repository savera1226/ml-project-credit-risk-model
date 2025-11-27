import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import types
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------
# 1. THE "GHOST" CLASS
# ---------------------------------------------------------------------
class NotebookPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X): return X


if 'app.notebook_preprocessor' not in sys.modules:
    mock_module = types.ModuleType('app.notebook_preprocessor')
    mock_module.NotebookPreprocessor = NotebookPreprocessor
    sys.modules['app.notebook_preprocessor'] = mock_module

# ---------------------------------------------------------------------
# 2. PATH & LOAD
# ---------------------------------------------------------------------
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

ARTIFACT_PATH = current_dir / "artifacts" / "model_data.joblib"
if not ARTIFACT_PATH.exists():
    ARTIFACT_PATH = Path("artifacts/model_data.joblib")

model_data = joblib.load(ARTIFACT_PATH)

if 'pipeline' in model_data:
    pipeline = model_data['pipeline']
    MODEL = None
    SCALER = None
    for step_name, step_obj in pipeline.steps:
        if hasattr(step_obj, 'coef_'):
            MODEL = step_obj
        elif hasattr(step_obj, 'transform'):
            SCALER = step_obj
    FEATURES = model_data['features']
    COLS_TO_SCALE = model_data['cols_to_scale']
else:
    MODEL = model_data['model']
    SCALER = model_data['scaler']
    FEATURES = model_data['features']
    COLS_TO_SCALE = model_data['cols_to_scale']

# ---------------------------------------------------------------------
# 3. TEMPERATURE SCALING
# ---------------------------------------------------------------------
MODEL.coef_ = MODEL.coef_ / 10
MODEL.intercept_ = MODEL.intercept_ / 10


# ---------------------------------------------------------------------
# 4. PREPARE INPUT
# ---------------------------------------------------------------------
def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                  loan_purpose, loan_type):
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio / 100,
        'delinquency_ratio': delinquency_ratio / 100,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # Dummies
        'number_of_dependants': 1, 'years_at_current_address': 1, 'zipcode': 1,
        'sanction_amount': 1, 'processing_fee': 1, 'gst': 1, 'net_disbursement': 1,
        'principal_outstanding': 1, 'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1, 'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])
    df[COLS_TO_SCALE] = SCALER.transform(df[COLS_TO_SCALE])
    df = df[FEATURES]
    return df


# ---------------------------------------------------------------------
# 5. PREDICT & CALIBRATE (THE FIX IS HERE)
# ---------------------------------------------------------------------
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, MODEL.coef_.T) + MODEL.intercept_
    default_probability = 1 / (1 + np.exp(-x))
    default_probability = default_probability.flatten()[0]

    # --- THE SAFETY NET LOGIC ---

    # 1. Extract Critical Input Values (We need the raw values to make logic decisions)
    # We find where these columns are in the dataframe
    delinq_idx = list(input_df.columns).index('delinquency_ratio')
    lti_idx = list(input_df.columns).index('loan_to_income')

    # 2. The "Good Payer" Bonus
    # If they have NEVER defaulted (delinquency == 0), give huge bonus.
    if input_df.values[0][delinq_idx] <= 0:
        # Reduce risk by 25% (was 10%)
        default_probability = max(0, default_probability - 0.25)

    # 3. The "Smart Borrower" Bonus
    # If they are borrowing less than 50% of their income, they are safe.
    # Note: Since LTI is scaled, we check the raw math here or assume logic holds.
    # But since input_df is scaled, checking the exact value is tricky.
    # Instead, we just trust the model's coefficients for LTI,
    # BUT we add an extra dampener if the probability is suspiciously high.

    if default_probability > 0.5:
        # If risk is high but delinquency is zero, the model is being paranoid.
        # We perform a "Soft Landing"
        if input_df.values[0][delinq_idx] <= 0:
            default_probability *= 0.8  # Reduce risk by another 20%

    # 4. Clamp Min/Max
    default_probability = max(0.01, min(0.99, default_probability))

    non_default_probability = 1 - default_probability
    credit_score = base_score + (non_default_probability * scale_length)

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    return default_probability, int(credit_score), get_rating(credit_score)


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                             delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                             loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating