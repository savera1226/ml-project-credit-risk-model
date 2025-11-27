import streamlit as st
import prediction_helper as helper
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# ---------------------------------------------------------------------
# CONFIGURATION & ASSETS
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Lauki Finance AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None


# Load Assets (Modern AI/Finance Animations)
lottie_credit = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_5njp3vgg.json")
lottie_success = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_s2lryxtd.json")

# ---------------------------------------------------------------------
# CUSTOM CSS (The "Sundar Pichai" Polish)
# ---------------------------------------------------------------------
st.markdown("""
    <style>
    /* Google Font: Roboto/Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark Modern Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #374151;
    }

    /* Input Fields styling */
    .stNumberInput, .stSelectbox {
        color: white;
    }

    /* Card-like visuals for metrics */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Header Styling */
    h1, h2, h3 {
        background: -webkit-linear-gradient(0deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SIDEBAR: INPUT DATA
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Applicant Data")

    # Group 1: Personal
    with st.expander("👤 Personal Details", expanded=True):
        age = st.number_input('Age', min_value=18, max_value=100, value=28)
        income = st.number_input('Annual Income (₹)', min_value=0, value=1200000, step=50000)
        residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])

    # Group 2: Loan Details
    with st.expander("💰 Loan Details", expanded=True):
        loan_amount = st.number_input('Loan Amount (₹)', min_value=0, value=2560000, step=50000)
        loan_tenure_months = st.number_input('Tenure (Months)', min_value=0, value=36)
        loan_purpose = st.selectbox('Purpose', ['Education', 'Home', 'Auto', 'Personal'])
        loan_type = st.selectbox('Type', ['Unsecured', 'Secured'])

    # Group 3: History
    with st.expander("📉 Credit History", expanded=True):
        avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20)
        delinquency_ratio = st.number_input('Delinquency Ratio (%)', 0, 100, 30)
        credit_utilization_ratio = st.number_input('Credit Utilization (%)', 0, 100, 30)
        # Allow up to 50 accounts. The 'value' defaults to 2.
        num_open_accounts = st.number_input('Open Accounts', min_value=1, max_value=50, value=2)

    # Calculate LTI dynamically for display
    loan_to_income = loan_amount / income if income > 0 else 0
    st.info(f"📊 **Loan-to-Income Ratio:** {loan_to_income:.2f}")

    btn_calculate = st.button("🚀 Run Risk Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Lauki Finance AI")
    st.markdown("### Next-Gen Credit Scoring System")
    st.markdown("Powered by *Machine Learning* & *Predictive Analytics*")

with col_head2:
    if lottie_credit:
        st_lottie(lottie_credit, height=150, key="header_anim")

st.markdown("---")

if btn_calculate:
    # 1. RUN PREDICTION
    try:
        probability, credit_score, rating = helper.predict(
            age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type
        )

        # 2. DETERMINE STATUS COLOR
        if rating == "Poor":
            color = "#ef4444"  # Red
            status_icon = "🛑"
        elif rating == "Average":
            color = "#eab308"  # Yellow
            status_icon = "⚠️"
        elif rating == "Good":
            color = "#22c55e"  # Green
            status_icon = "✅"
        else:  # Excellent
            color = "#3b82f6"  # Blue
            status_icon = "💎"

        # 3. LAYOUT RESULTS
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🎯 Credit Score Analysis")

            # GAUGE CHART (Plotly)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=credit_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{status_icon} {rating}", 'font': {'size': 24, 'color': color}},
                delta={'reference': 600, 'increasing': {'color': "#22c55e"}},
                gauge={
                    'axis': {'range': [300, 900], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [300, 500], 'color': '#500000'},
                        {'range': [500, 650], 'color': '#4a3800'},
                        {'range': [650, 750], 'color': '#003300'},
                        {'range': [750, 900], 'color': '#001a4d'}
                    ],
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"},
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### ⚠️ Risk Assessment")

            # ANIMATED METRIC CARDS
            st.metric(label="Default Probability", value=f"{probability:.2%}",
                      delta=f"{(1 - probability):.2%} Safety Score")
            st.metric(label="Credit Rating Tier", value=rating)
            st.metric(label="Projected Interest Rate", value=f"{5 + (probability * 20):.2f}%")

            # PROGRESS BAR FOR PROBABILITY
            st.markdown("Risk Factor")
            st.progress(min(probability, 1.0))

            if probability > 0.5:
                st.error("High Risk Application: Rejection Recommended")
            else:
                st.success("Low Risk Application: Approval Recommended")

    except Exception as e:
        st.error(f"Error in calculation: {e}")

else:
    st.info("👈 Enter applicant details in the sidebar to generate the credit report.")