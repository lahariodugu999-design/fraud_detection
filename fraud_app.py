import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Provider Fraud Predictor", layout="centered")

# ---------------------------------------------------------------
# Load trained artifacts (produced by the modelling notebook)
# ---------------------------------------------------------------
model = joblib.load("fraud_model.pkl")
feature_cols = joblib.load("model_features.pkl")
model_name = joblib.load("model_name.pkl")
try:
    scaler = joblib.load("model_scaler.pkl")
except FileNotFoundError:
    scaler = None

st.title("🏥 Healthcare Provider Fraud Predictor")
st.caption(f"Model in use: **{model_name}**")
st.write(
    "Upload a provider's **Inpatient**, **Outpatient** and **Beneficiary** claim CSVs "
    "(same format as the case study data) and this app engineers the same provider-level "
    "features used in training, then predicts fraud risk."
)


# ---------------------------------------------------------------
# Same cleaning / feature-engineering functions used in the notebook
# ---------------------------------------------------------------
def clean_columns(df):
    df.columns = (df.columns.str.strip().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False))
    return df


def convert_dates(df):
    date_cols = [c for c in df.columns if ("Dt" in c) or (c in ["DOB", "DOD"])]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def preprocess_beneficiary(beneficiary):
    beneficiary = clean_columns(beneficiary)
    beneficiary = convert_dates(beneficiary)
    today = pd.Timestamp("2009-12-31")
    beneficiary["Age"] = np.where(
        beneficiary["DOD"].isna(),
        ((today - beneficiary["DOB"]).dt.days / 365.25),
        ((beneficiary["DOD"] - beneficiary["DOB"]).dt.days / 365.25),
    )
    beneficiary["Age"] = beneficiary["Age"].round()
    chronic_cols = [c for c in beneficiary.columns if "ChronicCond" in c]
    for c in chronic_cols:
        beneficiary[c] = beneficiary[c].replace({2: 0})
    beneficiary["TotalChronicDiseases"] = beneficiary[chronic_cols].sum(axis=1)
    beneficiary["RenalDiseaseIndicator"] = beneficiary["RenalDiseaseIndicator"].replace({"Y": 1, "0": 0}).astype(int)
    beneficiary["IsDeceased"] = beneficiary["DOD"].notna().astype(int)
    return beneficiary


def preprocess_claims(inpatient, outpatient):
    inpatient = clean_columns(inpatient.copy())
    outpatient = clean_columns(outpatient.copy())
    inpatient = convert_dates(inpatient)
    outpatient = convert_dates(outpatient)
    inpatient["ClaimType"] = "Inpatient"
    outpatient["ClaimType"] = "Outpatient"
    inpatient["HospitalStay"] = (inpatient["DischargeDt"] - inpatient["AdmissionDt"]).dt.days
    inpatient.loc[inpatient["HospitalStay"] < 0, "HospitalStay"] = np.nan
    outpatient["HospitalStay"] = 0
    inpatient["ClaimDuration"] = (inpatient["ClaimEndDt"] - inpatient["ClaimStartDt"]).dt.days
    outpatient["ClaimDuration"] = (outpatient["ClaimEndDt"] - outpatient["ClaimStartDt"]).dt.days
    physician_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
    for c in physician_cols:
        inpatient[c] = inpatient[c].fillna("Unknown")
        outpatient[c] = outpatient[c].fillna("Unknown")
    for df in [inpatient, outpatient]:
        diag_cols = [c for c in df.columns if "DiagnosisCode" in c]
        for c in diag_cols:
            df[c] = df[c].fillna("Unknown")
        proc_cols = [c for c in df.columns if "ProcedureCode" in c]
        for c in proc_cols:
            df[c] = df[c].fillna("Unknown")
        if "DeductibleAmtPaid" in df.columns:
            df["DeductibleAmtPaid"] = df["DeductibleAmtPaid"].fillna(0)
        df["InscClaimAmtReimbursed"] = df["InscClaimAmtReimbursed"].fillna(0)
    return inpatient, outpatient


def build_claims(inpatient, outpatient, beneficiary):
    claims = pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    claims = claims.merge(beneficiary, on="BeneID", how="left")
    return claims


def diagnosis_diversity(df, provider_col="Provider"):
    diag_cols = [c for c in df.columns if "ClmDiagnosisCode" in c]
    proc_cols = [c for c in df.columns if "ClmProcedureCode" in c]
    diag_long = df[[provider_col] + diag_cols].melt(id_vars=provider_col, value_name="code")
    diag_long = diag_long[diag_long["code"] != "Unknown"]
    diag_diversity = diag_long.groupby(provider_col)["code"].nunique().rename("UniqueDiagnosisCodes")
    proc_long = df[[provider_col] + proc_cols].melt(id_vars=provider_col, value_name="code")
    proc_long = proc_long[(proc_long["code"] != "Unknown") & (proc_long["code"].notna())]
    proc_diversity = proc_long.groupby(provider_col)["code"].nunique().rename("UniqueProcedureCodes")
    return pd.concat([diag_diversity, proc_diversity], axis=1).reset_index()


def build_provider_features(claims):
    agg_dict = {
        "InscClaimAmtReimbursed": ["sum", "mean", "max", "std"],
        "DeductibleAmtPaid": ["sum", "mean"],
        "Age": ["mean", "max"],
        "HospitalStay": ["mean", "max"],
        "ClaimDuration": ["mean"],
        "TotalChronicDiseases": ["mean"],
        "RenalDiseaseIndicator": ["mean"],
        "IsDeceased": ["mean"],
    }
    feats = claims.groupby("Provider").agg(agg_dict)
    feats.columns = ["_".join(c) for c in feats.columns]
    feats = feats.reset_index()
    feats["TotalClaims"] = claims.groupby("Provider").size().values
    feats["UniqueBeneficiaries"] = claims.groupby("Provider")["BeneID"].nunique().values
    feats["ClaimsPerBeneficiary"] = feats["TotalClaims"] / feats["UniqueBeneficiaries"]
    feats["UniqueAttendingPhysicians"] = claims.groupby("Provider")["AttendingPhysician"].nunique().values
    feats["UniqueOperatingPhysicians"] = claims.groupby("Provider")["OperatingPhysician"].nunique().values
    feats["UniqueOtherPhysicians"] = claims.groupby("Provider")["OtherPhysician"].nunique().values
    claim_type_counts = claims.groupby(["Provider", "ClaimType"]).size().unstack(fill_value=0)
    claim_type_counts = claim_type_counts.rename(columns={"Inpatient": "InpatientClaims", "Outpatient": "OutpatientClaims"})
    for col in ["InpatientClaims", "OutpatientClaims"]:
        if col not in claim_type_counts.columns:
            claim_type_counts[col] = 0
    claim_type_counts = claim_type_counts.reset_index()
    feats = feats.merge(claim_type_counts, on="Provider", how="left")
    feats["InpatientRatio"] = feats["InpatientClaims"] / feats["TotalClaims"]
    feats["PctMale"] = claims.groupby("Provider")["Gender"].apply(lambda s: (s == 1).mean()).values
    feats["UniqueStates"] = claims.groupby("Provider")["State"].nunique().values
    diversity = diagnosis_diversity(claims)
    feats = feats.merge(diversity, on="Provider", how="left")
    feats = feats.fillna(0)
    return feats


# ---------------------------------------------------------------
# UI — three file uploaders (Inpatient / Outpatient / Beneficiary)
# ---------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    inpatient_file = st.file_uploader("Inpatient claims CSV", type="csv")
    beneficiary_file = st.file_uploader("Beneficiary details CSV", type="csv")
with col2:
    outpatient_file = st.file_uploader("Outpatient claims CSV", type="csv")

st.divider()
run = st.button("Predict Fraud Risk", type="primary", use_container_width=True)

if run:
    if not (inpatient_file and outpatient_file and beneficiary_file):
        st.error("Please upload all three files (Inpatient, Outpatient, Beneficiary).")
    else:
        with st.spinner("Engineering features and scoring providers..."):
            inpatient = pd.read_csv(inpatient_file)
            outpatient = pd.read_csv(outpatient_file)
            beneficiary = pd.read_csv(beneficiary_file)

            beneficiary = preprocess_beneficiary(beneficiary)
            inpatient, outpatient = preprocess_claims(inpatient, outpatient)
            claims = build_claims(inpatient, outpatient, beneficiary)
            provider_features = build_provider_features(claims)

            for col in feature_cols:
                if col not in provider_features.columns:
                    provider_features[col] = 0
            X = provider_features[feature_cols]

            X_input = scaler.transform(X) if scaler is not None else X
            proba = model.predict_proba(X_input)[:, 1]
            pred = np.where(proba >= 0.5, "Yes", "No")

            results = pd.DataFrame({
                "Provider": provider_features["Provider"],
                "FraudProbability": proba.round(4),
                "PredictedFraud": pred,
            }).sort_values("FraudProbability", ascending=False)

        st.success(f"Scored {len(results)} providers.")
        st.metric("Providers flagged as likely fraud", int((results["PredictedFraud"] == "Yes").sum()))
        st.dataframe(results, use_container_width=True, hide_index=True)

        st.download_button(
            "Download predictions as CSV",
            results.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    "This tool aggregates raw claim-level data into provider-level features "
    "(claim volume, billed amounts, physician network size, diagnosis diversity, etc.) "
    "and scores each provider's probability of being a potentially fraudulent Medicare provider."
)
