"""
Clinical risk score calculators used as LangChain tools.

Implements:
    - CHA₂DS₂-VASc  (stroke risk in atrial fibrillation)
    - GRACE score    (ACS mortality estimate)
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def cha2ds2_vasc_score(
    age: int,
    sex: str = "male",
    chf: bool = False,
    hypertension: bool = False,
    stroke_tia_history: bool = False,
    vascular_disease: bool = False,
    diabetes: bool = False,
) -> dict:
    """Calculate CHA₂DS₂-VASc score for stroke risk in atrial fibrillation.

    Args:
        age: Patient age in years.
        sex: 'male' or 'female'.
        chf: Congestive heart failure history.
        hypertension: Hypertension history.
        stroke_tia_history: Prior stroke / TIA / thromboembolism.
        vascular_disease: Prior MI, PAD, or aortic plaque.
        diabetes: Diabetes mellitus.

    Returns:
        dict with score, risk_category, and annual_stroke_rate.
    """
    score = 0
    if chf:
        score += 1
    if hypertension:
        score += 1
    if stroke_tia_history:
        score += 2
    if vascular_disease:
        score += 1
    if diabetes:
        score += 1
    if sex.lower() == "female":
        score += 1
    if 65 <= age < 75:
        score += 1
    elif age >= 75:
        score += 2

    stroke_rates = {
        0: 0.0, 1: 1.3, 2: 2.2, 3: 3.2, 4: 4.0,
        5: 6.7, 6: 9.8, 7: 9.6, 8: 6.7, 9: 15.2,
    }
    annual_rate = stroke_rates.get(min(score, 9), 15.2)

    if score == 0:
        risk = "Low"
    elif score == 1:
        risk = "Low-Moderate"
    else:
        risk = "Moderate-High"

    return {
        "score": score,
        "risk_category": risk,
        "annual_stroke_rate_pct": annual_rate,
        "recommendation": (
            "Anticoagulation recommended" if score >= 2
            else "Consider anticoagulation" if score == 1
            else "No anticoagulation needed"
        ),
    }


@tool
def grace_score(
    age: int,
    heart_rate: int = 80,
    systolic_bp: int = 120,
    creatinine: float = 1.0,
    killip_class: int = 1,
    cardiac_arrest: bool = False,
    st_deviation: bool = False,
    elevated_troponin: bool = False,
) -> dict:
    """Estimate GRACE score for in-hospital mortality after ACS.

    This is a simplified approximation.  A production system must use the
    validated regression from the GRACE study.

    Args:
        age: Patient age.
        heart_rate: Beats per minute.
        systolic_bp: mmHg.
        creatinine: mg/dL.
        killip_class: 1-4.
        cardiac_arrest: At admission.
        st_deviation: ST-segment deviation on ECG.
        elevated_troponin: Troponin above normal.

    Returns:
        dict with estimated score, risk_category, and mortality estimate.
    """
    score = 0
    # Age contribution (simplified)
    if age < 30:
        score += 0
    elif age < 40:
        score += 8
    elif age < 50:
        score += 25
    elif age < 60:
        score += 41
    elif age < 70:
        score += 58
    elif age < 80:
        score += 75
    else:
        score += 91

    # Heart rate
    if heart_rate < 50:
        score += 0
    elif heart_rate < 70:
        score += 3
    elif heart_rate < 90:
        score += 9
    elif heart_rate < 110:
        score += 15
    elif heart_rate < 150:
        score += 24
    else:
        score += 38

    # Systolic BP (inverse)
    if systolic_bp < 80:
        score += 58
    elif systolic_bp < 100:
        score += 53
    elif systolic_bp < 120:
        score += 43
    elif systolic_bp < 140:
        score += 34
    elif systolic_bp < 160:
        score += 24
    else:
        score += 0

    # Creatinine
    if creatinine > 2.0:
        score += 28
    elif creatinine > 1.5:
        score += 21
    elif creatinine > 1.0:
        score += 14
    else:
        score += 7

    # Killip class
    score += (killip_class - 1) * 20

    if cardiac_arrest:
        score += 39
    if st_deviation:
        score += 28
    if elevated_troponin:
        score += 14

    if score <= 108:
        risk = "Low"
        mortality = "<1%"
    elif score <= 140:
        risk = "Intermediate"
        mortality = "1-3%"
    else:
        risk = "High"
        mortality = ">3%"

    return {
        "grace_score": score,
        "risk_category": risk,
        "in_hospital_mortality": mortality,
    }
