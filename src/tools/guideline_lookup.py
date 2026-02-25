"""
Simulated AHA / ESC guideline retriever.

In a production system this would query a real vector-store index over
guideline PDFs.  For the baseline we embed a small curated lookup table.
"""

from __future__ import annotations

from langchain_core.tools import tool

_GUIDELINES: dict[str, str] = {
    "atrial_fibrillation": (
        "AHA/ACC/HRS 2023: For patients with AF and CHA₂DS₂-VASc ≥2, "
        "oral anticoagulation (preferably DOAC) is recommended (Class I). "
        "Rate control target <110 bpm at rest. Rhythm control with catheter "
        "ablation is reasonable for symptomatic AF refractory to ≥1 AAD (Class IIa)."
    ),
    "stemi": (
        "ESC 2023 STEMI Guidelines: Immediate reperfusion therapy is recommended "
        "within 12 h of symptom onset. Primary PCI is the preferred strategy if "
        "door-to-balloon time <120 min. Dual antiplatelet therapy (aspirin + P2Y12 "
        "inhibitor) for ≥12 months. Beta-blockers within 24 h if no contraindication."
    ),
    "nstemi": (
        "ESC 2023 NSTE-ACS Guidelines: Early invasive strategy (<24 h) recommended "
        "for high-risk patients (GRACE >140). Fondaparinux preferred unless PCI "
        "planned within 24 h. Ticagrelor or prasugrel preferred over clopidogrel."
    ),
    "heart_failure": (
        "ACC/AHA 2022 HF Guidelines: GDMT for HFrEF includes ARNI/ACEi, "
        "beta-blocker, MRA, and SGLT2 inhibitor (Class I). ICD for primary "
        "prevention if LVEF ≤35% despite ≥3 months GDMT. CRT for LBBB with "
        "QRS ≥150 ms and LVEF ≤35%."
    ),
    "lvh": (
        "ESC 2024 Hypertension Guidelines: LVH is an independent risk factor "
        "for cardiovascular events. BP target <130/80 mmHg. Preferred agents: "
        "ACEi/ARB + CCB or thiazide. Screen for secondary causes if resistant."
    ),
    "bradycardia": (
        "ACC/AHA/HRS 2018 Bradycardia Guidelines: Permanent pacing indicated "
        "for symptomatic sinus node dysfunction (Class I). In asymptomatic "
        "patients with HR <40 bpm or pauses >3 s, pacing is reasonable (Class IIa)."
    ),
    "qt_prolongation": (
        "AHA/ACC 2017: QTc >500 ms significantly increases risk of torsades de "
        "pointes. Discontinue offending drugs. Correct electrolytes (K⁺ >4.0, "
        "Mg²⁺ >2.0). Consider isoproterenol or temporary pacing if recurrent TdP."
    ),
    "ventricular_tachycardia": (
        "ESC 2022 VA/SCD Guidelines: Sustained VT in structural heart disease "
        "warrants ICD (Class I). Catheter ablation for recurrent VT despite AAD "
        "(Class I). Amiodarone + beta-blocker for electrical storm."
    ),
    "default": (
        "No specific guideline found. Recommend comprehensive cardiac evaluation "
        "including 12-lead ECG, echocardiography, and laboratory workup. "
        "Follow general ACC/AHA risk assessment framework."
    ),
}


@tool
def lookup_guideline(condition: str) -> str:
    """Look up AHA/ESC clinical practice guideline summary for a cardiac condition.

    Args:
        condition: Cardiac condition keyword (e.g. 'atrial_fibrillation',
                   'stemi', 'heart_failure', 'lvh', 'bradycardia').

    Returns:
        Guideline summary text.
    """
    key = condition.lower().replace(" ", "_").replace("-", "_")
    for gk in _GUIDELINES:
        if gk in key or key in gk:
            return _GUIDELINES[gk]
    return _GUIDELINES["default"]
