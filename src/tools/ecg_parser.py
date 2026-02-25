"""
ECG parameter parser / interpreter tool.

Converts raw ECG measurement parameters into a structured clinical summary
that agents can reason over.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def parse_ecg_parameters(
    rr_interval: int | None = None,
    p_onset: int | None = None,
    p_end: int | None = None,
    qrs_onset: int | None = None,
    qrs_end: int | None = None,
    t_end: int | None = None,
    p_axis: int | None = None,
    qrs_axis: int | None = None,
    t_axis: int | None = None,
    report_text: str = "",
) -> dict:
    """Parse raw ECG interval/axis measurements and return a clinical interpretation.

    Args:
        rr_interval: RR interval in ms.
        p_onset: P-wave onset in ms.
        p_end: P-wave end in ms.
        qrs_onset: QRS onset in ms.
        qrs_end: QRS end in ms.
        t_end: T-wave end in ms.
        p_axis: P-wave axis in degrees.
        qrs_axis: QRS axis in degrees.
        t_axis: T-wave axis in degrees.
        report_text: Machine-generated ECG report text.

    Returns:
        Dictionary with derived intervals and flags.
    """
    findings: list[str] = []
    derived: dict[str, float | str | None] = {}

    # Heart rate from RR interval
    if rr_interval and rr_interval > 0 and rr_interval < 29000:
        hr = round(60000 / rr_interval)
        derived["heart_rate_bpm"] = hr
        if hr < 60:
            findings.append(f"Bradycardia (HR {hr} bpm)")
        elif hr > 100:
            findings.append(f"Tachycardia (HR {hr} bpm)")
        else:
            findings.append(f"Normal rate (HR {hr} bpm)")

    # PR interval
    if p_onset is not None and qrs_onset is not None:
        if p_onset < 29000 and qrs_onset < 29000:
            pr = qrs_onset - p_onset
            derived["pr_interval_ms"] = pr
            if pr > 200:
                findings.append(f"Prolonged PR interval ({pr} ms) – possible 1st degree AV block")
            elif pr < 120:
                findings.append(f"Short PR interval ({pr} ms) – consider pre-excitation")

    # QRS duration
    if qrs_onset is not None and qrs_end is not None:
        qrs_dur = qrs_end - qrs_onset
        derived["qrs_duration_ms"] = qrs_dur
        if qrs_dur > 120:
            findings.append(f"Wide QRS ({qrs_dur} ms) – consider bundle branch block")
        elif qrs_dur < 80:
            findings.append(f"Narrow QRS ({qrs_dur} ms)")

    # QT / QTc
    if qrs_onset is not None and t_end is not None and rr_interval:
        qt = t_end - qrs_onset
        derived["qt_interval_ms"] = qt
        if rr_interval > 0 and rr_interval < 29000:
            rr_sec = rr_interval / 1000
            qtc = qt / (rr_sec ** 0.5)
            derived["qtc_bazett_ms"] = round(qtc)
            if qtc > 500:
                findings.append(f"Markedly prolonged QTc ({round(qtc)} ms) – HIGH RISK for TdP")
            elif qtc > 450:
                findings.append(f"Prolonged QTc ({round(qtc)} ms)")

    # Axis
    if qrs_axis is not None and qrs_axis < 29000:
        derived["qrs_axis_deg"] = qrs_axis
        if -30 <= qrs_axis <= 90:
            findings.append("Normal QRS axis")
        elif -90 <= qrs_axis < -30:
            findings.append("Left axis deviation")
        elif 90 < qrs_axis <= 180:
            findings.append("Right axis deviation")
        else:
            findings.append("Extreme / indeterminate axis")

    # Report text
    if report_text:
        derived["report"] = report_text

    return {
        "derived_parameters": derived,
        "findings": findings,
        "summary": "; ".join(findings) if findings else "No significant findings from provided parameters.",
    }
