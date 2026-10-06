"""
prescriptive_engine.py
-----------------------
Dynamic Risk Scoring (Governance Risk Index) and ISO/IEC 27001 / NIST
SP 800-53 control recommendation rules.
"""

# Threat severity weights, W_k in [0, 10], per predicted class.
SEVERITY_WEIGHTS = {
    "normal": 0,
    "scanning": 3,
    "password": 5,
    "xss": 6,
    "injection": 6,
    "dos": 6,
    "mitm": 7,
    "ddos": 8,
    "backdoor": 10,
    "ransomware": 10,
}

TIERS = [(0, 30, "Low"), (30, 70, "Medium"), (70, 101, "High")]


def compute_risk_score(proba_by_class: dict) -> float:
    """
    Risk = ( sum_k P(Y=k) * W_k ) * 10, clipped to [0, 100].
    proba_by_class: {class_name: probability}
    """
    weighted = sum(
        p * SEVERITY_WEIGHTS.get(cls, 5)  # unseen classes default to mid severity
        for cls, p in proba_by_class.items()
    )
    score = weighted * 10.0
    return float(max(0.0, min(100.0, score)))


def risk_tier(score: float) -> str:
    for lo, hi, name in TIERS:
        if lo <= score < hi:
            return name
    return "High"


# ---------------------------------------------------------------------------
# Technical remediation rules
# ---------------------------------------------------------------------------
# Each rule: (predicate(raw_record, contributor_features) -> bool, action_text)

def _has_contrib(contrib_features, *needles):
    return any(any(n in f for n in needles) for f in contrib_features)


def _technical_rules(raw_record: dict, contributor_features: list):
    actions = []
    conn_state = str(raw_record.get("conn_state", "")).upper()
    dst_port = raw_record.get("dst_port", 0)
    duration = raw_record.get("duration", 0)
    src_bytes = raw_record.get("src_bytes", 0)
    src_pkts = raw_record.get("src_pkts", 0)

    try:
        dst_port = int(dst_port)
    except (TypeError, ValueError):
        dst_port = 0

    if conn_state == "REJ" and dst_port in (8080, 8081, 4444, 1337, 6666):
        actions.append("Deploy ingress port-filtering rule; block source subnet on the edge firewall.")

    if conn_state in ("REJ", "S0") and _has_contrib(contributor_features, "dst_port", "src_pkts"):
        actions.append("Rate-limit / block the source IP at the perimeter — connection pattern matches port scanning or a SYN flood.")

    if float(duration or 0) > 30 and float(src_bytes or 0) > 2000:
        actions.append("Terminate the active session; inspect the endpoint's process tree for reverse-shell or C2 beacon activity.")

    if int(src_pkts or 0) > 150:
        actions.append("Enable flow-rate anomaly detection / upstream scrubbing to absorb the volumetric burst.")

    if dst_port in (445, 3389):
        actions.append("Isolate the host from the network (contain), snapshot for forensics, and verify offline/immutable backups are intact before any recovery action.")

    if raw_record.get("service") == "http" and _has_contrib(contributor_features, "src_bytes", "dst_bytes"):
        actions.append("Enable a WAF rule set (OWASP CRS) in front of the affected web service and review recent request payloads for injection/XSS patterns.")

    if dst_port in (22, 21, 23, 3389) and conn_state in ("S0", "REJ", "SF"):
        actions.append("Enforce account lockout / MFA on the exposed authentication service; review recent auth logs for brute-force attempts.")

    if not actions:
        actions.append("Log and monitor: correlate this flow with adjacent traffic before escalating; no high-confidence technical trigger matched.")

    return actions


# ---------------------------------------------------------------------------
# Governance & compliance control mapping
# ---------------------------------------------------------------------------

CONTROL_MAP = {
    "normal": [],
    "backdoor": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.7", "name": "Protection against malware"},
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.20", "name": "Network security"},
        {"framework": "NIST SP 800-53", "control": "SI-3", "name": "Malicious Code Protection"},
        {"framework": "NIST SP 800-53", "control": "IR-4", "name": "Incident Handling"},
    ],
    "ransomware": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.7", "name": "Protection against malware"},
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.13", "name": "Information backup"},
        {"framework": "NIST SP 800-53", "control": "CP-9", "name": "System Backup"},
        {"framework": "NIST SP 800-53", "control": "IR-4", "name": "Incident Handling"},
    ],
    "scanning": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.16", "name": "Monitoring activities"},
        {"framework": "NIST SP 800-53", "control": "SC-7", "name": "Boundary Protection"},
        {"framework": "NIST SP 800-53", "control": "SI-4", "name": "System Monitoring"},
    ],
    "dos": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.6", "name": "Capacity management"},
        {"framework": "NIST SP 800-53", "control": "SC-5", "name": "Denial of Service Protection"},
    ],
    "ddos": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.6", "name": "Capacity management"},
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.22", "name": "Segregation of networks"},
        {"framework": "NIST SP 800-53", "control": "SC-5", "name": "Denial of Service Protection"},
    ],
    "password": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.5.17", "name": "Authentication information"},
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.5", "name": "Secure authentication"},
        {"framework": "NIST SP 800-53", "control": "IA-5", "name": "Authenticator Management"},
    ],
    "injection": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.28", "name": "Secure coding"},
        {"framework": "NIST SP 800-53", "control": "SI-10", "name": "Information Input Validation"},
    ],
    "xss": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.28", "name": "Secure coding"},
        {"framework": "NIST SP 800-53", "control": "SI-10", "name": "Information Input Validation"},
    ],
    "mitm": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.24", "name": "Use of cryptography"},
        {"framework": "NIST SP 800-53", "control": "SC-8", "name": "Transmission Confidentiality and Integrity"},
    ],
    "unauthorized_service_port": [
        {"framework": "ISO/IEC 27001:2022", "control": "A.8.9", "name": "Configuration management"},
    ],
}


def _governance_controls(predicted_type: str, raw_record: dict):
    controls = list(CONTROL_MAP.get(predicted_type, []))
    known_ports = {80, 443, 53, 22, 21, 23, 445, 3389, 8080, 123}
    try:
        dst_port = int(raw_record.get("dst_port", 0))
    except (TypeError, ValueError):
        dst_port = 0
    if dst_port and dst_port not in known_ports:
        controls.extend(CONTROL_MAP["unauthorized_service_port"])
    return controls


def generate_recommendations(predicted_type: str, contributors: list, raw_record: dict):
    contributor_features = [c["feature"] for c in contributors]
    return {
        "technical_actions": _technical_rules(raw_record, contributor_features),
        "governance_controls": _governance_controls(predicted_type, raw_record),
    }
