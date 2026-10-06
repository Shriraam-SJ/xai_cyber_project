"""
data_pipeline.py
-----------------
Ingestion, cleaning, feature engineering, and encoding for the ToN_IoT
network flow dataset.

Feature design note (why this isn't just flow metadata):
Transport-only fields (dst_port, proto, service, conn_state, byte/packet
counts) let web-layer attacks (injection, xss, password) collide with
ordinary browsing on dst_port=80/443 -- both look like "tcp, http, SF".
This pipeline adds two kinds of extra signal to break that collision:
  1. The low-cardinality, non-identifying application-layer fields ToN_IoT
     already provides (http_method, http_status_code, request/response
     body lengths, ssl_established/resumed, dns_rejected/AA/RD/RA,
     dns_qtype/rcode). These are NOT dropped as leakage -- unlike raw
     identifiers (src_ip/dst_ip) or free text (http_uri, ssl_subject/
     issuer, dns_query, which stay dropped below), they're exactly the
     kind of signal a real WAF/IDS uses and they generalize across
     captures.
  2. Engineered behavioral-shape / interaction features (payload_ratio,
     packet_density, is_standard_web_port, web_anomalous_state, etc.)
     that capture *how* a flow behaves independent of which port it used.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Raw numeric fields, filled directly from the CSV (or defaults).
BASE_NUMERIC_COLS = [
    "duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts",
    "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "src_port", "dst_port",
    "http_status_code", "http_request_body_len", "http_response_body_len",
    "http_trans_depth", "dns_qtype", "dns_rcode",
]

# Computed from BASE_NUMERIC_COLS (+ conn_state) after cleaning -- capture
# behavioral shape (burstiness, asymmetry, standard-port-but-abnormal-state)
# so the model has signal that survives even when port/service collide.
ENGINEERED_NUMERIC_COLS = [
    "bytes_per_pkt_src", "bytes_per_pkt_dst", "payload_ratio", "pkt_ratio",
    "bytes_per_sec", "packet_density", "is_standard_web_port",
    "web_anomalous_state", "weird_present",
]

NUMERIC_COLS = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS

CATEGORICAL_COLS = [
    "proto", "service", "conn_state", "http_method", "ssl_version",
    "ssl_resumed", "ssl_established", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
]

# Dropdown options per categorical field, for the /api/schema endpoint and
# frontend data-entry form.
CATEGORICAL_OPTIONS = {
    "proto": ["tcp", "udp", "icmp"],
    "service": ["unknown", "http", "dns", "ssl", "ftp", "ssh", "dhcp", "smtp"],
    "conn_state": ["S0", "S1", "SF", "REJ", "RSTO", "RSTR", "OTH"],
    "http_method": ["unknown", "GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
    "ssl_version": ["unknown", "TLSv10", "TLSv11", "TLSv12", "TLSv13", "SSLv3"],
    "ssl_resumed": ["unknown", "T", "F"],
    "ssl_established": ["unknown", "T", "F"],
    "dns_AA": ["unknown", "T", "F"],
    "dns_RD": ["unknown", "T", "F"],
    "dns_RA": ["unknown", "T", "F"],
    "dns_rejected": ["unknown", "T", "F"],
}

# Backwards-compatible aliases.
PROTOS = CATEGORICAL_OPTIONS["proto"]
SERVICES = CATEGORICAL_OPTIONS["service"]
CONN_STATES = CATEGORICAL_OPTIONS["conn_state"]

# Columns that are direct leakage / high-cardinality identifiers -- dropped
# before modeling. These identify a specific host/session (the model would
# memorize an IP or a URI string instead of learning generalizable
# behavior) or are unbounded free text.
LEAKAGE_COLS = [
    "ts", "src_ip", "dst_ip", "http_uri", "ssl_subject", "ssl_issuer",
    "dns_query", "http_user_agent",
]

TARGET_LABEL = "label"   # binary: 0 normal, 1 attack
TARGET_TYPE = "type"     # multi-class

ATTACK_TYPES = [
    "normal", "backdoor", "dos", "ddos", "scanning",
    "injection", "ransomware", "password", "mitm", "xss",
]

# Classes most prone to the port/service confound -- flagged for a
# low-confidence-context caveat when stage-2 payload-level signal is weak.
WEB_LAYER_CLASSES = {"injection", "xss", "password"}

ALL_INPUT_COLS = NUMERIC_COLS + CATEGORICAL_COLS

FEATURE_DEFAULTS = {
    "duration": 0.0, "src_bytes": 0, "dst_bytes": 0, "missed_bytes": 0,
    "src_pkts": 0, "src_ip_bytes": 0, "dst_pkts": 0, "dst_ip_bytes": 0,
    "src_port": 0, "dst_port": 0,
    "http_status_code": 0, "http_request_body_len": 0, "http_response_body_len": 0,
    "http_trans_depth": 0, "dns_qtype": 0, "dns_rcode": 0,
    "proto": "tcp", "service": "unknown", "conn_state": "SF",
    "http_method": "unknown", "ssl_version": "unknown",
    "ssl_resumed": "unknown", "ssl_established": "unknown",
    "dns_AA": "unknown", "dns_RD": "unknown", "dns_RA": "unknown", "dns_rejected": "unknown",
}

STANDARD_WEB_PORTS = (80, 443)


# ---------------------------------------------------------------------------
# Real-data loading + cleaning + feature engineering
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_data(df)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Behavioral-shape ratios and interaction features, computed from
    already-cleaned numeric/categorical fields (conn_state is lowercased
    by this point -- compare against "sf", not "SF").
    """
    df = df.copy()
    src_pkts_safe = df["src_pkts"].replace(0, np.nan)
    dst_pkts_safe = df["dst_pkts"].replace(0, np.nan)
    duration_safe = df["duration"].clip(lower=0.001)

    df["bytes_per_pkt_src"] = (df["src_bytes"] / src_pkts_safe).fillna(0)
    df["bytes_per_pkt_dst"] = (df["dst_bytes"] / dst_pkts_safe).fillna(0)
    df["payload_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["pkt_ratio"] = df["src_pkts"] / (df["dst_pkts"] + 1)
    df["bytes_per_sec"] = (df["src_bytes"] + df["dst_bytes"]) / duration_safe
    df["packet_density"] = (df["src_pkts"] + df["dst_pkts"]) / duration_safe
    df["is_standard_web_port"] = df["dst_port"].isin(STANDARD_WEB_PORTS).astype(int)
    df["web_anomalous_state"] = (
        (df["is_standard_web_port"] == 1) & (df["conn_state"] != "sf")
    ).astype(int)

    if "weird_name" in df.columns:
        df["weird_present"] = df["weird_name"].notna().astype(int)
    else:
        df["weird_present"] = 0

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Normalize null/placeholder tokens. In Zeek/Bro-style ToN_IoT exports,
    # "-" means "not applicable / no value" (e.g. service="-" when no
    # application protocol was identified) -- it is data, not a missing
    # value, so it maps to the same "unknown"/0 defaults used elsewhere.
    # This is also the right pre-drop moment to capture "weird_name
    # present", since "-" becomes NaN here.
    placeholder_tokens = ["-", "none", "None", "NONE", "NaN", "nan", ""]
    df = df.replace(placeholder_tokens, np.nan)

    for col in BASE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = FEATURE_DEFAULTS[col]

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.lower()
        else:
            df[col] = FEATURE_DEFAULTS[col]

    df = _engineer_features(df)

    if TARGET_LABEL not in df.columns and TARGET_TYPE in df.columns:
        df[TARGET_LABEL] = (df[TARGET_TYPE].str.lower() != "normal").astype(int)

    if TARGET_TYPE in df.columns:
        df[TARGET_TYPE] = df[TARGET_TYPE].fillna("normal").astype(str).str.lower()

    return df


def finalize_row(raw_record: dict) -> pd.DataFrame:
    """
    Turns a single raw flow record (data-entry form / CLI / API) into a
    one-row DataFrame with defaults applied, cleaning/engineering run, and
    columns ordered to match ALL_INPUT_COLS -- the same path a training row
    goes through, so live prediction and training are processed identically.
    """
    merged = {**FEATURE_DEFAULTS, **raw_record}
    row_df = pd.DataFrame([merged])
    row_df = clean_data(row_df)
    return row_df[ALL_INPUT_COLS]


# ---------------------------------------------------------------------------
# Encoding / preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    RobustScaler for numeric flow/byte/packet counters (robust to the heavy
    outliers typical of DoS/DDoS bursts), One-Hot for the low-cardinality
    categorical protocol fields.
    """
    return ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer):
    num_names = NUMERIC_COLS
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS))
    return list(num_names) + cat_names


def prepare_dataset(df: pd.DataFrame, test_size: float = 0.25):
    """
    Single-stage prep: X/y split on the multi-class `type` target directly.
    Used by demo_inference.py. For the hierarchical two-stage architecture
    used by app.py, see prepare_dataset_two_stage().
    """
    df = clean_data(df) if TARGET_LABEL not in df.columns else df
    X_raw = df[ALL_INPUT_COLS].copy()
    y = df[TARGET_TYPE].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = get_feature_names(preprocessor)

    return X_train, X_test, y_train.values, y_test.values, preprocessor, feature_names, X_train_raw, X_test_raw


def prepare_dataset_two_stage(df: pd.DataFrame, test_size: float = 0.25):
    """
    Prep for the hierarchical architecture: one shared preprocessor (fit
    once, so stage 1 and stage 2 share an identical feature space), plus
    both the binary `label` and multi-class `type` targets so app.py can
    train stage 1 (anomaly gate) on the full set and stage 2 (attack
    categorizer) on the anomalous subset only.

    Returns:
        X_train, X_test,
        y_label_train, y_label_test,   # binary: 0 normal / 1 anomaly
        y_type_train, y_type_test,     # string class incl. "normal"
        preprocessor, feature_names, X_train_raw, X_test_raw
    """
    df = clean_data(df) if TARGET_LABEL not in df.columns else df
    X_raw = df[ALL_INPUT_COLS].copy()
    y_label = df[TARGET_LABEL].astype(int).copy()
    y_type = df[TARGET_TYPE].copy()

    (X_train_raw, X_test_raw, y_label_train, y_label_test,
     y_type_train, y_type_test) = train_test_split(
        X_raw, y_label, y_type, test_size=test_size,
        random_state=RANDOM_STATE, stratify=y_type,
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = get_feature_names(preprocessor)

    return (X_train, X_test, y_label_train.values, y_label_test.values,
            y_type_train.values, y_type_test.values, preprocessor, feature_names,
            X_train_raw, X_test_raw)


def normal_background_sample(X_train, y_type_train, max_rows: int = 50, seed: int = RANDOM_STATE):
    """
    A small sample of preprocessed *normal*-class training rows, used as
    the reference background for differential SHAP/occlusion attribution
    (root-cause drivers are computed relative to what "normal" looks like,
    not relative to an arbitrary global average that mixes attack rows in).
    Returns a dense 2D numpy array, or None if no normal rows are present.
    """
    mask = np.asarray(y_type_train) == "normal"
    if not mask.any():
        return None
    X_normal = X_train[mask]
    if hasattr(X_normal, "toarray"):
        X_normal = X_normal.toarray()
    X_normal = np.asarray(X_normal)

    n = X_normal.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        X_normal = X_normal[idx]
    return X_normal