"""
Create SMALL FAKE files in the CICIDS2017 and NSL-KDD formats.

They exist only to check that the code runs before the real dataset is
downloaded. The rows are random numbers, so any score obtained on them is
meaningless and must not be reported as a result.

Run from the project root:
    python ml/make_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

CICIDS_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
CICIDS_LABELS = {"BENIGN": 0.80, "DoS Hulk": 0.08, "PortScan": 0.06, "DDoS": 0.04,
                 "FTP-Patator": 0.01, "Web Attack \x96 Brute Force": 0.006, "Bot": 0.004}


def make_cicids(n=6000):
    labels = rng.choice(list(CICIDS_LABELS), size=n, p=list(CICIDS_LABELS.values()))
    attack = labels != "BENIGN"
    df = pd.DataFrame(rng.lognormal(3, 1.5, size=(n, len(CICIDS_FEATURES))), columns=CICIDS_FEATURES)
    df.loc[attack, CICIDS_FEATURES[1:12]] *= rng.uniform(2, 5, size=(attack.sum(), 11))  # signal
    for col in ["Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count"]:
        df[col] = 0  # always-zero columns, like the real data
    df["Fwd Header Length.1"] = df["Fwd Header Length"]
    df.loc[rng.random(n) < 0.01, "Flow Bytes/s"] = np.inf  # invalid values like the real data
    df.loc[rng.random(n) < 0.01, "Flow Packets/s"] = np.nan
    df["Label"] = labels
    df = pd.concat([df, df.sample(200, random_state=0)])  # duplicate rows
    df.columns = [" " + c for c in df.columns]  # the real headers have leading spaces
    return df


def make_nslkdd(n=4000):
    labels = rng.choice(["normal", "neptune", "smurf", "ipsweep", "satan", "guess_passwd",
                         "buffer_overflow"], size=n, p=[0.53, 0.25, 0.08, 0.05, 0.05, 0.03, 0.01])
    attack = labels != "normal"
    data = {
        "duration": rng.integers(0, 100, n),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n),
        "service": np.where(attack, rng.choice(["private", "http", "ecr_i"], n),
                            rng.choice(["http", "smtp", "ftp_data", "domain_u"], n)),
        "flag": np.where(attack, rng.choice(["S0", "REJ", "SF"], n), "SF"),
        "src_bytes": rng.lognormal(5, 2, n).astype(int),
        "dst_bytes": rng.lognormal(6, 2, n).astype(int),
    }
    df = pd.DataFrame(data)
    rest = [
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    for col in rest:
        df[col] = rng.random(n).round(2)
    df["num_outbound_cmds"] = 0
    df.loc[attack, "serror_rate"] = rng.uniform(0.5, 1, attack.sum()).round(2)
    df["label"] = labels
    df["difficulty"] = rng.integers(1, 22, n)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    make_cicids().to_csv(out / "cicids2017_sample.csv", index=False)
    make_nslkdd().to_csv(out / "nslkdd_sample.txt", index=False, header=False)
    print(f"Wrote FAKE test files to {out}/ (for checking the code only, not for results)")
