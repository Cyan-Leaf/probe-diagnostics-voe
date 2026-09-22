"""Verify that the row-space reformulation in t4_caliper._rowspace is numerically exact.

Fits the same L2 logistic regression (a) directly on the standardised p-dim features and
(b) on the r-dim row-space representation, and compares decision values / AUC / timing.

Run: uv run --no-project python scripts/t4_verify_rowspace.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from t4_caliper import _load_xy, _rowspace

OUT = Path(__file__).resolve().parents[1] / "results" / "T4_caliper"


def main() -> None:
    X, y, g = _load_xy({"source": "real_dit", "backbone": "real", "layer": 10, "t": 600, "seed": 0})
    tr, te = next(GroupKFold(5).split(X, y, g))
    sc = StandardScaler().fit(X[tr])
    A, B = sc.transform(X[tr]), sc.transform(X[te])

    rows = []
    with threadpool_limits(limits=8):
        for C in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            t0 = time.time()
            m1 = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(A, y[tr])
            s1 = m1.decision_function(B)
            t1 = time.time() - t0

            t0 = time.time()
            Z, Zte = _rowspace(A, B)
            m2 = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Z, y[tr])
            s2 = m2.decision_function(Zte)
            t2 = time.time() - t0

            rows.append(
                {
                    "C": C,
                    "dim_direct": int(A.shape[1]),
                    "dim_rowspace": int(Z.shape[1]),
                    "auc_direct": float(roc_auc_score(y[te], s1)),
                    "auc_rowspace": float(roc_auc_score(y[te], s2)),
                    "max_abs_decision_diff": float(np.abs(s1 - s2).max()),
                    "corr_decisions": float(np.corrcoef(s1, s2)[0, 1]),
                    "secs_direct": round(t1, 2),
                    "secs_rowspace": round(t2, 2),
                    "speedup": round(t1 / t2, 1),
                }
            )
            print(json.dumps(rows[-1]), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rowspace_equivalence_check.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT/'rowspace_equivalence_check.json'}")


if __name__ == "__main__":
    main()
