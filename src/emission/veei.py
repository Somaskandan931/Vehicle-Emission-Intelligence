"""VEEI = sum_i w_i * (P_i / L_i)   (paper definition).

Limits, weights and their sources MUST come from config.yaml (veei section).
The code refuses to run with missing values so nothing is silently invented.
"""


def _require(cfg):
    v = cfg["veei"]
    problems = [p for p in cfg["data"]["target_cols"]
                if v["limits"].get(p) is None or v["weights"].get(p) is None]
    if problems:
        raise ValueError(
            f"VEEI limits/weights missing for {problems}. Fill veei.limits / veei.weights / "
            "veei.sources in config.yaml with values you can cite in the paper.")
    return cfg["data"]["target_cols"], v["limits"], v["weights"]


def compute_veei(df, cfg, suffix="_pred"):
    """df has columns '<pollutant><suffix>'. Returns a Series of VEEI values."""
    pols, limits, weights = _require(cfg)
    return sum(weights[p] * df[f"{p}{suffix}"] / limits[p] for p in pols)
