def main(cfg=None):
    """Generate every paper figure whose inputs exist."""
    from src.common import load_config
    from src.visualization import ablation, emission, metrics, predictions
    cfg = cfg or load_config()
    for mod in (predictions, metrics, ablation, emission):
        mod.main(cfg)
