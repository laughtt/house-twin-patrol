"""House Twin: an iPhone LiDAR walk → house.json → the duck's simulated home. Settings in config.toml."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
