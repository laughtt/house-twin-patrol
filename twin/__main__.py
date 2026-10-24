"""House Twin, one command per step. From house-twin-patrol/:

python -m twin build <scan.zip|folder> [--markers m.json] [--out dir] [--no-model] iPhone LiDAR scan → house.json (+ top.png, cloud.ply)
python -m twin sim [house.json] --spawn x,y,heading                         house.json → Pollen duck-sim scene + previews
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser(prog="twin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="LiDAR scan → house.json")
    b.add_argument("scan")
    b.add_argument("--markers", help="hand-placed markers (house frame) when the scan shows none")
    b.add_argument("--out")
    b.add_argument("--no-model", action="store_true", help="skip the curate model (config lists decide)")
    b.add_argument("--tags", help="extra tags in the app's format [{i, uv, label}] (from the results page)")
    sub.add_parser("sim", help="house.json → duck-sim scene", add_help=False)
    a, rest = ap.parse_known_args()
    if a.cmd == "build":
        sys.path.insert(0, str(ROOT / "lidar/pipeline"))
        import build_house
        house, out = build_house.build(a.scan, a.out, extra_markers=a.markers, use_model=not a.no_model, extra_tags=a.tags)
        print((out / "report.json").read_text())
        print(f"wrote {out}  (next: python -m twin sim {out / 'house.json'} --spawn x,y,heading)")
    else:
        from twin import to_mjcf
        to_mjcf.main(rest)


if __name__ == "__main__":
    main()
