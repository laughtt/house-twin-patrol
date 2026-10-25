#!/usr/bin/env bash
# Launch the House Twin Patrol demo (see DEMO.md).
#   scripts/demo.sh offline [house]      self-location, camera + approval tests, fast tests on samples/synth-multi (no phone, robot or AWS)
#   scripts/demo.sh live [house] [--ask] live walk on Pollen's duck-sim with the camera, then the film
#   scripts/demo.sh server               the app's server on http://127.0.0.1:8790 (needs API_KEY, TOKEN_SECRET, APP_PASSWORD_HASH)
#   scripts/demo.sh app                  generate the Xcode project and open it
set -euo pipefail
cd "$(dirname "$0")/.."
PY=(uv run --python 3.12 --with-requirements requirements.lock python)  # pinned versions
MICRODUCK=${MICRODUCK:-$HOME/Pollen/microduck}
SCENE=${DUCK_SIM_SCENE:-$HOME/Pollen/microduck_rl/src/mjlab_microduck/robot/microduck/scene_myhouse.xml}

case "${1:-}" in
offline)
    house=${2:-samples/synth-multi}  # a synthetic 4-room home, already built (TESTING.md)
    "${PY[@]}" eval/selfloc.py --houses "$house" --variants base,spin+rowfix+range+fit+relook+jump --seeds 4 --run offline
    "${PY[@]}" eval/vision_walk.py --house "$house" --seeds 8
    "${PY[@]}" -m pytest eval/adversarial -q ||
        echo "known failure: twin_rooms_tv_cue (rooms that look alike to the ToF; DEMO.md, limits)"
    ;;
live)
    house=out/houses/lidar-20261024-120616 ask=""
    for a in "${@:2}"; do if [ "$a" = --ask ]; then ask=--ask; else house=$a; fi; done
    out=out/walk/$(basename "$house")
    [ -e "$out" ] && mv "$out" "$out-$(date +%Y%m%d-%H%M%S)"  # keep the previous walk
    mkdir -p "$out"
    (cd "$MICRODUCK" && scripts/duck-sim down >/dev/null 2>&1 || true
     DUCK_SIM_SCENE=$SCENE DUCK_SIM_CAMERAS=a DUCK_SIM_VIEWER=0 scripts/duck-sim)
    uv run --python 3.12 --with numpy python -m agent.duck_recorder --out "$out" > /dev/null 2>&1 &
    recorder=$!
    trap 'kill $recorder 2>/dev/null || true' EXIT
    sleep 3
    "${PY[@]}" -m agent.walk --house "$house" --no-standin $ask
    kill -TERM $recorder; sleep 2
    "${PY[@]}" -m agent.film --house "$house" --speed 2
    echo "trace, summary and film: $out"
    ;;
server)
    : "${API_KEY:?set API_KEY}" "${TOKEN_SECRET:?set TOKEN_SECRET}" "${APP_PASSWORD_HASH:?make one with: python -m service.passwd}"
    cd service && uv run --python 3.12 --with-requirements requirements.txt uvicorn app:app --host 127.0.0.1 --port 8790
    ;;
app)
    cd app && xcodegen generate && open HouseTwinLidar.xcodeproj
    ;;
*)
    sed -n 2,6p "$0"; exit 1
    ;;
esac
