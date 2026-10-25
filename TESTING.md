# Testing House Twin Patrol (no iPhone needed)

Three ways in, from quickest to fullest. None needs the iPhone app: a synthetic home is already built in
`samples/synth-multi` (kitchen + living room, hallway, bedroom), and the same home is on the server as a ready map.

## A. On your machine: 1 minute, no phone, no robot, no AWS
Needs macOS or Linux and [uv](https://docs.astral.sh/uv/). Versions are pinned in `requirements.lock`.
```
git clone https://github.com/laughtt/house-twin-patrol && cd house-twin-patrol
scripts/demo.sh offline
```
It runs three things on `samples/synth-multi` (results land in `out/`):

1. **Self-location** (`eval/selfloc.py`): the robot is dropped somewhere unknown and patrols every room, with the
   old filter (`base`) and the final one, with a perfect sensor (`sim`) and a rough one (`real`: 2.5 m range, 20 % of
   zones dropped). Expected (4 starts each):

   | | base, sim | final, sim | base, rough | final, rough |
   |---|---|---|---|---|
   | time sure but more than 50 cm off | 5.8 % | 0 % | 55.9 % | 0 % |
   | runs that locked onto a wrong place | 4/4 | 0/4 | 4/4 | 0/4 |
   | false arrivals | 0 | 0 | 6 | 0 |
   | goals reached | 15/16 | 16/16 | 7/16 | 16/16 |

   Page with the table: `out/selfloc/offline/report.html`.
2. **The agent loop with the camera** (`eval/vision_walk.py`, 8 starts × 4 goals): each step the twin renders what
   the robot's camera sees and OpenCV 5 reads it. Expected: ToF alone 32/32 goals, ToF + camera 31/32, **14 decisions
   changed by the camera**. See them:
   ```
   grep -h vision_decision out/vision/synth-multi/default/tof+camera/seed*/trace.jsonl | head -3
   ```
   Each line has the decision **before** the frame and **after** it (look/go, the room it believes it's in, the route).
3. **Fast tests** (`eval/adversarial`): 40 pass, 16 skip (they need the author's private flat twin; with it, 72 of 73
   pass, see REPORT.md).

Your own variations: `eval/selfloc.py --help`, `eval/vision_walk.py --decline` (the person always says no),
`--stickers` (ArUco markers on), `--seeds 16`.

## B. The recorded live walk: the evidence, no setup
`samples/live_walk/` is the trace of the live run shown in `docs/robot_walk.gif`: Pollen's real robot software and
walking policy driving a simulated body in the twin of a scanned house, ToF + camera only, zero stickers.
```
grep -E '"event": "(vision_decision|approval|arrive)"' samples/live_walk/trace.jsonl
python3 -m json.tool samples/live_walk/summary.json | head -30
```
You will see 4 camera decisions, 2 questions to a person ("Lost for 20 s … Keep looking for the bedroom_2?") and 4
arrivals (13–30 cm from the goal). Every step has the truth and the robot's belief side by side (`true`, `est`).

## C. On the web: the judge account
Server: `https://game.adtomic.co` (AWS EC2 Graviton4). The user name and password are in the submission form.
1. Get a token (valid 30 days):
   ```
   curl -s -X POST https://game.adtomic.co/auth/login -H 'Content-Type: application/json' \
     -d '{"user":"judge","password":"<from the form>"}'
   ```
2. Open the ready sample map in a browser, with your token as `k`:
   - 3D: `https://game.adtomic.co/lidar/3d?s=lidar-20261025-180000&k=<token>`
   - all maps and their state: `https://game.adtomic.co/lidar/maps` with header `Authorization: Bearer <token>`
3. Build a map yourself on the server (OpenCV 5 on Graviton): make the synthetic scan (3 minutes), upload it, and it
   builds by itself:
   ```
   uv run --python 3.12 --with-requirements requirements.lock python eval/synth_lidar.py --house multi
   curl -X POST "https://game.adtomic.co/upload?session=lidar-$(date +%Y%m%d-%H%M%S)&room=lidar&ext=zip" \
     -H "Authorization: Bearer <token>" -H "Content-Type: application/zip" --data-binary @out/lidar/synth-multi.zip
   ```
   Then watch it in `/lidar/maps` (state `building` → `done`, about 2 minutes) and open it as in step 2.
   (Building on your own machine needs the model files in `service/models/`, which are not in the repo.)

## D. The iPhone app (optional)
Needs an iPhone with LiDAR and Xcode: `scripts/demo.sh app`, then sign in with the judge account. My maps shows the
sample map. Build and signing: `app/README.md`. Screenshots: `docs/app_screens.jpg`.

## E. The live robot (optional)
Needs Pollen's duck-sim (`~/Pollen/microduck`, `~/Pollen/microduck_rl`) with the house as its scene:
`scripts/demo.sh live --ask` (you answer the robot's questions in the terminal), then the film is rendered. We are
happy to run it over a live screen-share.

## What to look for
- **OpenCV 5 builds the map:** `docs/map_build.gif`, the stage pictures, REPORT.md §4.
- **OpenCV output changes what the robot does:** `vision_decision` events (A2, B), the banners in the film.
- **Human control:** `approval` events (B), `--ask` (E), `--decline` (A).
- **Failure handling and limits:** REPORT.md §8.
