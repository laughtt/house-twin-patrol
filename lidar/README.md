# LiDAR capture → house.json

`../app/` is the iPhone app that records RGB, LiDAR depth and the ARKit pose while you walk through the house.
`pipeline/build_house.py` turns one recording into `house.json` (PLAN §6.3), `top.png`, `cloud.ply` and `report.json`.

## Install the app (once, about 10 min)
Needs an iPhone with LiDAR (12 Pro or newer Pro model) and a free Apple ID. Other phones show a "no LiDAR" message.

1. On the Mac, open `house-twin-patrol/app/HouseTwinLidar.xcodeproj` in Xcode.
2. Click the blue **HouseTwinLidar** project → target **HouseTwinLidar** → **Signing & Capabilities**:
   tick *Automatically manage signing*, set **Team** to your Apple ID (Personal Team; add it under Xcode → Settings → Accounts if needed).
   `project.yml` sets Mauricio's team and `io.housetwinpatrol.lidar.mc`; on another Apple ID pick your team and a unique
   bundle id such as `io.<yourname>.lidar`.
3. Plug the iPhone in with a cable, unlock it and tap **Trust** this computer.
4. On the iPhone: Settings → Privacy & Security → **Developer Mode** → on → restart → confirm. (The switch appears after step 3.)
5. In Xcode's top bar pick your iPhone as the run destination and press **Run** (▶).
6. The first launch fails with "untrusted developer": on the iPhone go to Settings → General → **VPN & Device Management** →
   your Apple ID → **Trust**. Open *Twin LiDAR* again and allow the camera.

Free (personal team) apps stop opening after 7 days: plug in and press Run again.

## Record
The robot duck films from about 20 cm above the floor and finds its place from fixed objects and the look of each room,
so every zone needs a chest-height sweep, a slow duck-height turn and a few tagged objects. A checklist card guides this per zone.

1. Sign in with your account (see `../app/README.md` for the server side); Settings has the zone list and the tag labels.
2. Good light, doors open. Tap the zone you start in, then **Start**. Point at the floor for a moment: the HUD shows
   "Height 1.32 m" once ARKit has found it.
3. **Walls**: walk slowly (about half a normal pace), the phone at chest height and tilted a little down so the floor shows,
   and sweep every wall (25 s per zone at ≥ 0.9 m). When you go through a doorway, tap **Doorway**, then tap the new
   room's zone just after you step in. After 8 m walked in one zone the cyan banner asks "New room? Tap its name":
   tap the new room (or the same zone if you are still in it) so rooms don't merge into one.
4. **Low pass**: go down to duck height (phone below your knee, under 60 cm, camera level (not at the floor)) and turn slowly all the way round, holding still
   about a second in each direction. The 8-segment ring fills one 45° sector per direction (yellow = where you face now).
   The cyan banner says "Now go low" when the walls are done and "Slower" when you move too fast down there.
5. **Hold**: the crosshair's ring fills when the phone is still (1.5 s, then a light tick). Hold still on things you want sharp.
6. **Tags**: aim the crosshair at big things that never move (fridge, oven, sofa, bed, toilet, TV) and tap **Tag**, then pick
   the label (recording keeps running; Cancel drops it). About 3 per zone.
7. If the orange "Tracking limited" banner shows, slow down or point at a textured area until it goes away.
   The view starts dark and lights up where the LiDAR mesh has covered a surface (up to 5 m away): sweep until the
   walls and floor are bright. "Scanned: N m²" grows while you capture new surfaces; "Nothing new here" means move on.
   The veil can be turned off in Settings; it is feedback only and not saved.
8. **Stop** when every zone is done (keep it under 500 MB, the upload limit is 600 MB; the counter shows the size).
   A **Recording check** card scores it: zones used (red: one zone for over 15 m walked; orange: a zone under 4 m),
   duck-height directions per zone (green ≥ 6/8, orange 3–5, red < 3), tags per zone (green ≥ 2), length, frames and MB
   (red over 500 MB), and one verdict line. The numbers also go to `session.json` → `score`.
   **Review zones & tags** continues to the review; **Skip** packs the zip right away.
9. The review screen opens: drag on the colour strip or tap a thumbnail to find a moment. Tapping a zone relabels
   *from there to the next change* (moves a boundary earlier or adds a zone you forgot); with **Select range from here**
   it relabels just the selected range (moves a boundary later). **Save** (or **Skip**) packs the zip; with tags it is
   **Next** and the Tags step lists each tag with a crop of its frame: tap to change the label, swipe to delete, **Save**.
10. **Share / AirDrop** to the Mac, or **Upload** (sends it to `/upload?session=<id>&room=lidar&ext=zip`). The screen
   stays awake while uploading; on failure tap **Upload** again.
11. After the upload the app polls `/lidar/status?session=<id>` every 5 s (up to 10 min) and shows a **House built** card:
   each space's size, duck-height frames, doors, objects, landmarks and candidates, the curator model and its first
   warnings. **Open results** and **View in 3D** open the web pages in Safari. Reopening a session from the folder
   shows the card again if the server has built it.
   The folder icon lists earlier recordings (they survive app updates): tap one, then **Zones & tags** to review it again,
   share or upload it; swipe to delete. They also show in the Files app under *On My iPhone → Twin LiDAR*.

## Build the house
```
cd house-twin-patrol
uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python lidar/pipeline/build_house.py ~/Downloads/lidar-20261024-101500.zip
```
Output: `out/lidar/<session>/house.json, top.png, cloud.ply, report.json`. Test without a phone:
`uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python eval/synth_lidar.py`.

## Export format (`<session>.zip` → `<session>/…`)
| file | content |
|---|---|
| `session.json` | session id, device, fps, `"format": 2`, a short note on each format, and `score` (written on Stop): `{"zones": 3, "walked_m": {zone: m}, "sectors": {zone: 0–8}, "tags": {zone: n}, "duration_s", "frames", "mb"}` |
| `frames.jsonl` | one JSON line per frame (below), about 4 per second |
| `rgb/NNNNNN.jpg` | 960×720 JPEG (q 0.6) in the **sensor's landscape orientation** (not rotated for portrait) |
| `depth/NNNNNN.bin` | 256×192 Float16 little-endian, metres along the camera's viewing axis (z-depth), row-major |
| `conf/NNNNNN.bin` | 256×192 UInt8 ARKit confidence: 0 low, 1 medium, 2 high |
| `events.jsonl` | taps: `{"t","i","event":"room"\|"doorway","room","T"}` (`i` = next frame index) and tags (below) |
| `labels.json` | only if zones were fixed or there are tags: `{"ranges": [{"from": i, "to": i, "room": "Kitchen"}, …], "tags": [{"i", "label", "uv", "room"}, …], "note"}` (inclusive frame `i`). The app also rewrites `room` in `frames.jsonl`; `build_house.py` lets `labels.json` win. `tags` (reviewed: relabelled or deleted) wins over the tag events, even when empty |

Tag event (one per tagged object; `i` = the frame saved right at the tap, `T` = its pose, `uv` = the crosshair in that
frame's saved JPEG, normalised 0..1 from the top-left; usually about `[0.5, 0.5]`):
```json
{"t": 41.2, "i": 163, "event": "tag", "label": "Fridge", "room": "Kitchen", "T": [16 numbers], "uv": [0.5, 0.5]}
```

Frame line:
```json
{"i": 12, "t": 3.01, "rgb": "rgb/000012.jpg", "depth": "depth/000012.bin", "conf": "conf/000012.bin",
 "rgb_wh": [960, 720], "K": [fx, fy, cx, cy], "depth_wh": [256, 192], "K_depth": [fx, fy, cx, cy],
 "T": [16 numbers, row-major 4×4], "tracking": "normal" | "limited:<reason>" | "notAvailable", "room": "Kitchen", "h": 1.32}
```
- `h` (format 2) = camera height above the floor in metres (ARKit's lowest floor plane, 0.01 m), `null` until a floor is
  found. Duck-height frames have `h` < 0.5.
- `K` is for the saved JPEG and `K_depth` for the depth map (ARKit's intrinsics scaled from the 1920×1440 capture).
- `T` is ARKit's camera-to-world transform. ARKit world: metres, **y up** (gravity), origin where the session started.
  ARKit camera: x right, y up, z backwards (it looks along −z). For OpenCV (x right, y down, z forward) use
  `T_cv = T @ diag(1, -1, -1, 1)`. The pipeline then maps the world to z up: `(x, y, z) = (x_ar, -z_ar, y_ar)`.

## How the pipeline works
1. Keep tracking-normal frames and high-confidence depth up to 5 m.
2. Fuse every frame into an OpenCV 5 `cv.Volume` (HashTSDF, 2.5 cm voxels). Back-project with `cv.depthTo3d`
   and get per-pixel normals with `cv.RgbdNormals`, then put the points in a 2.5 cm voxel grid with the mean position,
   normal, colour and the majority room label of the frames that saw each voxel.
3. Floor = the lowest dense layer of upward-facing voxels (ARKit is gravity-aligned, so heights are direct).
   Manhattan angle = the 4θ circular mean of wall normals.
4. Each room = a rectangle: per side, the farthest dense band of wall voxels that face into the room (its label, 0.3–2.0 m high).
   Facing walls of two rooms less than 40 cm apart become one shared wall, as thick as the gap.
5. Doors: where the walk crosses a wall line near a room-label change. The width is the empty gap in that wall's points.
6. Objects: YOLOX (`service/v0/scan.py: detect`) on every 2nd frame, after rotating it upright using gravity. Each box
   is lifted to 3D with the depth: floor and room-wall points are dropped and the biggest connected piece on a 4 cm
   top-down grid is kept. Pieces of one class that touch merge across frames into an oriented box (`cv.minAreaRect`, height = top).
7. Origin = the largest room's corner, x along its wall. `cloud.ply` = the TSDF surface raycast from every 8th pose.
