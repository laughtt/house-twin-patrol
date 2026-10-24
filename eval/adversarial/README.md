# Adversarial tests for the zero-sticker work

These tests try to break the zero-sticker localizer (`agent/localize.py`) and the landmark pipeline (`twin/curate.py`,
`twin/landmarks.py`, `twin/scan_io.py`, `lidar/pipeline/build_house.py`). A failing test means the code broke on that
input. The criteria were written before the first run and were not tuned afterwards. Production code is not touched.

```
# fast set (~2 min on 10 cores): unit cases, curate/tags, scan builds, 8 core route cases x 4 seeds
uv run -q --python 3.12 --with pytest --with opencv-python-headless==5.0.0.93 --with numpy python -m pytest eval/adversarial -q
# full set: every route case x 8 seeds + MuJoCo cross-check (~20 min)
uv run -q --python 3.12 --with pytest --with mujoco --with opencv-python-headless==5.0.0.93 --with numpy python -m pytest eval/adversarial -q --slow
```
`ADV_SEEDS=N` overrides the seed count. At the end, pytest prints a PASS/FAIL table with the numbers and writes it to
`out/adversarial/report.json`.

Inputs: `out/current/house.json`, the real LiDAR kitchen and living room (read at run time, never copied). Scan cases
use `out/lidar/synth-multi.zip`; make it with `eval/synth_lidar.py --house multi`. MuJoCo cases use the scene that
`eval/localize_test.py` writes.

## Files
- `sim.py`: a fast stand-in for `eval/localize_test.py`. It uses the same route, odometry error model and simulated
  detector, but the ToF is ray-cast on the world's own grid instead of MuJoCo. That makes it optimistic: about 3 cm after
  lock, against 6–9 cm in MuJoCo. A failure here is therefore a real failure. Every input can be tampered with per frame.
- `test_routes.py`: whole-route cases (kidnap, symmetry, moved landmarks, lying detector, broken sensors, room colour).
- `test_localize_unit.py`: map edge cases and numerical robustness, driven directly.
- `test_curate.py`: hostile model answers (with `curate.ask` or `boto3` faked), prompt contents, malformed user tags.
- `test_scans.py`: `build_house.build` on hostile variants of the synthetic scan.
- `test_mujoco.py` (slow): key route cases again with MuJoCo's ToF.

## Pass criteria
Route terms:
- **lock**: error under 0.3 m and held for 2 s.
- **std**: locks within 20 s in at least 75 % of seeds, and the median error after lock is at most 0.15 m.
- **re-lock N s**: after the event, locks again within N s in at least 75 % of seeds.
- **holds X**: error under 0.3 m in at least X of the frames, in at least 75 % of seeds.

| # | case | what it does | pass when |
|---|---|---|---|
| – | baseline | unknown start, ToF + objects | std |
| 1 | kidnap_other_room / kidnap_same_room | carried at 30 s to frame 330 (corridor into the living room) or to frame 240 (2.4 m away in the same room); odometry reports nothing | re-lock 15 s |
| 1 | kidnap_lights_off | the same-room kidnap with no detections | re-lock 15 s |
| 2 | twin_rooms_no_cue | house with 180° symmetry: two identical rooms | room B keeps at least 10 % of the weight (no collapse) in 75 % of seeds |
| 2 | twin_rooms_tv_cue / colour_cue | the same house, but a TV only in room A / different room colours | holds 0.9 over the last 10 s |
| 2 | lookalikes_6chairs_8counters | 6 extra chairs and 3 extra counters on the map | std |
| 3 | fridge_removed, sofa_moved_2m, movables_shuffled_1m | today's world differs from the scan-day map | std |
| 3 | fixed_fridge_moved_3m | a "fixed" landmark moved | std, and at most 5 % of after-lock frames over 1 m off |
| 3 | all_fixed_moved_1m | renovation: every fixed landmark moved 1 m | std with a median of at most 0.2 m |
| 4 | fp_burst_person_20s | 3 junk detections of any class per frame for 20 s | std, and holds 0.85 during the burst |
| 4 | couch_table_swapped, couch_called_bed | systematic wrong class | std |
| 4 | tv_shows_fridge | a TV keeps showing a fridge (phantom of a fixed class) | std, and at most 5 % of frames over 1 m off |
| 4 | bad_detector | 50 % hits, 30 % wrong class, 50 % false sightings | std with a median of at most 0.2 m |
| 4 | lights_off_unknown_start / known_start | no detections | std / holds 0.9 |
| 4 | stuck_phantom_fridge | the only detection is a fridge at 0°, 1 m, every frame (known start) | holds 0.8 |
| 5 | tof_no_data_all_254 | ToF never has data (objects only) | std with a median of at most 0.3 m |
| 5 | tof_no_target / covered / stuck (from 20 s) | ToF reads all "no target", all 15 mm, or one frozen frame | holds 0.8 after the failure |
| 5 | wheel_slip_5s | stuck against a wall for 5 s while odometry says 0.25 m/s | re-lock 5 s after the slip, and worst error at most 0.5 m |
| 5 | heading_bias_10deg_per_m | odometry heading bias of 10°/m | std |
| 5 | odo_jump_2m / odo_nan_once | one odometry reading 2 m too long / NaN | re-lock 5 s |
| 6 | colour_true / lights_changed / says_wrong_room / identical_rooms / missing_2_rooms | `update_room` with a correct colour, hue shifted 2 bins, another room's colour, all rooms the same, `colour_hs` missing | std |
| 7 | map_* | no landmarks; landmarks without cls or weight; no rooms; a room with no free cell; negative coordinates | no exception; same ToF as the unshifted map |
| 7 | real_map_landmarks_inside_walls | every landmark of the real house lies inside the walls | none outside |
| 8 | nan/inf/negative bearing and range, nan odometry, nan ToF zone | poisoned inputs (known start) | particles finite, estimate within 0.3 m; a NaN zone keeps at least 0.9 of the weight on the truth |
| 8 | n_1_2_10_particles, all_zero_weights, bearing_plus_4pi | degenerate filters and inputs | no exception, same result |
| 8 | 30_detections_per_frame_time | 30 detections of a fixed class, 2000 particles | `update_landmarks` at most 200 ms (one 5 Hz frame) |
| 8 | estimate_grid_split | 2/3 of the particles spread around the truth, 1/3 in a tight blob 3 m away | estimate within 0.3 m of the truth |
| 8 | estimate_hysteresis_switch | 50/50 split with the last estimate on the wrong side, true ToF arriving | switches within 5 frames |
| 9 | curate_* | merge cycles and chains; unknown or non-string ids; answer not a dict; keep="false"; mobility outside the list; empty, 500-char, number or null label; duplicate entries; all keep=false; a user tag merged away; exception; reply without a tool call | no crash; strings are not read as booleans; labels are 1–64 char strings; user tags survive; fallback to the config lists |
| 9 | curate_slow_call_bounded | boto timeouts | worst case at most 120 s |
| 10 | prompt_user_label_* | injection, braces, emoji, 2000 chars in a user label | reaches the prompt only as a JSON string, at most 64 chars |
| 10 | tag_* / tags_* | `i` null or missing, label a number or whitespace, uv short, strings, outside [0,1] or null, unknown frame, empty space, duplicate tags, `tags: []` vs missing vs null | no crash; nothing created or renamed from a bad tag |
| 11 | scan_* | zero detections, all or half of the frames with limited tracking, no events/labels, no low frames, tags end to end, tag on a missing frame | builds (or fails with a clear message); one user fridge; nothing outside the walls |
