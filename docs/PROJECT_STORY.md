# House Twin Patrol
**One walk with your iPhone. A robot that knows your home. No stickers.**

## Inspiration
Here's a story every home-robot owner is about to live.

You unbox a small robot and set it down in the living room. It looks around, decides it's in the bedroom, and drives
confidently into the sofa. It isn't broken. It just has no idea where it is: every home is different, the robot never
practised in yours, and the usual fixes are the ones nobody wants: QR stickers on every wall, beacons in every room,
or an afternoon of the robot mapping your house by itself.

I lived a version of this. My first robot brain worked with stickers. Then I took the stickers off, because nobody
puts QR codes on their walls, and measured honestly: on one patrol through a synthetic test home it spent **62 % of the time sure of the wrong
place**. On the live robot it took 61 seconds before it was sure and right, and even then it was confidently wrong
18 % of the time.

The robot is Pollen's Microduck: a little duck with an 8×8 depth sensor (64 pixels) and a small camera. It ships at the
end of 2026, so nobody has one yet. But everybody already has the thing that can map a home in five minutes: a phone
with LiDAR. So I built House Twin Patrol.

## What it does
House Twin Patrol turns one iPhone walk into a digital twin of your home, built with OpenCV 5 on AWS, and gives the
robot a brain that finds itself, patrols every room and asks a person when it isn't sure.

| What happens | What House Twin Patrol does |
|---|---|
| You walk through your home with the app | Records LiDAR depth, colour and the ARKit pose; you tap each room as you enter |
| You tap Upload | The server on AWS Graviton builds the twin with OpenCV 5: floor, walls, doors, rooms, furniture, 3D |
| The map is ready | My maps shows it: floor plan with room sizes, a 3D view, a notification |
| The robot is switched on somewhere unknown | It turns once on the spot and matches that 360° scan against every place in the map: sure in about 8 seconds |
| The robot walks to a room | Every belief is checked against the depth sensor; if the walls stop fitting, it stops and looks again |
| A strong guess appears across the house | Treated as a false positive: the robot keeps the position it trusted |
| The camera sees the room | OpenCV votes for the room; if that changes a decision, it is logged and the route is planned again |
| The robot stays lost for 20 seconds | It asks a person: "Keep looking for the bedroom?" |
| It reaches a goal, but the camera disagrees | "I think I'm in the bedroom, but the camera's colours say kitchen. Say I've arrived?" |
| Someone wants to know why it did that | Every step is in a trace; a film replays it with the robot's eye, the map and the brain in words |

## Finding itself with 64 pixels: the spin match
A single 8×8 depth frame says almost nothing. A full turn says a lot. While the duck turns on the spot, its readings
are stacked into one 360° scan (120 slices of 3°). After the turn, that scan is compared with what the map says you
would see from **every free spot in the house, at every heading**: first on a 20 cm grid, then 10 cm around the best
places. It only counts as found when the best place clearly beats the runner-up. Directions where the sensor saw
nothing are evidence too: "open space this way".

## Verify before trust
Being sure fast is easy. Being sure and right is the job.
- **Fit check:** from where the robot thinks it is, the map predicts what each depth zone should read. If the readings
  stop fitting for 3 seconds, the robot is lost and looks again. A reading *shorter* than the map is not counted
  against it: that's a person, a bag, or the floor when the walking head tips down. You can't see through a wall,
  so only *longer* readings contradict.
- **Jump check:** once a position is trusted, a new estimate more than 1 m + 10 % of the distance walked away is a
  false positive. The guesses near the trusted position stay the answer, and the robot keeps going.
- **Arrival needs a fresh fit:** in the middle of a 132 m² living room nothing is in range, so the robot may drive on
  what it knows but may not declare "I'm here".

## Agentic Vision: OpenCV changes what the robot does
Five times a second: sense → locate → see → decide → ask → act. OpenCV 5 reads every camera frame (`cvtColor`,
`inRange`, `calcHist`): which room do these colours look like? That vote moves the robot's belief, and when a frame
changes the decision, the trace records it (here, from the live walk, the frame moved its belief to the living
room, where it really was):

```json
{"t": 0.41, "event": "vision_decision", "cause": "room colour (bedroom)",
 "before": {"mode": "look", "room": "bedroom_2", "conf": 0.06},
 "after": {"mode": "look", "room": "living_room", "conf": 0.07}, "goal": "bedroom"}
```

And when the evidence isn't enough, a person decides: "Lost for 20 s (50 % sure, best guess: living room, the camera's
colours say: bedroom). Keep looking for bedroom 2?" Say no, and it stops. Every answer is logged.

## Key features
- **Zero stickers:** no markers, no beacons; the map comes from one walk.
- **Real robot software:** Pollen's `robotd` and its trained walking policy (ONNX, 50 Hz) drive the duck; my brain
  only sends velocity commands and sees what the robot sees.
- **An app people can use:** sign in, record, upload, watch it build, open the floor plan and 3D. Open the Xcode
  project and press Run.
- **Observability:** a trace of every step and decision, and a film with the chase camera, the robot's eye, the map
  (truth vs belief) and banners whenever OpenCV changed the plan or the robot asked.
- **An honest test bench:** 22 homes × several starts × a perfect and a rough sensor, the old brain against the new one,
  on the same seeds.

## How I built it
| Part | What it is |
|---|---|
| iPhone app (SwiftUI, ARKit) | Records the walk, room taps, recording check and review; login with tokens in the Keychain; My maps |
| Server (FastAPI on AWS EC2 Graviton4) | Caddy HTTPS, scrypt passwords, signed 30-day tokens, rate limiting; runs the twin build per upload; Amazon Bedrock (Claude) for the room-measure agent |
| Twin build (OpenCV 5) | Depth fusion (`cv.Volume` TSDF, `RgbdNormals`) → floor → walls (`morphologyEx`, `HoughLinesP`) → doors → rooms (`connectedComponents`, `findContours`, `approxPolyDP`) → landmarks (CLIP in `cv.dnn`) → `house.json` and a MuJoCo scene |
| The brain (`agent/loc/`) | One module per job: the map and fast ray casts (`distanceTransform`), depth reading, spin match, fit check, landmarks; an orchestrator with the same API as before |
| The agent loop (`agent/walk.py`) | OpenCV on the camera, `vision_decision` events, A* routes, the two approval moments |
| Evaluation | `eval/selfloc.py` (22-house sandbox), `eval/vision_walk.py` (camera vs no camera), 73 fast adversarial tests, live walks, `agent/film.py` |

Everything runs with one launcher: `scripts/demo.sh offline | live | server | app`. A built sample home ships in the
repo, so anyone can test it in a minute without a phone.

## Challenges I ran into
- **The robot turned on the spot for 24 minutes.** The sandbox was perfect; the live robot never left its start. The
  trained walking policy sways the body about 58 cm per turn while ending 2 cm from where it began, and my scan threw
  itself away whenever the robot "moved" more than 10 cm. Fix: count net displacement, not path length.
- **Fit 0 at the right pose.** Walking tips the head down, and the depth sensor hits the floor 1.5–2 m ahead. The pose
  was 2 cm off, and the fit said "completely wrong". Fix: a reading shorter than the map is no evidence.
- **My first jump check made things worse.** It reset and re-turned on every jump. Live, the estimate flipped about 20
  times in 7 minutes, the robot got lost for 150 seconds, and the walking policy fell over mid-turn. I replaced it with
  "far estimate = false positive, keep going".
- **Stickers hurt once the brain was good.** Letting an ArUco sighting end the search cost goals (48 → 32 of 48),
  because the marker's two possible poses were told apart by a guess that was still random. Markers are now off.
- **The camera's room vote is weak live.** Single live frames name the right room 46 % of the time; the colour
  fingerprints come from the twin, not the robot's camera. So the camera votes and checks; it never decides alone.
- **No hardware.** Nobody has a Microduck yet, so the live test is Pollen's real software and walking policy on a
  simulated body inside the twin of a real scanned home.

## Accomplishments I'm proud of
- **Live: sure and right in 8 seconds instead of 61**, never sure of a wrong place (was 18 % of the time), four rooms
  out of four, each reached within 30 cm.
- **22 test homes, rough sensor:** runs that locked onto the wrong place 89 → 3 of 96, false arrivals 56 → 0, goals
  reached 51 % → 94 %, bumps 1390 → 305.
- **Zero stickers**, end to end: from an iPhone walk to a robot patrol, with a working app and a live web endpoint.
- **Four bugs only the real robot could show**, found by recording a live spin and replaying it offline.

## What I learned
- **A clean simulator is not a working robot.** Every live-only bug was physical: sway, head pitch, falls.
- **"Verify before trust" beats "be sure fast".** The new brain takes about 11 seconds to say "found" (the old one
  said it in 4, and was often wrong). Almost never wrong is worth seven seconds.
- **Not seeing is not evidence against.** Open space and short readings say nothing about a wrong position; treating
  them as evidence made the robot doubt itself constantly.
- **Asking a person is a feature only if it's rare.** Counting every look as "lost" asked 7 times a walk; asking only
  when it's really lost, 2.

## What's next
- Camera place recognition for rooms that look alike to the depth sensor, and glass walls it sees through
- Room-colour fingerprints from the robot's own camera instead of the twin's
- The real Microduck as soon as it ships
- COOL on Graviton for the twin build, with a measured baseline
- Per-user accounts and S3 storage for scans
- A household mode: "go check the bedroom", patrol schedules, alerts

## Built With
OpenCV 5 · Python · NumPy · MuJoCo · Pollen Microduck (`robotd`, `tofd`, `mediad`, ONNX walking policy) · Swift ·
SwiftUI · ARKit · iPhone LiDAR · FastAPI · Caddy · AWS EC2 Graviton4 · Amazon Bedrock (Claude) · uv · ffmpeg · XcodeGen
