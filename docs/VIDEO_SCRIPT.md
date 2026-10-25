# 3-minute video script

About 400 words of voice-over (~2:50 at 140 words a minute). The rules require the team on camera: the first 10 s. Every number below is measured (DEMO.md).
Clip times in the live film refer to `out/walk/selfloc_live_camera/walk_2x.mp4` (2× speed, 3:34 long).

## Shot list and voice-over

### 0:00–0:10 · The team (required by the rules)
**Show:** Jose Mauricio Carpio on camera (webcam or phone), at home.
**On screen:** "Jose Mauricio Carpio · House Twin Patrol · OpenCV AI Competition 2026"
**Say (own voice):**
> Hi, I'm Jose Mauricio Carpio, and this is House Twin Patrol.

### 0:10–0:22 · Hook
**Show:** live film 0:00–0:06 (the duck turns on the spot, the "sure" bar fills), then a title card: *House Twin Patrol*.
**On screen:** "No stickers. No GPS. One iPhone walk."
**Say:**
> This little robot has never been told where it is. No stickers on the walls, no beacons. In eight seconds it will know
> which room it's in, to within centimetres, using a map we built with one walk and an iPhone.

### 0:22–0:45 · The app
**Show:** `docs/app_screens.jpg`, panning left to right (sign in → My maps → floor plan → 3D). Better: a screen
recording on a LiDAR iPhone of a short recording, Upload, and My maps switching from "Building" to "Ready".
**On screen:** "Record → Upload → Ready"
**Say:**
> It starts with our app. You sign in, walk through your home once, tapping each room as you enter, and upload. The
> server builds the map while you wait, and it appears under My maps: a floor plan with room sizes, and a 3D view.

### 0:45–1:15 · OpenCV 5 builds the twin
**Show:** `docs/map_build.gif` (floor → walls → doors → rooms), then `docs/house_3d.gif` turning.
**On screen:** the stage captions; "OpenCV 5: TSDF fusion · morphology · Hough lines · connected components · contours"
**Say:**
> OpenCV 5 does the heavy lifting. It fuses half a million LiDAR points, then finds the floor, the walls that stand a
> metre tall, the doors between them, and finally the rooms, as the floor falls apart along its walls. The result is a
> digital twin of the house: every room, door and piece of furniture.

### 1:15–2:15 · The live patrol
**Show:** live film, cut together:
- 0:00–0:05: the turn on the spot (a "camera changed the plan" banner flashes at 0:00: it corrected the room to living room), "sure of its position" going green at 0:04 (7.8 s of robot time).
- 0:20–0:32: walking to the bedroom, the arrival at 0:31 ("bedroom, 18 cm off").
- 1:05–1:12: looking around near bedroom 2; the "asked a person: keep looking?" banner at 1:08.
- 1:30–1:38: arriving in bedroom 2 at 1:37.
- 3:28–3:34: the last arrival, living room, 13 cm off.

**On screen:** "Pollen's real robot software · trained walking policy · simulated body in the twin"
**Say:**
> Now the patrol. This is Pollen's real robot software and its trained walking policy, driving a simulated body inside
> the twin. Our brain only sees what the robot sees: its steps, an eight-by-eight depth sensor and its camera.
> To find itself, it turns once on the spot and matches that scan against every place in the map. Then it checks every
> belief: if the depth readings stop fitting the map, it stops and looks again. And once it trusts a position, a sudden
> jump across the house is treated as a false positive.
> The camera runs OpenCV on every frame and votes for the room it sees. When the robot stays lost, it asks a person
> before carrying on. Four rooms out of four, each reached within thirty centimetres.

### 2:15–2:40 · The agent loop
**Show:** `docs/agent_loop.gif` full screen, then `docs/architecture.svg`.
**On screen:** "sense → locate → see → decide → ask → act, five times a second"
**Say:**
> Five times a second the agent senses, works out where it is, looks, decides and acts. What OpenCV sees can change the
> route, and when the camera disagrees that it has arrived, the robot asks a person to confirm. Every decision is logged,
> so you can replay why it did what it did.

### 2:40–3:00 · Results and close
**Show:** a results card, then the repo URL.
**On screen:**
- "Sure and right: 61 s → 8 s (live)"
- "Sure but wrong: 18 % → 0 % of the time (live)"
- "22 test houses: wrong locks 89 → 3 of 96 runs, false arrivals 56 → 0 (rough sensor)"
**Say:**
> Against our first version, it finds itself eight times faster and was never sure of a wrong place in the live walk. Across
> twenty-two test houses, wrong locks fell from eighty-nine runs to three. What's next: telling apart rooms that look
> the same, and a real robot in a real home. House Twin Patrol: one walk, and your home has a guardian.

## Honest notes for the edit
- Say "simulated body" whenever the live film is on screen: the robot software and walking policy are real, the body,
  sensors and house are simulated.
- In this live walk the person's answers were automatic (logged as "auto"). For a real person answering, run
  `scripts/demo.sh live --ask` and record the terminal.
- The arrival check ("say I've arrived?") was added after this walk; it is shown offline (`eval/vision_walk.py`: 3
  questions in 48 arrivals). Don't show it as part of this walk.
- The camera's room vote is weak live (right 46 % of single frames). The voice-over says "votes", not "recognises".

## Recording tips
- App: `xcrun simctl io booted recordVideo app.mp4` records only the simulator screen. Recording a real walk needs a
  LiDAR iPhone (Control Centre → Screen Recording).
- Film clips: `ffmpeg -ss 0:20 -to 0:32 -i out/walk/selfloc_live_camera/walk_2x.mp4 -c copy clip_bedroom.mp4`.
- GIFs into video: `ffmpeg -i docs/map_build.gif -movflags faststart -pix_fmt yuv420p map_build.mp4`.
