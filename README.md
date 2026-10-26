# House Twin Patrol

Small home robots like Pollen's open-source **Microduck** are coming, but a new robot in a home is lost:  We asked a simple question: *what if you could just show the robot your home first?* Everyone already carries a phone with a camera, and many have LiDAR. So we built a way to film a house once and give any robot a ready-made, measured map it can find itself in. With an agentic loop the robot can decide - follow orders locate itself - dynamcly take decisions and achieve the goal we propose in 98% out of the cases. 

Vision** (OpenCV AI Competition 2026). Team: Jose Mauricio Carpio.

**Technical report: [REPORT.md](REPORT.md). Project story: [docs/PROJECT_STORY.md](docs/PROJECT_STORY.md). Test it without an iPhone: [TESTING.md](TESTING.md).** **Run it: `scripts/demo.sh offline | live | server | app`. Full walkthrough, commands, results and limits: [DEMO.md](DEMO.md).** The agent loop and its trace: [agent/README.md](agent/README.md).

![Architecture](docs/architecture.svg)

## The app: walk, upload, see your map
![Sign in, My maps, floor plan, 3D, dark mode](docs/app_screens.jpg)

## The map, built with OpenCV 5: floor → walls → doors → rooms
![Map construction stages](docs/map_build.gif)

## The first 3D map of the house (LiDAR point cloud, cut at 1.8 m)
![3D map of the house](docs/house_3d.gif)

## The robot walk (live: Pollen's robotd + walking policy, simulated body in the twin; 12× speed)
![Live robot walk](docs/robot_walk.gif)

## The agent loop, with the real values of that walk
![Agent loop](docs/agent_loop.gif)

| live walk, 4 rooms | old filter | final (ToF + camera, zero stickers) |
|---|---|---|
| rooms reached | 4/4 | 4/4 |
| until sure and right | 61 s | 7.8 s |
| time sure but more than 50 cm off | 17.7 % | 0 % |
| camera: decisions changed / questions asked | – | 4 / 2 |
