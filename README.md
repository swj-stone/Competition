# Competition Projects
**English** / [简体中文](./README_CN.md)

This repository collects selected code and solution notes from several competitions. Each entry below describes the problem, the approach where available, and where to find the relevant materials.

| Directory | Project | Problem and approach |
|---|---|---|
| [AIJ2026](./AIJ2026/) | [Coordination Referee](./AIJ2026/Coordination_Referee/README.md) | Where did a team of agents first go wrong, and did it still achieve its goal? The solution ranks candidate turns using evidence from messages and handoffs, then identifies the root cause and assesses goal completion separately. [Python implementation](./AIJ2026/Coordination_Referee/src/) |
| [AIJ2026](./AIJ2026/) | [Guardian of Truth](./AIJ2026/Guardian_of_Truth/) | Given an agent’s answer and the context it received, the challenge is to detect claims that contradict the context or lack support in it. |
| [GOAI2026](./GOAI2026/) | [CampusMatch](./GOAI2026/README_EN.md) | Specialized agents match job requirements to passages in a student’s application materials. The report shows the evidence for each match and flags requirements the documents leave unanswered. [Runnable demo](https://github.com/swj-stone/GOAI/tree/campusmatch-demo/demo) |
| [IMCCHALLENGE](./IMCCHALLENGE/) | [IMC Challenge 2026 · Problem B](./IMCCHALLENGE/README.md) | Simplify a triangle mesh with up to 1.1 million vertices while preserving its appearance. The solver removes most vertices using weighted quadric error metrics, then checks the visual impact of further edge collapses from six camera views. [C++ solver](./IMCCHALLENGE/final.cpp) |
| [Phone-detect](./Phone-detect/) | [PhoneGuard-P2](./Phone-detect/README_EN.md) | Locate mobile phones in exam-room images and video frames. The model adds a higher-resolution P2 detection head to YOLOv8 to capture small phones and uses staged training to handle blur, low light, and partial occlusion. |
