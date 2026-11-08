# 竞赛代码与解题思路
[English](./README.md) / **简体中文**

本仓库收录了几项比赛的部分代码与解题思路。下表按项目列出赛题、已有方案及对应的说明文档、源码或演示入口。

| 目录 | 项目 | 赛题与思路 |
|---|---|---|
| [AIJ2026](./AIJ2026/) | [Coordination Referee · 多智能体协作故障诊断](./AIJ2026/Coordination_Referee/README_CN.md) | 一组智能体从哪一轮开始配合出错，最后又是否完成了目标？方案从消息和任务交接记录中提取证据，给可能的故障起始轮次排序，再分别判断故障根因和任务完成情况。[Python 源码](./AIJ2026/Coordination_Referee/src/) |
| [AIJ2026](./AIJ2026/) | [Guardian of Truth · 回答可信性检测](./AIJ2026/Guardian_of_Truth/) | 给定智能体的回答及其收到的上下文，识别回答中与上下文矛盾或缺少依据的陈述。 |
| [GOAI2026](./GOAI2026/) | [CampusMatch · 高校可信求职协同基座](./GOAI2026/README.md) | 多个智能体将岗位要求逐项对应到学生申请材料的原文，说明每项匹配的依据，并标出材料尚未覆盖的要求。[可运行的 Demo](https://github.com/swj-stone/GOAI/tree/campusmatch-demo/demo) |
| [IMCCHALLENGE](./IMCCHALLENGE/) | [IMC Challenge 2026 · B 题](./IMCCHALLENGE/README_CN.md) | 在尽量保持外观的前提下，简化最多含 110 万个顶点的三角网格。求解器先用加权二次误差度量进行边折叠，移除大部分顶点，再从六个视角检查后续折叠对渲染结果的影响。[C++ 求解器](./IMCCHALLENGE/final.cpp) |
| [Phone-detect](./Phone-detect/) | [PhoneGuard-P2 · 考场手机检测](./Phone-detect/README.md) | 在考场图像和视频帧中定位手机。方案在 YOLOv8 上加入分辨率更高的 P2 检测头，以识别画面中较小的手机，并通过分阶段训练处理模糊、弱光和局部遮挡。 |
