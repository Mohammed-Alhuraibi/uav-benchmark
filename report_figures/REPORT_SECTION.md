# Object Detection

For detecting the target UAV, we benchmarked four YOLO models on a 3,558-image subset of the MMFW-UAV dataset [1] under identical training conditions. The table below summarizes our results.

> [INSERT fig_model_comparison_table.png]

We selected YOLOv11s+P2 because it achieved the highest recall (99.7%) and precision (99.4%) — the two metrics that matter most for the competition, where a missed target fails the lock-on and a false detection costs -30 points. The "+P2" refers to an extra detection head at stride 4, which lets the model detect targets as small as ~4 pixels. This is important since over 40% of our test images contain small, distant UAVs.

> [INSERT fig6_p2_head_architecture.png]

For tracking, we use ByteTrack to assign a persistent ID to the target across frames. This feeds into a lock-on state machine that counts 4 seconds of continuous detection before confirming a lock, with 200ms dropout tolerance for brief detection gaps.

---

## References

[1] Liu, Y. et al. "MMFW-UAV dataset: multi-sensor and multi-view fixed-wing UAV dataset for air-to-air vision tasks." *Sci Data* 12, 176 (2025).
