# UAV Detection Benchmark Results

## Comparison Table

| experiment       |   mAP50 |   mAP50-95 |   precision |   recall |     f1 |   tiny_images |   tiny_ratio |   speed_ms |   fps |
|:-----------------|--------:|-----------:|------------:|---------:|-------:|--------------:|-------------:|-----------:|------:|
| yolo26s          |  0.9939 |     0.8505 |      0.9803 |   0.9835 | 0.9819 |           143 |        0.404 |       33.5 |  29.9 |
| yolo11s_p2       |  0.9943 |     0.8365 |      0.9937 |   0.9972 | 0.9955 |           143 |        0.404 |       33   |  30.3 |
| yolo11s_baseline |  0.9843 |     0.8343 |      0.9914 |   0.9813 | 0.9864 |           143 |        0.404 |       32.5 |  30.8 |
| yolo26n          |  0.992  |     0.8155 |      0.9708 |   0.9661 | 0.9684 |           143 |        0.404 |       33   |  30.3 |


## Key Findings

- **Best mAP@0.5:0.95**: yolo26s (0.8505)
- **Best Recall**: yolo11s_p2 (0.9972)
- **Fastest**: yolo11s_baseline (32.5 ms/img)
- **Tiny object images in test set**: 143 (40.4%)

## Generated Artifacts

- `comparison_table.csv` — full metrics table
- `loss_curves.png` — training loss overlay
- `metric_curves.png` — mAP progression overlay
- `visual_grid.png` — side-by-side predictions
