# Dataset Setup

This directory is **gitignored**. You need to populate it on the target machine.

## Expected Structure

```
data/
├── images/
│   ├── train/    (2846 images)
│   ├── val/      (358 images)
│   └── test/     (354 images)
└── labels/
    ├── train/    (2846 .txt files)
    ├── val/      (358 .txt files)
    └── test/     (354 .txt files)
```

**Total**: 3,558 matched image/label pairs.
**Split**: 80/10/10 stratified by sortie bucket (12 MMFW-UAV sorties).
**Class**: Single class `0` = `Fixed_Wing_UAV`.
**Label format**: YOLO normalized (class cx cy w h).

## How to Populate

Download the clean dataset from your cloud storage and extract it here so that
`data/images/train/` and `data/labels/train/` contain files directly (no extra
nesting).

## Quick Verify

After placing the data, verify integrity:

```bash
python src/benchmark.py --verify-only
```

Expected output:
```
  train :  2846 pairs, 0 orphan imgs, 0 orphan lbls [OK]
  val   :   358 pairs, 0 orphan imgs, 0 orphan lbls [OK]
  test  :   354 pairs, 0 orphan imgs, 0 orphan lbls [OK]
```

## Source

MMFW-UAV (Nature Scientific Data, 2025) — processed through the team's
ad-hoc conversion pipeline. See `docs/dataset-forensic-report.md` for
full provenance analysis.
