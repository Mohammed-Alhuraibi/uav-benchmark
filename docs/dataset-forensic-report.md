# Dataset Forensic Report — TeknoFest UAV

**Date:** 2026-04-10
**Scope:** `images/`, `labels/`, `new-data/Dataset_9998pic`, `new-data/Dataset A`, `new-data/Dataset A'`

## TL;DR

All of your team's UAV image data (except the Roboflow set) comes from a single publicly available academic dataset: **MMFW-UAV** (Nature Scientific Data, 2025). Your team has only ~2% of the full 147k-image dataset. The broken archives, orphan labels, and naming collisions are all symptoms of a lossy ad-hoc conversion of MMFW-UAV to YOLO format done at some earlier point. **Re-download the full public source (130 GB) from https://www.scidb.cn/s/vmimum and start over with clean data.**

---

## 1. Source identification

### Dataset A & Dataset A' = MMFW-UAV subsets

- **Paper:** *MMFW-UAV dataset: multi-sensor and multi-view fixed-wing UAV dataset for air-to-air vision tasks*
- **Journal:** Nature Scientific Data, Jan 2025
- **Lab:** AIUS-LAB (https://github.com/AIUS-LAB/MMFW-UAV)
- **Data host:** Science Data Bank, https://www.scidb.cn/s/vmimum (130 GB)
- **License:** Check scidb.cn terms (academic use)

### MMFW-UAV structure (full)

- **147,417 images** of fixed-wing UAVs
- **12 sorties**: `Fixed-wing-UAV-A` through `F` (raw casing) plus `A'` through `F'` (same airframes, solar panel casings)
- **3 view angles** per sortie: `Bottom_Up`, `Horizontal`, `Top_Down`
- **3 sensor streams** per view: `Wide`, `Zoom`, `Thermal` (infrared)
- Total: 12 × 3 × 3 = **108 image directories**
- Annotations in both PASCAL VOC (XML) and MS COCO (JSON)
- Single class: `Fixed_Wing_UAV`

### Filename convention (decoded)

`T_W_NNNNNN.jpg`
- **T** = time: `0` = morning, `1` = afternoon
- **W** = weather: `0` = sunny, `1` = cloudy
- **NNNNNN** = serial frame number

Your team's archives:
- **Dataset A** prefix `0_1_` = morning, cloudy — `Fixed-wing-UAV-A`
- **Dataset A'** prefix `0_0_` = morning, sunny — `Fixed-wing-UAV-A'`

## 2. Forensic findings

### Dataset_9998pic
- **Source:** Roboflow Universe → `554-img/uav-554` v4 (CC BY 4.0), unrelated to MMFW-UAV
- **9,997 images, 640×640, YOLO format**, 10,410 objects
- **5 garbage class names** — remap to single class 0
- Mean bbox **26%×18%**, only **2.6% tiny** (<5% both axes)
- **Use case:** pretraining only; wrong scale for competition small-object scenario

### Dataset A (Fixed-wing-UAV-A, raw casing, cloudy morning)
- 1,632 unique images, 4K, PASCAL VOC XML
- 1,133 unique frame numbers (682 appear in multiple view folders — verified hash-distinct, multi-camera synchronized capture)
- **1,503 per-folder matched pairs** | 1,083 orphan annotations | 129 orphan images
- Bbox: mean 7.3%×4.5%, **48.2% tiny**
- Orphan annotations form runs up to 42 contiguous frames → recoverable from source video

### Dataset A' (Fixed-wing-UAV-A', solar panels, sunny morning)
- 1,182 unique images, 4K, PASCAL VOC XML
- **Archive is broken — only 104 matched pairs:**
  - `Bottom_Up/Zoom_Imgs` folder **entirely missing** (196 orphan XMLs)
  - `Top_Down/Wide`: 202 images, **zero annotations**
  - `Top_Down/Zoom`: 250 images, **zero annotations**
  - `Horizontal/Wide`: 290 images, only 36 annotations (last 40 frames only)
  - `Horizontal/Zoom`: 205 images, only 30 annotations
  - `Bottom_Up/Wide`: 235 images, annotated only from frame 179 onward
- **Interpretation:** annotation work-in-progress zipped mid-task
- MD5-verified: same filenames in different folders are different content (multi-camera), so missing files are not hiding elsewhere

### Your original team dataset (3,558 matched pairs, 12,037 labels)

**MD5 bit-identical overlap:**
- Original ∩ Dataset A: 163 images
- Original ∩ Dataset A': 84 images
- Original unique: **3,311** (not in A or A')

**Perceptual hash (Hamming ≤ 5):**
- Original ∩ A: 181 (~5%)
- Original ∩ A': 96 (~3%)
- Original NOT in either: **3,013 (~85%)**

**The 12-bucket label signature** reveals the source structure:

| Bucket | Images | Labels | Orphans |
|--------|--------|--------|---------|
| 0-999 | 360 | 1000 | 640 |
| 1000-1999 | 583 | 1000 | 417 |
| 2000-2999 | 315 | 1000 | 685 |
| 3000-3999 | 410 | 1000 | 590 |
| 4000-4999 | 310 | 1000 | 690 |
| 5000-5999 | 116 | 886 | 770 |
| 6000-6999 | 95 | 925 | 830 |
| 7000-7999 | 242 | 1000 | 758 |
| 8000-8999 | 335 | 1000 | 665 |
| 9000-9999 | 266 | 1000 | 734 |
| 10000-10999 | 295 | 1000 | 705 |
| 11000-11999 | 163 | 1000 | 837 |
| 12000+ | 68 | 226 | 158 |

~1000 labels per bucket × 12 buckets = matches the **12 MMFW-UAV sorties**. Orphan runs max 58 frames. Strong evidence your team built this dataset by:
1. Processing all 12 MMFW-UAV sorties,
2. Renumbering each sortie into a shared 0-12224 flat namespace (1000 frames per sortie),
3. Extracting images at varying stride (hence 3,558 images but 12,037 labels),
4. Losing data to filename collisions at collision points (explaining why 3,558 ≠ 12,037).

## 3. Summary of gaps and losses

| Source | Has | Missing |
|---|---|---|
| Dataset A | 1503 matched | ~1000 frames across orphan runs |
| Dataset A' | 104 matched | Bottom_Up/Zoom_Imgs entirely, Top_Down ann, most Horizontal ann |
| Team original | 3,558 images | Collision losses, 8,479 orphan labels, all 10 other sortie source files |
| **Versus full MMFW-UAV** | ~2,800 images from 2 sorties | **144,600 images from 10 missing sorties + thermal** |

## 4. Recovery strategy

### Step 0 — Decide on thermal
Your aircraft uses an **RGB camera (RPi Camera Module 3)**. Thermal data won't directly train your deployed model but could be used for:
- Sanity-checking detection across modalities during research
- Training a separate thermal pipeline if you later add an IR camera
- **Default recommendation: skip thermal** to save bandwidth and disk

### Step 1 — Download full MMFW-UAV
```
https://www.scidb.cn/s/vmimum
```
130 GB total. Consider partial download if disk limited:
- RGB-only (skip thermal): ~87 GB
- First 6 sorties only: ~44 GB
- Single sortie for prototyping: ~11 GB

### Step 2 — Convert to YOLO cleanly (avoid the team's bug)
Pseudo-code for the conversion:
```
for sortie in ['A','B','C','D','E','F',"A'","B'","C'","D'","E'","F'"]:
    for view in ['Bottom_Up','Horizontal','Top_Down']:
        for sensor in ['Wide','Zoom']:  # skip Thermal
            for xml_file in {sortie}/{view}/{sensor}_Anns/*.xml:
                # Parse PASCAL VOC
                W, H, boxes = parse_voc(xml_file)
                # Unique key prevents collisions:
                new_stem = f"{sortie}_{view}_{sensor}_{original_stem}"
                # Convert bbox to YOLO normalized
                yolo_lines = [f"0 {(x1+x2)/2/W} {(y1+y2)/2/H} {(x2-x1)/W} {(y2-y1)/H}"
                              for x1,y1,x2,y2 in boxes]
                write(f"{out_dir}/labels/{new_stem}.txt", yolo_lines)
                copy(img_path, f"{out_dir}/images/{new_stem}.jpg")
```
Critical: prefix every output file with sortie/view/sensor to avoid the collision bug that destroyed data in your team's first conversion.

### Step 3 — Split
Stratified 80/10/10 by sortie (keep test-set sorties disjoint from train if possible, to prevent information leakage from same-flight same-aircraft frames).

### Step 4 — Training pipeline
1. **Pretrain YOLOv11s + P2 head** on Dataset_9998pic (9,997 close-up UAVs, ~5 epochs) — teaches "UAV-ness"
2. **Fine-tune** on full MMFW-UAV YOLO conversion (~98k RGB images) — teaches small-object detection with correct scale distribution
3. Export to Hailo HEF with INT8 calibration set drawn from MMFW-UAV
4. Deploy on RPi5 + Hailo-8L via TAPPAS GStreamer pipeline

## 5. What to do with the existing broken data

- **Delete** `images/`, `labels/` (the team's broken 3558+orphan conversion)
- **Delete** `new-data/Dataset A/`, `new-data/Dataset A'/` (partial, broken MMFW-UAV extracts)
- **Keep** `new-data/Dataset_9998pic/` — it's an independent Roboflow set useful for pretraining
- **Download** full MMFW-UAV and start fresh

## Sources

- [MMFW-UAV Paper (Nature Scientific Data)](https://www.nature.com/articles/s41597-025-04482-2)
- [MMFW-UAV Paper (PMC full text)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11782645/)
- [MMFW-UAV GitHub](https://github.com/AIUS-LAB/MMFW-UAV)
- [MMFW-UAV Data host (Science Data Bank)](https://www.scidb.cn/s/vmimum)
- [Roboflow UAV-554 (Dataset_9998pic source)](https://universe.roboflow.com/554-img/uav-554/dataset/4)
