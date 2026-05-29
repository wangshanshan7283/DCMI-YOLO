DCMI-YOLO: Nighttime Camera-Trap Wildlife Detection
A lightweight YOLOv8-based detector for nighttime / low-light camera-trap wildlife images, designed for endangered-species monitoring (e.g. Amur tiger and Amur leopard). The model integrates four targeted components on top of the Ultralytics YOLOv8 framework:
SCI-Gamma — low-light image enhancement at the input stage
Star-ADown — lightweight downsampling module
MultiSEAM — attention module for occluded / small targets
DF-CIoU — improved bounding-box regression loss
The model detects 17 wildlife categories and is optimized for accuracy–efficiency trade-off (≈2.7 M parameters, ≈7.1 GFLOPs).
> ⚠️ **Data availability**: Due to the conservation sensitivity of the species involved (endangered Amur tiger/leopard), **the dataset is NOT publicly released**. Raw images and location/timestamp information are withheld to avoid any risk of misuse (e.g. poaching). Researchers seeking access for legitimate academic purposes may contact the corresponding author; access may require a data-use / confidentiality agreement.
---
Classes (17)
```
AmurTiger, Badger, BlackBear, Cow, Dog, Hare, Leopard, LeopardCat,
MuskDeer, RaccoonDog, RedFox, RoeDeer, Sable, SikaDeer, Weasel,
WildBoar, Y.T.Marten
```
---
Repository structure
```
DCMI-YOLO/
├── README.md
├── LICENSE                         # AGPL-3.0 (inherited from Ultralytics)
├── requirements.txt
├── .gitignore
├── models/
│   └── StarADown_MUITISEAM_SCIGamma.yaml   # model architecture (nc=17)
├── train.py                        # training entry
├── val.py                          # validation entry
├── eval_metrics.py                 # per-class metrics + params + GFLOPs (CSV export)
├── FPS.py                          # latency / FPS benchmark
├── data/
│   └── A_my_data.yaml.template     # dataset config TEMPLATE (no real paths/data)
└── results/                        # training curves & metrics (NO raw imagery)
    ├── results.csv
    ├── results.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── P_curve.png
    ├── R_curve.png
    ├── F1_curve.png
    └── PR_curve.png
```
> Note: `train_batch*.jpg`, `val_batch*.jpg`, `labels*.jpg`, and TensorBoard event files are intentionally **excluded** because they contain raw camera-trap imagery / dataset statistics.
---
Installation
```bash
# Python 3.8+ recommended
pip install -r requirements.txt
```
`requirements.txt` (minimum):
```
ultralytics>=8.2.0
torch>=2.0.0
numpy
tqdm
```
> The exact `ultralytics` / `torch` versions used in the paper are pinned in `requirements.txt`. If you build on a different YOLO version, results may differ slightly.
---
Dataset format
This project expects the standard Ultralytics detection layout. Prepare your own dataset and a `data.yaml` like:
```yaml
# A_my_data.yaml  (template — fill in YOUR OWN paths)
path: ./datasets/your_dataset      # dataset root (use a neutral path)
train: images/train
val: images/val
test: images/test                  # optional

nc: 17
names: [AmurTiger, Badger, BlackBear, Cow, Dog, Hare, Leopard, LeopardCat,
        MuskDeer, RaccoonDog, RedFox, RoeDeer, Sable, SikaDeer, Weasel,
        WildBoar, Y.T.Marten]
```
---
Usage
1. Train
```bash
python train.py
```
Key settings used in the paper: `imgsz=640`, `epochs=200`, `batch=8`, `optimizer=SGD`, `close_mosaic=0`. Edit the model/data paths inside `train.py` before running.
2. Validate (per-class metrics + Params + GFLOPs)
```bash
python eval_metrics.py
```
Outputs per-class Precision / Recall / mAP@0.5 / mAP@0.5:0.95, overall metrics, model parameters and GFLOPs, and saves a `per_class_metrics.csv`.
> For **reproducing the end-of-training val numbers**, use `rect=True`.
> For **paper-reportable, comparable metrics**, use `rect=False` (square evaluation).
3. Benchmark speed (FPS / latency)
```bash
python FPS.py --weights path/to/best.pt --batch 1 --imgs 640 640 --device 0
```
---
Pretrained weights
The trained weights (`best.pt`, ≈5.6 MB) are provided via the Releases page.
> ⚠️ Note: weights were trained on sensitive endangered-species data. They are released for **research reproducibility** only. Please use responsibly and do not deploy in ways that could facilitate locating or harming wildlife.
---
Results
Final validation performance (see `results/results.csv` and `results/confusion_matrix.png`):
Metric	Value
mAP@0.5	~0.948
mAP@0.5:0.95	~0.801
Parameters	~2.7 M
GFLOPs	~7.1
(Per-class numbers: run `eval_metrics.py`.)
---
License
This project is released under the GNU AGPL-3.0 license, inherited from Ultralytics YOLO, on which it is built. See `LICENSE` for the full text. If you use this code, you must comply with AGPL-3.0 terms (including making source available for networked use).
---
Acknowledgements & Citation
This work is built upon Ultralytics YOLOv8:
> Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
If you use this repository in your research, please also cite our paper:
```bibtex
@article{your_dcmiyolo_2026,
  title   = {DCMI-YOLO: <full paper title>},
  author  = {<authors>},
  journal = {<journal>},
  year    = {2026},
  note    = {Code: https://github.com/<your-username>/DCMI-YOLO}
}
```
> (Fill in the citation block once the paper is accepted.)
