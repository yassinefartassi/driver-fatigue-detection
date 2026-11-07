# Data and models

**No datasets, recordings or model weights are stored in this repository.**

## Datasets

None are needed. The system uses no trained classifier of its own. It
computes the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) from facial
landmarks, adds heart rate from the HBPM sensor, and applies thresholds set in
`config.py`. There is nothing to train, so no training data is required.

At runtime, the optional data logger writes measurements to `data/`
(override with `FATIGUE_DATA_DIR`), and logs go to `logs/` (override with
`FATIGUE_LOG_FILE`). Both folders are git-ignored because they hold
measurements of whoever was driving.

## Model: dlib 68-point facial landmark predictor

| | |
|---|---|
| File | `models/shape_predictor_68_face_landmarks.dat` (≈100 MB, ≈64 MB compressed) |
| Download | `python main.py --download-model` or `scripts/download_data.sh`, from <https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2> |
| Override location | `FATIGUE_MODELS_DIR` |
| Provider | [dlib](http://dlib.net/) (Davis E. King) |
| Licence note | dlib states that this model was trained on the iBUG 300-W dataset, whose licence **excludes commercial use**. Check the terms at <https://github.com/davisking/dlib-models> before any commercial use. |

### Citations

- King, D. E. (2009). Dlib-ml: A Machine Learning Toolkit. *Journal of Machine Learning Research*, 10, 1755–1758.
- Kazemi, V., & Sullivan, J. (2014). One millisecond face alignment with an ensemble of regression trees. *CVPR 2014*.
- Sagonas, C., Antonakos, E., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2016). 300 Faces In-The-Wild Challenge: database and results. *Image and Vision Computing*, 47, 3–18.
- Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. *21st Computer Vision Winter Workshop* (the EAR metric).
