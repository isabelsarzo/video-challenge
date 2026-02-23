# Automated Detection of Infantile Epileptic Spasms Based on Video-Derived Pose Estimation Landmarks

Submission for the [2026 Video-Based Seizure Detection Challenge](https://computational-neurology.org/#/video_challenge) organized by the Section on Computational Neurology at Charité - Universitätsmedizin Berlin, in partnership with the International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders 2026.

## Authors
Isabel Sarzo Wabi, Daniel Alejandro Galindo Lazo, François Hardy, Ralph Saber, Elie Bou Assi

## Pretrained Models (Docker Images)
The pretrained models can be found here: [Link to Docker Images](https://github.com/orgs/CRCHUM-Epilepsy-Group/packages/container/package/video-challenge).

## Abstract
**Rationale:** Infantile epileptic spasm syndrome (**IESS**) is a severe early form of epilepsy characterized by sudden, repetitive movements of the head and extremities. These events are frequently mistaken for normal infant movements, leading to delayed diagnosis and treatment. Automated detection of epileptic spasms from video could facilitate earlier intervention and improve long-term developmental outcomes. However, video-based detection is challenging due to the sparsity of spasm events and noise introduced by pose estimation in complex real-world scenarios. 

**Algorithm:** We developed an automated spasm detection pipeline that extracts temporal and spectral features from three-dimensional acceleration signals derived from pose estimation landmarks. Two tabular-data classifiers were evaluated within this framework: a machine learning–based model and a deep learning–based model.

**Data Processing:** Raw 5-second video clips recorded at 30 fps were processed using MediaPipe to extract 33 pose landmarks, resulting in 150-frame video segments. Three-dimensional acceleration (**ACC**; _x_, _y_, _z_) was computed for each landmark as the second derivative of position using a Savitzky–Golay filter. From each acceleration axis and the overall ACC magnitude, we extracted both linear and nonlinear features in the temporal and spectral domains.

**Training and Validation:** An extreme gradient boosting (**XGBoost**) classifier and a tabular neural network (**TabNet**) were trained to distinguish _spasm_ from n_on-spasm_ segments. The dataset provided in the challenge was split patient-wise, with 20% of patients reserved as a held-out test set. The remaining data were used for 5-fold cross-validation with patient-level separation between training and validation sets. To reduce feature dimensionality, the most informative features were selected using mutual information (**MI**). The number of selected features and model hyperparameters were optimized during training, using an Optuna-based search to maximize the mean F1-score across validation folds.

**Performance:** On the held-out test set, the XGBoost model achieved 76.5% sensitivity, 37.4% precision, and an F1-score of 50.2%. The TabNet model achieved 90.6% sensitivity, 32.0% precision, and an F1-score of 47.2%.