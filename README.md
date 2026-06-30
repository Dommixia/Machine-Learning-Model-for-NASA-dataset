# Kepler Object of Interest (KOI) Classification

A machine learning pipeline for classifying Kepler Objects of Interest (KOIs) as confirmed exoplanets or false positives, using physical transit characteristics and stellar parameters derived from the NASA Exoplanet Archive cumulative KOI table.

## Overview

The Kepler space telescope identified over 9,000 candidate planetary transit signals (KOIs) by monitoring the brightness of more than 150,000 stars for periodic dimming events. Each candidate requires vetting to distinguish a genuine planetary transit from astrophysical false positives such as eclipsing binaries, background blends, and instrumental artifacts. This project trains an ensemble classifier to perform that vetting task using only physically meaningful features of the transit signal and host star, replicating the reasoning an astronomer applies during manual disposition review rather than relying on pipeline-derived vetting statistics.

The model treats this as a binary classification problem:

- **CONFIRMED**: the KOI has been independently validated as a genuine exoplanet.
- **FALSE POSITIVE**: the KOI has been determined not to be a planet (e.g., eclipsing binary, background contamination, stellar variability, or instrumental noise).

Candidates with a `CANDIDATE` disposition (unresolved) are excluded from training, since their ground truth label does not yet exist.

## Methodology

### Target Variable

The target is the `koi_disposition` column, restricted to `CONFIRMED` and `FALSE POSITIVE` rows and label-encoded to a binary indicator y ∈ {0, 1}.

### Feature Selection and Leakage Prevention

A central design decision in this project was distinguishing between two categories of columns in the cumulative KOI table:

1. **Physical features**: quantities derived from the transit light curve geometry and stellar characterization (period, depth, duration, radius, temperature, signal strength). These represent the same evidence an astronomer uses to judge whether a signal is consistent with a real planet.
2. **Vetting diagnostic features**: columns that are themselves outputs of the Kepler disposition pipeline's automated false-positive tests, such as centroid offset statistics (`koi_dikco_msky`, `koi_dicco_msky`) and flux-weighted centroid motion significance (`koi_fwm_stat_sig`). These are designed explicitly to detect background eclipsing binaries and similar contaminants, and are computed using procedures correlated with how the original disposition label was assigned.

Early iterations of this model included the second category and achieved an artificially inflated AUC near 0.99, with `koi_dikco_msky` and `koi_fwm_stat_sig` alone accounting for over 35 percent of model weight. Including these features means the model is not learning to distinguish planets from false positives on physical grounds; it is partially decoding the answer key. This was confirmed by removing the diagnostic columns one group at a time and observing a feature-importance redistribution toward physically meaningful predictors with no equivalent collapse in test AUC, indicating the remaining physical signal was substantial and genuine.

The columns excluded prior to training fall into three groups:

```
Direct disposition leakage:
    koi_disposition, koi_pdisposition, koi_score,
    koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec

Identifiers (non-predictive):
    rowid, kepid, kepler_name, koi_datalink_dvr, koi_fittype

Vetting pipeline diagnostics (excluded via whitelist, see below):
    koi_dikco_msky, koi_dicco_msky, koi_fwm_stat_sig,
    koi_fwm_sdec_err, koi_fwm_srao_err, koi_max_mult_ev,
    koi_count, koi_num_transits, koi_tce_plnt_num
```

Rather than maintaining an exhaustive blacklist, the final model trains on an explicit whitelist of nineteen physically interpretable features (`clean_features`), described below. This is a stronger guarantee against leakage than blacklisting, since any new or unanticipated diagnostic column is excluded by default rather than requiring active removal.

### Final Feature Set

| Feature | Description | Units |
|---|---|---|
| `koi_period` | Orbital period of the candidate | days |
| `koi_duration` | Transit duration | hours |
| `koi_depth` | Fractional decrease in stellar flux during transit | parts per million |
| `koi_ror` | Planet-to-star radius ratio, R_p / R_star | dimensionless |
| `koi_impact` | Impact parameter, sky-projected distance between planet and star centers at conjunction, normalized by stellar radius | dimensionless |
| `koi_model_snr` | Transit signal-to-noise ratio from the detection pipeline's best-fit model | dimensionless |
| `koi_prad` | Planet radius | Earth radii |
| `koi_teq` | Equilibrium temperature, assuming zero albedo and full heat redistribution | Kelvin |
| `koi_insol` | Insolation flux received by the planet | Earth flux units |
| `koi_dor` | Distance over stellar radius, a / R_star (scaled semi-major axis) | dimensionless |
| `koi_sma` | Semi-major axis of the orbit | astronomical units |
| `koi_steff` | Stellar effective temperature | Kelvin |
| `koi_slogg` | Stellar surface gravity, log10(g) | cgs (log10 cm/s^2) |
| `koi_srad` | Stellar radius | Solar radii |
| `koi_smet` | Stellar metallicity, [Fe/H] | dex |
| `koi_smass` | Stellar mass | Solar masses |
| `koi_kepmag` | Kepler-band apparent magnitude | magnitude |
| `duty_cycle` | Engineered feature, see below | dimensionless |
| `depth_per_snr` | Engineered feature, see below | parts per million |

### Engineered Features

Two additional features were derived from the base measurements to encode transit-geometry consistency directly, rather than leaving the ensemble to infer these relationships implicitly across separate columns.

**Duty cycle**

The duty cycle is the fraction of the orbital period during which the planet is in transit:

```
duty_cycle = koi_duration / (koi_period * 24)
```

`koi_duration` is recorded in hours and `koi_period` in days, hence the multiplication by 24 to express the period in consistent units before dividing. Physically, duty cycle is bounded by orbital geometry: for a circular orbit and a star of radius R_star, the maximum transit duration is governed by

```
duty_cycle_max ≈ R_star / (pi * a)
```

where 'a' is the orbital semi-major axis. Transit signals with a duty cycle inconsistent with their orbital separation given stellar size are geometrically implausible as genuine transits and are more likely artifacts, instrumental systematics, or misfit periods. This feature dominates the trained ensemble's importance ranking (15.2 percent of total weight), confirming that transit-shape consistency is one of the strongest discriminators between confirmed planets and false positives.

**Depth per SNR**

```
depth_per_snr = koi_depth / (koi_model_snr + 1e-5)
```

This ratio normalizes the observed transit depth by the statistical confidence of the detection. A small epsilon term (1e-5) is added to the denominator to prevent division by zero in edge cases where SNR is reported as zero. The intuition is that two signals with identical measured depth can carry very different physical credibility depending on how strongly that depth is statistically supported; a deep transit measured with low SNR is more consistent with noise or a poorly constrained fit than a shallower transit measured with high SNR. This feature provides the ensemble with a single combined signal-quality indicator rather than requiring it to learn the relationship between depth and SNR independently across trees.

### Data Preprocessing

1. **Type filtering**: only numeric columns are retained (`select_dtypes(include='number')`), removing string and categorical metadata fields not suitable for direct use by the tree-based models.
2. **All-null column removal**: columns with no observed values for any row are dropped (`dropna(axis=1, how='all')`).
3. **Median imputation**: missing values are filled using the median of each column. Critically, this is computed and applied separately to prevent test-set information from leaking into training:
   - An initial median fill is applied to the full feature matrix prior to the train/test split for columns excluded from later modeling.
   - After splitting, `train_medians` is computed exclusively from `X_train` and used to fill missing values in both `X_train` and `X_test`. This ensures the imputation values are derived only from data the model is permitted to learn from, preserving the validity of the held-out test set as an unbiased performance estimate.
4. **Stratified train/test split**: an 80/20 split (`test_size=0.2`) is performed with `stratify=y` to preserve the class balance of CONFIRMED versus FALSE POSITIVE in both partitions, which is important given the moderate class imbalance in the dataset (approximately 968 confirmed-class versus 550 false-positive-class examples in the test partition).
5. **Feature engineering**: `duty_cycle` and `depth_per_snr` are computed after imputation and after the split, applied independently to `X_train` and `X_test` using each partition's own underlying values, to avoid any cross-contamination between train and test data.
6. **Whitelist filtering**: the final feature matrices `X_train_final` and `X_test_final` are restricted to the nineteen-column `clean_features` list described above.

## Model Architecture

The classifier is a soft-voting ensemble of three independently trained tree-based models, combining their class-probability outputs:

```
P_ensemble(y=1 | x) = (1/3) * [P_rf(y=1 | x) + P_xgb(y=1 | x) + P_gb(y=1 | x)]
```

| Model | Configuration |
|---|---|
| Random Forest | 100 estimators, random_state=42 |
| XGBoost | 100 estimators, learning_rate=0.05, eval_metric='logloss', random_state=42 |
| Gradient Boosting | 100 estimators, random_state=42 |

Soft voting was selected over hard voting because it incorporates each base learner's confidence rather than only its discrete class prediction, generally producing better-calibrated probability estimates and a smoother decision boundary when combining heterogeneous tree ensembles.

Feature importance is reported as the unweighted average of each base model's native importance scores:

```
importance_ensemble = (importance_rf + importance_xgb + importance_gb) / 3
```

normalized to a percentage of total importance across all retained features.

## Results

| Metric | Value |
|---|---|
| Training AUC | 0.9977 |
| Testing AUC | 0.9768 |
| Test accuracy | 0.93 |
| Precision (FALSE POSITIVE) | 0.90 |
| Recall (FALSE POSITIVE) | 0.90 |
| Precision (CONFIRMED) | 0.95 |
| Recall (CONFIRMED) | 0.94 |

The gap of approximately 0.02 between training and testing AUC indicates mild overfitting consistent with tree ensemble behavior at this depth and estimator count, rather than a data leakage artifact; it could be further reduced through cross-validated hyperparameter tuning or explicit depth regularization, which is noted as a direction for future refinement rather than a correction required for the present result.

### Feature Importance

The five highest-weighted features in the final model are:

1. `koi_prad` (planet radius) — 22.4 percent
2. `duty_cycle` (engineered) — 15.2 percent
3. `koi_model_snr` (signal-to-noise ratio) — 13.4 percent
4. `koi_dor` (scaled semi-major axis) — 7.3 percent
5. `koi_smet` (stellar metallicity) — 6.8 percent

This ranking is consistent with established exoplanet vetting principles: planet radius is the primary discriminant because false positives frequently exhibit stellar-scale radii inconsistent with a planetary interpretation; duty cycle and SNR capture transit-shape plausibility and detection confidence; and stellar metallicity reflects the empirically observed correlation between host star metal content and planet occurrence rate. No vetting pipeline diagnostic feature appears in the top fifteen, supporting the conclusion that the model's predictive power derives from genuine transit physics rather than disposition-pipeline shortcuts.

## Visualizations

### Feature Importance

![Feature Importance](Graphs/feature_importance.png)

This chart reports the ensemble-averaged importance of each retained feature, normalized to a percentage of total model weight. `koi_prad` (planet radius) is the single strongest predictor at 22.4 percent, followed by the engineered `duty_cycle` feature (15.2 percent) and `koi_model_snr` (13.4 percent). Together these three features account for over half of total model weight, indicating the classifier relies primarily on whether a candidate's size is physically plausible for a planet and whether its transit shape and detection strength are statistically convincing, rather than on any single stellar parameter. No vetting-pipeline diagnostic feature appears in this ranking, supporting the conclusion that predictive power is derived from genuine transit physics.

### ROC Curve

![ROC Curve](Graphs/roc_curve.png)

The ROC curve plots true positive rate against false positive rate across all classification thresholds, with an area under the curve (AUC) of 0.9768. The curve rises steeply near the origin, reaching a true positive rate above 0.85 while the false positive rate remains below 0.05, indicating the model achieves strong separation between classes at conservative decision thresholds. This is a substantially different curve shape from the near-perfect, almost rectangular ROC curve produced by the earlier leakage-affected model (AUC near 0.99 with `koi_dikco_msky`/`koi_fwm_stat_sig` included); the comparatively more gradual upper-left bend here reflects a classifier learning from genuine but imperfect physical evidence rather than near-deterministic vetting outputs.

### SHAP Summary Plot

![SHAP Summary](Graphs/shap_summary.png)

The SHAP (Shapley Additive exPlanations) plot decomposes individual predictions from the XGBoost component of the ensemble, showing both the magnitude and direction of each feature's contribution across the test set. Each point is one test sample; horizontal position indicates the feature's impact on the model's output toward CONFIRMED (positive) or FALSE POSITIVE (negative), and color indicates whether the underlying feature value was high (red/pink) or low (blue) for that sample.

The pattern for `koi_prad` is the clearest in the plot: low planet-radius values (blue) cluster tightly around small negative SHAP contributions, while high radius values (red) spread across a wide positive range extending past +4, indicating that as candidate radius increases into ranges consistent with realistic planet sizes, the model is pushed more strongly toward a CONFIRMED classification. `koi_smet` (stellar metallicity) shows a distinctive separated pattern, with both very high and very low metallicity values producing negative SHAP contributions while mid-range values cluster near zero, consistent with the astronomical understanding that planet occurrence correlates with metallicity in a non-linear way rather than monotonically. `koi_model_snr` shows that most low-to-moderate SNR values contribute near zero or slightly negative, while a subset of very high SNR values produce strong positive contributions, reflecting that high-confidence detections meaningfully increase the likelihood of a CONFIRMED prediction. This sample-level view complements the aggregate feature importance chart by showing not just which features matter, but the direction and consistency of their effect on individual predictions.

### Confusion Matrix

![Confusion Matrix](Graphs/confusion_matrix.png)

| | Predicted CONFIRMED | Predicted FALSE POSITIVE |
|---|---|---|
| **True CONFIRMED** | 497 | 53 |
| **True FALSE POSITIVE** | 56 | 912 |

Out of 1,518 test samples, the model correctly classifies 1,409 (92.8 percent), consistent with the reported 0.93 accuracy. The two error types are nearly balanced: 53 genuine planets are misclassified as false positives (a missed detection, reducing recall on the CONFIRMED class to approximately 0.90), and 56 false positives are misclassified as confirmed planets (a false alarm, reducing precision on the CONFIRMED class to approximately 0.90). Given the larger sample size of the FALSE POSITIVE class in this test split (968 versus 550), the model maintains comparable error rates across both classes rather than skewing predictions toward the majority class, indicating the stratified split and class-balanced feature signal are working as intended rather than producing a model biased purely by class frequency.

### Cross-Visualization Comparison

Taken together, these four visualizations corroborate a single, consistent conclusion. The feature importance and SHAP plots agree on which features matter most (`koi_prad`, `duty_cycle`, `koi_model_snr`), and the SHAP plot additionally confirms that their directional effect on individual predictions aligns with established astrophysical reasoning rather than an arbitrary correlation. The ROC curve's AUC of 0.9768, while still high, reflects a curve shape consistent with a model relying on continuous physical measurements with natural overlap between classes, rather than the artificially sharp separation produced when vetting-pipeline diagnostics were included. The confusion matrix's balanced error distribution across both classes is the practical expression of that same finding: the model is not exploiting a shortcut that perfectly separates classes, but making genuine probabilistic judgments based on physical evidence, with a residual error rate that is realistic for the difficulty of the underlying astrophysical classification task.

## Repository Structure

```
KOI_NASA_ML/
├── main.py                       # Full training and evaluation pipeline
├── KOI_Cumulative_clean.csv      # Source dataset (NASA Exoplanet Archive, cumulative KOI table)
└── Graphs/
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── shap_summary.png
```

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

## Limitations and Future Work

- The model is restricted to a static cumulative table and does not incorporate light-curve time-series data directly; a convolutional or sequence-based model operating on raw or folded light curves could capture additional signal not summarized in the tabulated parameters.
- Hyperparameters were not tuned via cross-validation; a grid or Bayesian search over estimator depth, learning rate, and regularization terms may narrow the train/test AUC gap.
- The current feature whitelist was constructed through iterative removal of suspected leakage sources rather than a formal causal feature-selection procedure; a more rigorous approach would involve consulting the official Kepler vetting documentation to enumerate all pipeline-derived diagnostic columns prior to model development.