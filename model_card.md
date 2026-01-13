# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Model type: Binary classifier predicting whether an individual's income is >50K or <=50K
- Algorithm: Logistic Regression (scikit-learn) with solver="liblinear" and max_iter=2000
- Preprocessing: OneHotEncoder (handle_unknown="ignore", dense output) for categorical features; LabelBinarizer for the target label. See ml/data.py.process_data
- Categorical features used: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Continuous features used: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- Training script: train_model.py
- Inference helper: ml/model.py: inference
- Model artifacts: model/model.pkl (trained LogisticRegression), model/encoder.pkl (trained OneHotEncoder)
- Serving entry point: main.py (FastAPI scaffold)
- Libraries: Python, scikit-learn, pandas, numpy, FastAPI

## Intended Use

- Purpose: Educational ML pipeline for tabular income classification on the US Census dataset; demonstrates preprocessing, model training, evaluation, and basic serving via FastAPI
- Input: A single record with the fields listed above
- Output: Binary prediction string ("<=50K" or ">50K")
- Intended users: Student, and WGU staff
- Out of scope: High-stakes automated decisions without oversight; deployment to populations substantially different from the training distribution without revalidation; use as a proxy for sensitive characteristics

## Training Data

- Source: (Census Income) dataset located at data/census.csv
- Target label: salary ("<=50K" or ">50K")
- Processing: LabelBinarizer on target; OneHotEncoder(handle_unknown="ignore") for categorical features; continuous features concatenated as-is (no scaling)
- Notes: Dataset contains sensitive attributes (e.g., sex, race, marital-status) that may reflect societal biases

## Evaluation Data

- Split: 80/20 train/test split on the raw DataFrame with stratification by salary and random_state=0 (see train_model.py)
- Transformation: Fit encoder/label binarizer on train only; apply to test consistently via process_data
- Slice evaluation: Per-categorical-feature slice metrics written to slice_output.txt

## Metrics

- Reported metrics: Precision, Recall, F1 (beta=1)
- Computation: compute_model_metrics in ml/model.py using predictions from inference(model, X_test)
- Reproduce: Run `python train_model.py`. Overall metrics print to stdout; per-slice metrics append to slice_output.txt
- Notes: zero_division=1 used to avoid undefined metrics on edge slices

## Ethical Considerations

- Potential bias due to sensitive attributes (sex, race, marital-status, age-related proxies). Performance may vary across groups
- Review slice metrics (slice_output.txt) for disparities; consider additional fairness metrics (e.g., demographic parity, equalized odds) before deployment
- Not suitable for consequential decisions without human oversight, legal review, and ongoing monitoring
- Ensure transparent documentation of features, thresholds, and updates; respect privacy when extending data

## Caveats and Recommendations

- Generalization risk to out-of-distribution data; validate before use in new settings
- OneHotEncoder(handle_unknown="ignore") mitigates unseen categories but heavy drift can still harm performance
- Periodically retrain and re-check per-slice metrics; maintain an audit log of changes
- Explore alternatives (e.g., tree-based models or regularization tuning) if linear separability limits performance
