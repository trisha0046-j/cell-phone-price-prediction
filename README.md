# Cell Phone Price Prediction

A multi-class classification project that predicts the **price range** of a mobile phone (Low, Medium, High, Very High) from its specifications.

## Problem Statement

A new mobile phone company wants to price its phones competitively but does not know how specifications relate to price. Using historical phone data, this project builds a model that classifies a phone into one of four price ranges based on features such as RAM, battery power, pixel resolution, memory, cameras, screen size and connectivity.

## Dataset

- File: `datasets_11167_15520_train.csv` (the public *Mobile Price Classification* dataset)
- 2,000 phones · 20 features · 1 target (`price_range`: 0 = Low, 1 = Medium, 2 = High, 3 = Very High)
- Perfectly balanced classes (500 phones per range), no missing values, no duplicates
- Features include `battery_power`, `ram`, `int_memory`, `px_height`, `px_width`, `fc`, `pc`, `sc_h`, `sc_w`, `talk_time`, `n_cores`, `clock_speed`, and binary flags (`blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi`)
- The dataset is **not included** in this repository. Place the CSV in the project folder before running the notebook.

## Workflow

1. **Data understanding** — shape, types, missing values, duplicates, unique values
2. **EDA** — target distribution, feature distributions, binary features, outliers, skewness, correlation heatmap, features vs price range
3. **Preprocessing** — stratified 80/20 split (1,600 train / 400 test), `StandardScaler` fitted on the training set only
4. **Models** — Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting (scaled data for Logistic Regression, KNN and SVM; raw data for tree models)
5. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, classification reports
6. **Cross-validation** — 5-fold on the training set
7. **Hyperparameter tuning** — `RandomizedSearchCV` for Logistic Regression, Random Forest, SVM and Gradient Boosting
8. **Feature importance** and **business analysis**

## Results

| Model | Test Accuracy | 5-fold CV Accuracy |
|---|---:|---:|
| **Logistic Regression** | **96.5%** | **94.6%** |
| Gradient Boosting | 91.3% | 88.7% |
| SVM (RBF) | 89.5% | 85.9% |
| Random Forest | 88.0% | 86.9% |
| Decision Tree | 83.0% | 82.9% |
| KNN (k=5) | 50.0% | 49.8% |

After tuning, Logistic Regression (`C=10`) was selected as the final model and reached **97.5% accuracy** on the held-out test set, with precision and recall around 0.95–0.99 for every price range.

**Key takeaways**

- **RAM is by far the most important feature** (about 48% of Random Forest importance), followed by battery power, pixel width and pixel height.
- The simple Logistic Regression beat the more complex models, which suggests the price classes are largely separable by roughly linear relationships in the specifications.
- Tuning gave Logistic Regression a small gain; other models changed little.
- KNN performed poorly (about 50%).
- Binary features such as Bluetooth, Wi-Fi, 4G and dual SIM contribute very little.

## Business Use

The model can help the company estimate which price segment a new phone belongs to and decide which specifications (especially RAM, battery and display resolution) to prioritise when designing phones for different segments.

## Limitations & Future Improvements

- Only 2,000 phones and a single train-test split; each of the 400 test phones is worth 0.25 percentage points of accuracy
- Tune SVM and KNN more thoroughly (wider search, feature selection for KNN) before concluding they are weaker
- Check feature importance for the final Logistic Regression model (coefficients) rather than only Random Forest
- Deploy as a small web app where a user enters specifications and gets a price range (Streamlit / Flask)

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter

## How to Run

```bash
git clone https://github.com/<your-username>/cell-phone-price-prediction.git
cd cell-phone-price-prediction

python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

# place datasets_11167_15520_train.csv in this folder, then:
jupyter notebook Cell_Phone_Price_Prediction.ipynb
```

## Project Structure

```
cell-phone-price-prediction/
├── Cell_Phone_Price_Prediction.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

**Trisha** — AI Engineer Trainee, DataMites
B.Tech CSE · Python · Machine Learning · Data Visualization

[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>)
