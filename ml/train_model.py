import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]
TARGET = 'Outcome'


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # basic cleaning
    df = df.dropna()
    return df


def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f'AUC: {auc:.3f}')
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.path.join('data', 'pima_sample.csv'))
    parser.add_argument('--out', type=str, default=os.path.join('ml', 'model.pkl'))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = load_data(args.data)
    model = train(df)
    joblib.dump(model, args.out)
    print(f'Saved model to {args.out}')


if __name__ == '__main__':
    main()
