import os
import time
import random
import logging
import pandas as pd
import yfinance as yf
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from ta.volatility import BollingerBands
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
CACHE_DIR = 'data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_hist(ticker, base_sleep=0.1, max_retry=3):
    path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception:
            pass
    for i in range(max_retry):
        try:
            df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
            df.dropna(inplace=True)
            if not df.empty:
                df.to_csv(path)
                return df
        except Exception:
            pass
        time.sleep(base_sleep * (2**i) + random.random() * 0.1)
    return None

def make_features(df):
    X = pd.DataFrame(index=df.index)
    diffs = df['Close'].diff().fillna(0)
    X['obv'] = (diffs.gt(0) * df['Volume'] - diffs.lt(0) * df['Volume']).cumsum()
    X['vol_pct'] = df['Volume'].pct_change().fillna(0)
    X['ma20_diff'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    pb = bb.bollinger_pband().fillna(0)
    arr = pb.values if hasattr(pb, 'values') else pb
    if arr.ndim > 1:
        arr = arr.flatten()
    X['bb_pctb'] = pd.Series(arr, index=df.index)
    X['adx'] = 0
    return X.dropna()

def label_future(df, days=10, target=0.6):
    fut = df['Close'].shift(-days)
    ret = fut / df['Close'] - 1
    return (ret >= target).astype(int).dropna()

if __name__ == '__main__':
    with open("tickers_nasdaq.txt") as f:
        tickers = [t.strip() for t in f if t.strip()]

    all_X, all_y = [], []
    for t in tickers:
        df = load_hist(t)
        if df is None or len(df) < 60:
            continue
        X = make_features(df)
        y = label_future(df)
        idx = X.index.intersection(y.index)
        all_X.append(X.loc[idx])
        all_y.append(y.loc[idx])

    X_full = pd.concat(all_X).sort_index()
    y_full = pd.concat(all_y).sort_index()

    tscv = TimeSeriesSplit(n_splits=5)
    clf = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
        cv=tscv,
        scoring={'precision': 'precision', 'recall': 'recall', 'f1': 'f1'},
        refit='precision',
        n_jobs=-1
    )
    clf.fit(X_full, y_full)

    joblib.dump(clf.best_estimator_, "best_model.pkl")
    shutil.make_archive("trained_model", 'zip', '.', "best_model.pkl")
    logging.info("모델 학습 완료. best_model.pkl 및 trained_model.zip 생성됨")
