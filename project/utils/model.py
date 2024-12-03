import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Tuple

def train_mmm_model(data_mmm: pd.DataFrame, shop_name: str = 'beauty_shop_1') -> Tuple[LinearRegression, pd.DataFrame, pd.Series]:
    data_test = data_mmm[data_mmm['provider_account'] == shop_name] \
        .set_index('event_date') \
        .drop(columns=['provider_account', 'total_price'])
    X = data_test.drop(columns=['total_price_usd'])
    y = data_test['total_price_usd']
    lr = LinearRegression()
    lr.fit(X, y)

    return lr, X, y




