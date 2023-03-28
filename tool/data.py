import pandas as pd
import numpy as np
import os


def split_data(df: pd.DataFrame,
               data_portion: list,
               normalized_feature: list,
               path: str,
               save=True):
    train_length = int(data_portion[0] * len(df))
    test_length = int(data_portion[1] * len(df))
    train_df = df.iloc[:train_length]
    test_df = df.iloc[train_length:train_length + test_length]
    if save:
        train_df.to_feather(os.path.join(path, "train.feather"))
        test_df.reset_index().to_feather(os.path.join(path, "test.feather"))

    mean = np.mean(train_df[normalized_feature])
    std = np.std(train_df[normalized_feature])
    train_df[normalized_feature] = (train_df[normalized_feature] -
                                    mean) / (std + 1e-4)
    test_df[normalized_feature] = (test_df[normalized_feature] - mean) / (std +
                                                                          1e-4)
    if save:
        train_df.to_feather(os.path.join(path, "normalized_train.feather"))
        test_df.reset_index().to_feather(
            os.path.join(path, "normalized_test.feather"))
    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_feather("data/BTCTUSD/2023/all.feather")
    data_portion = [0.6, 0.4]
    normalized_feature = [
        'imblance_volume_oe',
        'sell_spread_oe',
        'buy_spread_oe',
        'kmid2',
        'bid1_size_n',
        'ksft2',
        'ma_10',
        'ksft',
        'kmid',
        'ask1_size_n',
        'trade_diff',
        'qtlu_10',
        'qtld_10',
        'cntd_10',
        'beta_10',
        'roc_10',
        'bid5_size_n',
        'rsv_10',
        'imxd_10',
        'ask5_size_n',
        'ma_30',
        'max_10',
        'qtlu_30',
        'imax_10',
        'imin_10',
        'min_10',
        'qtld_30',
        'cntn_10',
        'rsv_30',
        'cntp_10',
        'ma_60',
        'max_30',
        'qtlu_60',
        'qtld_60',
        'cntd_30',
        'roc_30',
        'beta_30',
        'bid4_size_n',
        'rsv_60',
        'ask4_size_n',
        'imxd_30',
        'min_30',
        'max_60',
        'imax_30',
        'imin_30',
        'cntd_60',
        'roc_60',
        'beta_60',
        'cntn_30',
        'min_60',
        'cntp_30',
        'bid3_size_n',
        'imxd_60',
        'ask3_size_n',
        'sell_volume_oe',
        'imax_60',
        'imin_60',
        'cntn_60',
        'buy_volume_oe',
        'cntp_60',
        'bid2_size_n',
        'kup',
        'bid1_size_normalized',
        'ask1_size_normalized',
        'std_30',
        'ask2_size_n',
    ]
    split_data(df,
               data_portion,
               normalized_feature,
               path="data/BTCTUSD/2023",
               save=True)