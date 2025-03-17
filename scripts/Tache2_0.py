from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import pandas as pd


def scale_data(df, method="standard"):
    """
    Applique une technique de feature scaling sur un DataFrame pandas.

    :param df: DataFrame pandas contenant les données numériques à scaler
    :param method: Méthode de scaling à utiliser ("standard", "minmax", "robust", "maxabs")
    :return: DataFrame transformé
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler()
    }

    if method not in scalers:
        raise ValueError("Méthode de scaling non reconnue. Utilisez 'standard', 'minmax', 'robust' ou 'maxabs'.")

    scaler = scalers[method]
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    return df_scaled