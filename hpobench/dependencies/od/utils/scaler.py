from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_fitted_scaler(X_train, name=None):
    """
    Instantiates a scaler by a given name and fits the scaler
    with X_train.
    """

    if name == "MinMax":
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    elif name == "Standard":
        scaler = StandardScaler(copy=True)
    elif name is None or name == "None":
        return None
    else:
        raise NotImplementedError()

    scaler.fit(X_train)
    return lambda x: scaler.transform(x)
