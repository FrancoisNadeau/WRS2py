def lloc(x, est=tmean, *args):
    if isinstance(x, pd.DataFrame):
        x = x.values
        x = np.apply_along_axis(pd.to_numeric, 0, x)
    if not isinstance(x, list):
        val = est(x, *args)
    if isinstance(x, list):
        val = [est(item, *args) for item in x]
    if isinstance(x, np.ndarray):
        val = np.apply_along_axis(est, 0, x, *args)
    return val