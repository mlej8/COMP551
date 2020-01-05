def min_max_normalization(series):
    """ Function that takes as input a Series and outputs a normalized single column DataFrame """
    return ((series-series.min())/(series.max()-series.min()))

def z_score_normalization(series):
    """ Function that takes as input a Series and outputs a standardized single column DataFrame """
    mean = series.mean()
    std = series.std()
    return series.apply(lambda x: (x-mean)/std)