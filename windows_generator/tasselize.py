
import pandas as pd

from .decorators import multiprocess


@multiprocess
def tasselize_window(chrom: str, chromStart: int, chromEnd: int, window_size: int):
    return pd.DataFrame([
        {
            "chrom": chrom,
            "chromStart": chromStart + window_size*i,
            "chromEnd": chromStart + window_size*(i+1),
        }
        for i in range((chromEnd - chromStart)//window_size)
    ])
