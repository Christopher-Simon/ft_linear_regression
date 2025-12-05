"""
This module provides functions to visualize the \
    scaling of data using MinMax and Z-Score normalization.
"""

from pandas.core.frame import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from normalizers.minmax_normalizer import MinMaxNormalizer
from normalizers.z_score import ZScoreNormalizer
from visualisation.data_graph import plot_file_data


def scaling_viz() -> None:
    """
    Visualize the scaling of data using MinMax and Z-Score normalization.
    """
    data: DataFrame = pd.read_csv("data/data.csv")
    plot_file_data(data)
    minmax_normalizer = MinMaxNormalizer(data)
    plot_file_data(minmax_normalizer.transform(), title="MinMax Normalization")

    z_score_normalizer = ZScoreNormalizer(data)
    plot_file_data(z_score_normalizer.transform(), title="Z-Score Normalization")

    plt.show()
