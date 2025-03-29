import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from concave_hull import concave_hull_indexes
from typing import List, Optional
import numpy as np
import seaborn as sns
import pandas as pd

def generate_figure(
        coord_dataframe: pd.DataFrame,
        labels: Optional[List[int]] = None,
) -> None:
    """
    Generates a matplotlib scatterplot with optional bounded clusters.

    The plot is drawn from a dataframe of xy values, each point labelled by index `participant_id`.
    When a list of labels are supplied (corresponding to each row), concave hulls are drawn around them.

    Args:
        coord_dataframe (pd.DataFrame): A dataframe of coordinates with columns named `x` and `y`, indexed by `participant_id`.
        labels (List[int]): A list of labels, one for each row in `coord_dataframe`.

    Returns:
        None.
    """
    plt.figure(figsize=(7, 5), dpi=80)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.gca().invert_xaxis()

    # Label points with participant_id if no labels set.
    for participant_id, row in coord_dataframe.iterrows():
        plt.annotate(str(participant_id),
            (float(row["x"]), float(row["y"])),
            xytext=(2, 2),
            color="gray",
            textcoords='offset points')

    scatter_kwargs = defaultdict()
    scatter_kwargs["x"] = coord_dataframe.loc[:,"x"]
    scatter_kwargs["y"] = coord_dataframe.loc[:,"y"]
    scatter_kwargs["s"] = 10       # point size
    scatter_kwargs["alpha"] = 0.8  # point transparency
    if labels is not None:
        # Ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
        scatter_kwargs["cmap"] = "Set1"    # color map
        scatter_kwargs["c"] = labels        # color indexes

        print("Calculating convex hulls around clusters...")
        unique_labels = set(labels)
        for label in unique_labels:
            points_df = coord_dataframe[labels == label]
            print(f"Hull {str(label)}, bounding {len(points_df)} points")
            if len(points_df) < 3:
                # TODO: Accomodate 2 points like Polis platform does.
                print("Cannot create concave hull for less than 3 points. Skipping...")
                continue
            vertex_indices = concave_hull_indexes(np.asarray(points_df.loc[:, ["x", "y"]]), concavity=4.0)
            hull_points = points_df.iloc[vertex_indices, :]
            hull_points = hull_points.loc[:, ["x", "y"]]
            polygon = patches.Polygon(
                hull_points,
                fill=True,
                color="gray",
                alpha=0.3,
                edgecolor=None,
            )
            plt.gca().add_patch(polygon)
    scatter = plt.scatter(**scatter_kwargs)

    # Add a legend if labels are provided
    if labels is not None:
        plt.colorbar(scatter, label="Cluster", ticks=labels)

    plt.show()

    return None

class DataPresenter():
    def __init__(self, client):
        self.client = client

    def render_optimal_cluster_figure(self):
        print(f"Optimal clusters for K={self.client.optimal_k}")
        print("Plotting PCA embeddings with K-means, K="+str(np.max(self.client.optimal_cluster_labels)+1))
        self.generate_figure(coord_dataframe=self.client.projected_data, labels=self.client.optimal_cluster_labels)

    def generate_figure(self, coord_dataframe, labels=None):
        generate_figure(coord_dataframe=coord_dataframe, labels=labels)

    def generate_vote_heatmap(self, vote_df):
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_theme(font_scale=.7)
        sns.set_color_codes()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(vote_df, center=0, cmap="RdYlBu", ax=ax)
        plt.show()
