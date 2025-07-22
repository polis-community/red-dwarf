from collections import defaultdict
from typing import List, Optional
from reddwarf.types.polis import PolisRepness
import numpy as np

from reddwarf.implementations.base import PolisClusteringResult
from reddwarf.utils.consensus import ConsensusResult

from reddwarf.exceptions import try_import

# Support optional extras groups by throwing a warning more helpful than default error.
matplotlib = try_import("matplotlib", extra="plots")
sns = try_import("seaborn", extra="plots")
concave_hull = try_import("concave_hull", extra="plots")

plt = matplotlib.pyplot
patches = matplotlib.patches

GROUP_LABEL_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def generate_figure_polis(
    result: PolisClusteringResult,
    show_guesses=False,
    flip_x=True,
    flip_y=False,
    show_pids=True,
):
    """
    Generate a Polis-style visualization from clustering results.

    Args:
        result (PolisClusteringResult): The result object from run_pipeline
        show_guesses (bool): Show the initial cluster center guesses on the plot
        flip_x (bool): Flip the X-axis (default True to match Polis interface)
        flip_y (bool): Flip the Y-axis (default False)
        show_pids (bool): Show the participant IDs on the plot
    """
    participants_clustered_df = result.participants_df[
        result.participants_df["cluster_id"].notnull()
    ]
    cluster_labels = participants_clustered_df["cluster_id"].values

    coord_data = participants_clustered_df.loc[:, ["x", "y"]].values
    coord_labels = None
    # Add the init center guesses to the bottom of the coord stack. Internally, they
    # will be give a fake "-1" colored label that won't be used to draw clusters.
    # This is for illustration purpose to see the centroid guesses.
    if show_guesses:
        coord_data = np.vstack(
            [
                coord_data,
                np.asarray(
                    result.clusterer.init_centers_used_ if result.clusterer else []
                ),
            ]
        )

    if show_pids:
        coord_labels = [f"p{pid}" for pid in participants_clustered_df.index]

    generate_figure(
        coord_data=coord_data,
        coord_labels=coord_labels,
        cluster_labels=cluster_labels,
        # Always needs flipping to look like Polis interface.
        flip_x=flip_x,
        # Sometimes needs flipping to look like Polis interface.
        flip_y=flip_y,
    )


def generate_figure(
    coord_data,
    coord_labels=None,
    cluster_labels: Optional[List[int]] = None,
    flip_x: bool = False,
    flip_y: bool = False,
) -> None:
    """
    Generates a matplotlib scatterplot with optional bounded clusters.

    The plot is drawn from a dataframe of xy values, each point labelled by index `participant_id`.
    When a list of cluster labels are supplied (corresponding to each row), concave hulls are drawn around them.

    Signs of PCA projection coordinates are arbitrary, and can flip without
    meaning. Inverting axes can help compare results with Polis platform
    visualizations.

    Args:
        coord_data (pd.DataFrame): A dataframe of coordinates with columns named `x` and `y`, indexed by `participant_id`.
        cluster_labels (List[int]): A list of group labels, one for each row in `coord_dataframe`.
        flip_x (bool): Flip the presentation of the X-axis so it descends left-to-right
        flip_y (bool): Flip the presentation of the Y-axis so it descends top-to-bottom

    Returns:
        None.
    """
    plt.figure(figsize=(7, 5), dpi=80)
    plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    plt.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    if flip_x:
        plt.gca().invert_xaxis()
    if flip_y:
        plt.gca().invert_yaxis()

    # Label points when coordinate labels are provided.
    if coord_labels is not None:
        for label, xy in zip(coord_labels, coord_data):
            plt.annotate(
                str(label),
                (float(xy[0]), float(xy[1])),
                xytext=(2, 2),
                color="gray",
                textcoords="offset points",
            )

    scatter_kwargs = defaultdict()
    scatter_kwargs["x"] = coord_data[:, 0]
    scatter_kwargs["y"] = coord_data[:, 1]
    scatter_kwargs["s"] = 10  # point size
    scatter_kwargs["alpha"] = 0.8  # point transparency

    # Wrap clusters in hulls when cluster labels are provided.
    if cluster_labels is not None:
        # Ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
        scatter_kwargs["cmap"] = "Set1"  # color map

        # Pad cluster_labels to match the number of points
        CLUSTER_CENTER_LABEL = -2
        if len(cluster_labels) < len(coord_data):
            pad_length = len(coord_data) - len(cluster_labels)
            cluster_labels = np.concatenate(
                [cluster_labels, [CLUSTER_CENTER_LABEL] * pad_length]
            )

        scatter_kwargs["c"] = cluster_labels  # color indexes

        print("Calculating convex hulls around clusters...")
        # Subset to allow unlabelled points to just be plotted
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            if label in (-1, -2):
                continue  # skip hulls when special-case labels used

            label_mask = cluster_labels == label
            cluster_points = coord_data[label_mask]

            print(f"Hull {label}, bounding {len(cluster_points)} points")

            if len(cluster_points) < 3:
                # TODO: Accomodate 2 points like Polis platform does.
                print("Cannot create concave hull for less than 3 points. Skipping...")
                continue

            hull_point_indices = concave_hull.concave_hull_indexes(cluster_points, concavity=4.0)
            hull_points = cluster_points[hull_point_indices]

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
    if cluster_labels is not None:
        unique_labels = np.unique(cluster_labels)
        cbar = plt.colorbar(scatter, label="Cluster", ticks=unique_labels)

        tick_labels = []
        for lbl in unique_labels:
            if lbl == -1:
                tick_labels.append("[Unclustered]")
            elif lbl == -2:
                tick_labels.append("[Center Guess]")
            else:
                tick_labels.append(GROUP_LABEL_NAMES[lbl])
        cbar.ax.set_yticklabels(tick_labels)

    plt.show()

    return None


def print_selected_statements(
    result: PolisClusteringResult, statements_data: list[dict]
) -> None:
    """
    Print formatted output of consensus and group-representative statements from Polis algorithm.

    Arguments:
        result (PolisClusteringResult): The result object of the red-dwarf polis implementation.

    Returns:
        None
    """
    print("# CONSENSUS STATEMENTS")
    print()
    print_consensus(consensus=result.consensus, statements_data=statements_data)

    print()

    print("# GROUP-REPRESENTATIVE STATEMENTS")
    print()
    print_repness(repness=result.repness, statements_data=statements_data)


def print_consensus(consensus: ConsensusResult, statements_data: list[dict]) -> None:
    """
    Helper function to format printed output of Polis repness object.

    Arguments:
        consensus (ConsensusResult): Data that matches structure from polismath API
        statements_data (list[dict]): Statement data with keys `statement_id` and `txt`

    Returns:
        None
    """
    for direction, statements in consensus.items():
        print(f"## FOR {direction.upper()}MENT")
        print()
        if not statements:
            print("None.")
            print()
        else:
            for cons_st in list(statements):
                st_data = [
                    st for st in statements_data if st["statement_id"] == cons_st["tid"]
                ][0]
                tmpl = "\n".join(
                    [
                        "* {txt}\n"
                        "    {percent}% of everyone who voted on statement {statement_id} {direction}d.\n"
                    ]
                )
                print(
                    tmpl.format(
                        txt=st_data["txt"],
                        percent=int((cons_st["n-success"] / cons_st["n-trials"]) * 100),
                        statement_id=cons_st["tid"],
                        direction=direction,
                    )
                )


def print_repness(
    repness: PolisRepness,
    statements_data: list[dict],
) -> None:
    """
    Helper function to format printed output of Polis repness object.

    Arguments:
        repness (PolisRepness): Data that matches structure from polismath API
        statements_data (list[dict]): Statement data with keys `statement_id` and `txt`

    Returns:
        None
    """
    for gid, repful_statements in repness.items():
        gid = int(gid)
        group_label = GROUP_LABEL_NAMES[gid]
        print("## GROUP {group_label}".format(group_label=group_label))
        print()

        for rep_st in repful_statements:
            st_data = [
                st for st in statements_data if st["statement_id"] == rep_st["tid"]
            ][0]
            print(f"* {st_data['txt']}")
            if rep_st["repful-for"] == "agree":
                tmpl = "   {percent}% of those in group {group_label} who voted on statement {statement_id} agreed."
                print(
                    tmpl.format(
                        group_label=group_label,
                        statement_id=rep_st["tid"],
                        percent=int((rep_st["n-success"] / rep_st["n-trials"]) * 100),
                    )
                )
            else:
                tmpl = "   {percent}% of those in group {group_label} who voted on statement {statement_id} disagreed."
                print(
                    tmpl.format(
                        percent=int((rep_st["n-success"] / rep_st["n-trials"]) * 100),
                        group_label=group_label,
                        statement_id=rep_st["tid"],
                    )
                )
            print("")

        print("")


def generate_vote_heatmap(vote_df):
    sns.set_context("poster")
    sns.set_style("white")
    sns.set_theme(font_scale=0.7)
    sns.set_color_codes()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(vote_df, center=0, cmap="RdYlBu", ax=ax)
    plt.show()
