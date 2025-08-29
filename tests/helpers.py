from datetime import timedelta
from enum import Enum
import os
from requests_ratelimiter import SQLiteBucket
from sklearn.impute import SimpleImputer
from reddwarf.helpers import CachedLimiterSession
from reddwarf.types.polis import PolisRepness
from typing import Any, Union, Literal
import numpy as np
import json


def get_grouped_statement_ids(
    repness: PolisRepness,
) -> dict[str, list[dict[str, list[int]]]]:
    """A helper to compare only tid in groups, rather than full repness object."""
    groups = []

    for key, statements in repness.items():
        group = {
            "id": str(key),
            "members": sorted([stmt["tid"] for stmt in statements]),
        }  # type:ignore
        groups.append(group)

    return {"groups": groups}


def transform_base_clusters_to_participant_coords(base_clusters):
    """
    Transform base clusters data into a list of dictionaries with participant_id and xy coordinates.

    Args:
        base_clusters (dict): A dictionary containing base clusters data with 'id', 'members', 'x', and 'y' keys.

    Returns:
        list: A list of dictionaries, each containing a participant_id and their xy coordinates.
    """

    # For now, ensure failure if a base-cluster has more than one member, as the test assumes that.
    def get_only_member_or_raise(members):
        if len(members) != 1:
            raise Exception("A base-cluster has more than one member when it cannot")

        return members[0]

    return [
        {"participant_id": get_only_member_or_raise(members), "xy": [x, y]}
        for members, x, y in zip(
            base_clusters["members"], base_clusters["x"], base_clusters["y"]
        )
    ]


# Not used right now. Maybe later.
def groupsort_pids_by_cluster(df):
    """
    Helper function to gather statement IDS in clusters and sort them for easy
    comparison.

    This make comparison easy, even when kmeans gives different numeric labels.

    Args:
        df (pd.DataFrame): A dataframe with projected participants, columns "x",
        "y", "cluster_id"
    Returns:
        (list[list[int]]): A list of lists, each containing statement IDs in a
        cluster.
    """
    # Group by cluster_id and collect indices
    grouped = df.groupby("cluster_id").apply(lambda x: list(x.index))

    # Sort the groups by their length (number of members) in descending order
    sorted_groups = sorted(grouped, key=len, reverse=True)

    # Convert each inner list to integers
    return [list(map(int, group)) for group in sorted_groups]


NestedValue = Union[int, float, list["NestedValue"], dict[str, "NestedValue"]]
NestedDict = dict[str, NestedValue]


def flip_signs_by_key(nested_dict: NestedDict, keys: list[str] = []) -> Any:
    """
    Flips the signs of numeric values in a nested dict using dot-notation paths.
    Supports nested arrays, array indexing (like "foo.bar[0].baz[1]"), and array
    wildcards (like "foo.bar[*].baz).

    This helper is for quickly dealing with real polismath fixture data that has
    different signs than we calculate ourselves.

    NOTE: The need for this helper may be harmless artifacts of PCA methods, or
    other reasons related to agree/disagree signs. Consistently adjusting signs
    at the fixture level should help clarify this.

    Arguments:
        obj (NestedDict): A nested dict of arbitrary depth, with lists of float
            values at some keys. keys (list[str]): A list of dot-notation keys to
            flip signs within.

    Returns:
        Any: A nested dict with all the same original types, but with
            specific keys inverted. (Typed as "Any" for convenience.)
    """
    import copy
    import re

    result: NestedDict = copy.deepcopy(nested_dict)

    def flip_recursive(value: NestedValue) -> NestedValue:
        if isinstance(value, list):
            return [flip_recursive(v) for v in value]
        elif isinstance(value, (int, float)):
            return -value
        return value

    def parse_path_segment(segment: str) -> list[Union[str, int, Literal["*"]]]:
        parts: list[Union[str, int, Literal["*"]]] = []
        for match in re.finditer(r"([^\[\]]+)|\[(\d+|\*)\]", segment):
            key, idx = match.groups()
            if key:
                parts.append(key)
            elif idx == "*":
                parts.append("*")
            else:
                parts.append(int(idx))
        return parts

    def resolve_targets(
        root: NestedValue, path: str
    ) -> list[tuple[Union[dict, list], Union[str, int]]]:
        segments = path.split(".")
        parts: list[Union[str, int, Literal["*"]]] = []
        for seg in segments:
            parts.extend(parse_path_segment(seg))

        def recurse(
            current: NestedValue, remaining: list[Union[str, int, Literal["*"]]]
        ) -> list[tuple[Union[dict, list], Union[str, int]]]:
            if not remaining:
                return []

            part, *rest = remaining

            if len(rest) == 0:
                # Final step: return the parent + final key/index
                if isinstance(part, (str, int)):
                    return [(current, part)]
                elif part == "*":
                    if isinstance(current, list):
                        return [(current, i) for i in range(len(current))]
                return []

            if isinstance(part, str) and isinstance(current, dict) and part in current:
                return recurse(current[part], rest)

            elif (
                isinstance(part, int)
                and isinstance(current, list)
                and 0 <= part < len(current)
            ):
                return recurse(current[part], rest)

            elif part == "*" and isinstance(current, list):
                results = []
                for item in current:
                    results.extend(recurse(item, rest))
                return results

            return []

        return recurse(root, parts)

    for dot_key in keys:
        targets = resolve_targets(result, dot_key)
        for parent, final_key in targets:
            if (
                isinstance(final_key, str)
                and isinstance(parent, dict)
                and final_key in parent
            ):
                parent[final_key] = flip_recursive(parent[final_key])
            elif (
                isinstance(final_key, int)
                and isinstance(parent, list)
                and 0 <= final_key < len(parent)
            ):
                parent[final_key] = flip_recursive(parent[final_key])

    return result


def calculate_explained_variance(sparse_vote_matrix, means, components):
    """
    Derive explained variance from simpler polismath outputs.

    Explained variance is not sign-dependant, so great for unit tests.

    Arguments:
        sparse_vote_matrix (np.ndarray): Sparse vote_matrix 2D numpy array
        means (list[float]): List of feature means
        components (list[list[float]]): List of eigenvectors/components

    Returns:
        list[float]: Variance explained by each component (n_components,)
    """
    X_imputed = SimpleImputer().fit_transform(sparse_vote_matrix)
    means, comps = [np.asarray(arr) for arr in [means, components]]
    X_centered = X_imputed - means
    X_projected = X_centered @ comps.transpose()
    explained_variance = np.var(X_projected, axis=0, ddof=1)

    return explained_variance


def simulate_api_response(data):
    """
    Simulates a python object sent through JSON via REST API.

    Main change is that any int keys in python dictionary objects are converted
    to string keys, because all keys are strings in JSON.
    """
    return json.loads(json.dumps(dict(data)))

class ReportType(Enum):
    SUMMARY = "summary"
    VOTES = "votes"
    COMMENTS = "comments"
    PARTICIPANT_VOTES = "participant-votes"
    COMMENT_GROUPS = "comment-groups"

def fetch_csv(type: ReportType, output_dir, report_id):
    print(f"Downloading CSVs from remote server to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Source: https://github.com/JWCook/requests-ratelimiter/tree/main?tab=readme-ov-file#custom-session-example-requests-cache
    session = CachedLimiterSession(
        per_second=5,
        expire_after=timedelta(hours=1),
        cache_name="test_cache.sqlite",
        bucket_class=SQLiteBucket,
        bucket_kwargs={
            "path": "test_cache.sqlite",
            'isolation_level': "EXCLUSIVE",
            'check_same_thread': False,
        },
    )

    with open(f"{output_dir}/{report_id}_{type.value}.csv", 'w') as f:
        r = session.get(f"https://pol.is/api/v3/reportExport/{report_id}/{type.value}.csv")
        f.write(r.text)
    return f

def convert_ids_to_strings(data_list: list) -> list:
    """Convert integer statement and participant IDs to string IDs"""
    for item in data_list:
        if "participant_id" in item:
            item["participant_id"] = str(item["participant_id"])
        if "statement_id" in item:
            item["statement_id"] = str(item["statement_id"])
    return data_list

def convert_participant_ids_to_strings(participant_ids: list[int]) -> list[str]:
    """ Convert a list of participant IDs (int) to a list of string IDs"""
    return ["p{}".format(str(pid)) for pid in participant_ids]