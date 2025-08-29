from typing import Any
import pytest
import json
import re
from dataclasses import dataclass
from tests import helpers

REPORT_ID_RE = r"^r[a-z0-9]{20}$"
CONVO_ID_RE = r"^\d[a-z0-9]{4,9}$"

@dataclass
class PolisFixtureResult:
    math_data: Any
    data_dir: str
    filename: str
    keep_participant_ids: list[int] | list[str]

@pytest.fixture
def polis_convo_data(request):
    """
    Allows parametrized testing against curated local datasets, and remote
    datasets.

    - If the parameter does match a dataset descriptor, it's selected for use.
    - If the parameter doesn't match, but it matches the format of either a (1)
    report ID or (2) conversation ID on the Polis platform, this fixture will
    try to download that conversation's data for testing.
    """
    # We make this available to parametrized tests with indirect=True can still
    # make decisions based on what fixture is running.
    request.node._param = request.param

    path = None
    keep_participant_ids = []

    if request.param in ["small", "small-no-meta"]:
        # See: https://pol.is/4cvkai2ctw
        # See: https://pol.is/report/r6bpmcmizi2kyvhzkhfr7
        # 23 ptpts, 2 groups, 0/9 meta, strict=yes
        path = "tests/fixtures/below-100-ptpts"
        # We hardcode this because Polis has some bespoke rules that keep these IDs in for clustering.
        # TODO: Try to determine why each pid is kept. Can maybe determine by incrementing through vote history.
        #  5 -> 1 vote @ statement #26 (participant #2's)
        # 10 -> 2 vote @ statements #21 (participant #1's) & #29 (their own, moderated in).
        # 11 -> 1 vote @ statement #29 (participant #10's)
        # 14 -> 1 vote @ statement #27 (participant #6's)
        keep_participant_ids = [ 5, 10, 11, 14 ]
    elif request.param == "small-no-meta-bad":
        # BUG: DATA INTEGRITY ISSUES.
        # Missing votes via API and so doesn't match user_vote_count
        # See: https://pol.is/6ukkcvfbre
        # See: https://pol.is/report/r64ajcsmp9butzxhzj44c
        # 51 ptpts, 3 groups, 0/29 meta, strict=yes
        path = "tests/fixtures/50ptpt-3gp-strict-no-meta"
    elif request.param in ["small-with-meta", "small-with-mod-out-votes"]:
        # The final mod-out statement has votes, so will affect calculations if
        # removed from list and re-entered into calculations.
        #
        # See: https://pol.is/2dhnep37ie
        # See: https://pol.is/report/r6ipxzfudddppwesbmtmn
        # 27 ptpts, 3 groups, 4/57 meta, strict=no
        path = "tests/fixtures/25ptpt-3gp-no-strict-with-meta"
        keep_participant_ids = []
    elif request.param in ["medium", "medium-with-meta"]:
        # See: https://pol.is/3ntrtcehas
        # See: https://pol.is/report/r68fknmmmyhdpi3sh4ctc
        # 234 ptpts, 4 groups, 2/53 meta, strict=no
        path = "tests/fixtures/above-100-ptpts"
    elif request.param in ["medium-no-meta", "medium-no-mod-out"]:
        # See: https://pol.is/4asymkcrjf
        # See: https://pol.is/report/r4zdxrdscmukmkakmbz3k
        # 160 ptpts, 3 groups, 0/118 meta, strict=no
        path = "tests/fixtures/150ptpt-3gp-no-strict-no-meta"
    elif request.param == "6da26haxpe":
        keep_participant_ids = [ 4, 5, 6, 10 ]
        # sign_flip_keys = ["pca.center", "pca.comment-projection[0]", "base-clusters.x", "group-clusters[*].center[0]", "pca.comps[1]"]

    if path:
        filename = "math-pca2.json"
        with open(f"{path}/{filename}", 'r') as f:
            data = json.load(f)

        yield PolisFixtureResult(
            math_data=data,
            data_dir=path,
            filename=filename,
            keep_participant_ids=keep_participant_ids,
        )
        return

    # If no match of specific local fixture, see if this looks like remote Polis data.
    if re.match(REPORT_ID_RE, request.param) or re.match(CONVO_ID_RE, request.param):
        from reddwarf.data_loader import Loader
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Download remote data based on Polis report or conversation ID.
            loader = Loader(polis_id=request.param, output_dir=tmpdirname)

            filename = "math-pca2.json"
            yield PolisFixtureResult(
                math_data=loader.math_data,
                data_dir=tmpdirname,
                filename=filename,
                keep_participant_ids=loader.math_data["in-conv"],
            )
            # Tmpdir cleanup will happen after yield completes
            return

    raise ValueError("No fixture matching local or remote fixture data found")

@pytest.fixture(params=[True, False], ids=["with_id_strings", "without_id_strings"])
def convert_ids(request):
    """Optionally converts IDs to strings using helpers.convert_ids_to_strings."""
    def _apply(votes):
        if request.param:
            return helpers.convert_ids_to_strings(votes)
        return votes
    return _apply