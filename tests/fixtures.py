import pytest
import json

@pytest.fixture
def polis_convo_data(request):
    if request.param in ["small", "small-no-meta"]:
        # See: https://pol.is/4cvkai2ctw
        # See: https://pol.is/report/r6bpmcmizi2kyvhzkhfr7
        # 23 ptpts, 2 groups, 0/9 meta, strict=yes
        path = "tests/fixtures/below-100-ptpts"
    elif request.param == "small-no-meta-bad":
        # BUG: DATA INTEGRITY ISSUES.
        # Missing votes via API and so doesn't match user_vote_count
        # See: https://pol.is/6ukkcvfbre
        # See: https://pol.is/report/r64ajcsmp9butzxhzj44c
        # 51 ptpts, 3 groups, 0/29 meta, strict=yes
        path = "tests/fixtures/50ptpt-3gp-strict-no-meta"
    elif request.param == "small-with-meta":
        # See: https://pol.is/2dhnep37ie
        # See: https://pol.is/report/r6ipxzfudddppwesbmtmn
        # 27 ptpts, 3 groups, 4/57 meta, strict=no
        path = "tests/fixtures/25ptpt-3gp-no-strict-with-meta"
    elif request.param in ["medium", "medium-with-meta"]:
        # See: https://pol.is/3ntrtcehas
        # See: https://pol.is/report/r68fknmmmyhdpi3sh4ctc
        # 234 ptpts, 4 groups, 2/53 meta, strict=no
        path = "tests/fixtures/above-100-ptpts"
    elif request.param == "medium-no-meta":
        # See: https://pol.is/4asymkcrjf
        # See: https://pol.is/report/r4zdxrdscmukmkakmbz3k
        # 160 ptpts, 3 groups, 0/118 meta, strict=no
        path = "tests/fixtures/150ptpt-3gp-no-strict-no-meta"
    else:
        raise ValueError("No directory set for loading fixture data")

    filename = "math-pca2.json"
    with open(f"{path}/{filename}", 'r') as f:
        data = json.load(f)

    return data, path, filename
