import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytest
from tests.fixtures import polis_convo_data
from reddwarf.utils import pca as PcaUtils
from reddwarf.utils import matrix as MatrixUtils
from reddwarf.polis import PolisClient


def test_run_pca_toy():
    # Pre-calculated
    expected_projections = [
        [ 1.238168, -0.173377],
        [-0.998131, -0.268031],
        [-0.123951,  0.205794],
        [-0.116086,  0.235614],
    ]
    expected_eigenvectors = [
        [0.87418057, -0.48400607, -0.0393249],
        [0.47382435,  0.86790524, -0.1491005],
    ]
    expected_eigenvalues = [ 0.85272226, 0.06658833 ]

    initial_matrix = pd.DataFrame(
        [
            [ 1, 0,  0],
            [-1, 1,  0.1],
            [ 0, 1,  0.1],
            [ 0, 1, -0.1],
        ],
        columns=[0, 1, 2], # statement_ids
        index=[0, 1, 2, 3], # participant_ids
        dtype=float,
    )
    projected_data, eigenvectors, eigenvalues, *_ = PcaUtils.run_pca(vote_matrix=initial_matrix)
    assert_allclose(projected_data.values, expected_projections, rtol=10**-5)
    assert_allclose(eigenvectors, expected_eigenvectors, rtol=10**-5)
    assert_allclose(eigenvalues, expected_eigenvalues, rtol=10**-5)

@pytest.mark.parametrize("polis_convo_data", ["small", "small-with-meta", "medium"], indirect=True)
def test_run_pca_real_data(polis_convo_data):
    math_data, data_path, *_ = polis_convo_data
    expected_pca = math_data["pca"]

    # Just fetch the moderated out statements from polismath (no need to recalculate here)
    statement_ids_mod_out = math_data["mod-out"]

    client = PolisClient()
    client.load_data(filepaths=[f"{data_path}/votes.json"])
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=client.data_loader.votes_data)

    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        statement_ids_mod_out=statement_ids_mod_out,
    )

    _, actual_components, _, actual_means = PcaUtils.run_pca(vote_matrix=real_vote_matrix)

    # We test absolute because PCA methods don't always give the same sign, and can flip.
    assert np.absolute(actual_components[0]) == pytest.approx(np.absolute(expected_pca["comps"][0]))
    assert np.absolute(actual_components[1]) == pytest.approx(np.absolute(expected_pca["comps"][1]))

    assert np.absolute(actual_means) == pytest.approx(np.absolute(expected_pca["center"]))

# TODO: Find a good place to run integration tests against real remote datasets.
@pytest.mark.skip
def test_run_pca_real_data_testing():
    client = PolisClient()
    # client.load_data(report_id="r6iwc8tb22am6hpd8f7c5") # 20 voters, 0/146 meta WORKS
    # client.load_data(report_id="r5hr48j8y8mpcffk7crmk") # 27 voters, 11/63 meta WORKS strict=no
    # client.load_data(report_id="r7dr5tzke7pbpbajynkv8") #  69 voters, 74/133 meta NOPE 2/133 mismatch strict=yes
    # client.load_data(report_id="r8jhyfp54cyanhu26cz3v") #  90 voters, 0/51 meta WORKS strict=no
    # client.load_data(report_id="r2dtdbwbsrzu8bj8wmpmc") #  87 voters, 3/92 meta WORKS strict=no
    # client.load_data(report_id="r9cnkypdddkmnefz8k2bn") # 113 voters, 0/40 meta WORKS strict=no
    # client.load_data(report_id="r4zurrpweay6khmsaab4e") # 108 voters, 0/82 meta WORKS strict=yes
    # client.load_data(report_id="r5ackfcj5dmawnt2de7x7") # 109 voters, 16/95 meta NOPE DIFF SHAPES? bug?
    # client.load_data(report_id="r6bpmcmizi2kyvhzkhfr7") # 23 voters, 0/9 meta WORKS strict=yes
    # client.load_data(report_id="r4tfrhexm3dhawcfav6dy") # 116 voters, 0/170 meta WORKS
    # client.load_data(report_id="r7f89nk32sjuknecjbrxp") # 135 voters, 0/52 meta WORKS
    # client.load_data(report_id="r2xzujbewezpjdc9pmzhd") # 135 voters, 3/48 meta WORKS
    # client.load_data(report_id="r8bzudrhs8j6petppicrj") # 140 voters, 0/46 meta WORKS
    # client.load_data(report_id="r4zdxrdscmukmkakmbz3k") # 145 voters, 0/117 meta WORKS
    client.load_data(report_id="r45ru4zfmutun54ttskne") # 336 voters, 15/148 meta NOPE 76/148 mismatch
    # client.load_data(report_id="") # x voters, x/x meta ...
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=client.data_loader.votes_data)
    math_data = client.data_loader.math_data
    expected_pca = math_data["pca"]

    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        statement_ids_mod_out=math_data["mod-out"],
    )

    _, actual_components, _, actual_means = PcaUtils.run_pca(vote_matrix=real_vote_matrix)
    # powerit_pca = PcaUtils.powerit_pca(real_vote_matrix.values)

    # We test absolute because PCA methods don't always give the same sign, and can flip.
    assert np.absolute(actual_components[0]) == pytest.approx(np.absolute(expected_pca["comps"][0]))
    assert np.absolute(actual_components[1]) == pytest.approx(np.absolute(expected_pca["comps"][1]))

    assert np.absolute(actual_means) == pytest.approx(np.absolute(expected_pca["center"]))

@pytest.mark.skip
def test_scale_projected_data():
    raise

@pytest.mark.parametrize("polis_convo_data", ["small", "small-with-meta", "small-no-meta", "medium"], indirect=True)
def test_with_proj_and_extremity(polis_convo_data):
    math_data, _, _ = polis_convo_data
    expected_pca = math_data["pca"]

    pca = {
        "center": math_data["pca"]["center"],
        "comps": math_data["pca"]["comps"],
    }

    calculated_pca = PcaUtils.with_proj_and_extremity(pca)

    assert expected_pca["comment-projection"] == calculated_pca["comment-projection"]
    assert expected_pca["comment-extremity"] == calculated_pca["comment-extremity"]