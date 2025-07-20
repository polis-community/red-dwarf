import pytest
import numpy as np
from reddwarf.implementations.base import run_pipeline
from reddwarf.utils import matrix as MatrixUtils
from reddwarf.data_loader import Loader
from tests.fixtures import polis_convo_data
from reddwarf.utils.string_id import preprocess_votes

@pytest.mark.parametrize("reducer", ["pca", "pacmap", "localmap"])
@pytest.mark.parametrize(
    "polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True
)
def test_run_pipeline_with_statement_and_participant_id_casting(reducer,polis_convo_data):
    """Test that run_pipeline supports string statement and participant id's and outputs expected data"""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = loader.votes_data

    # Preprocess votes to convert participant and statement IDs to strings
    preprocessed_vote_data = preprocess_votes(votes)

    # preprocessed clustering run_pipeine
    preprocessed_clustering_result = run_pipeline(
       votes = preprocessed_vote_data,
       reducer=reducer
    )

    # Get preprocessed vote dict
    preprocessed_vote_matrix = preprocessed_clustering_result.raw_vote_matrix.count(axis="columns").to_dict()

    # Expected values from preprocessed fixture
    preprocessed_mathdata_results = fixture.math_data["user-vote-counts"]
    preprocessed_mathdata_results = {str(k)+"p":v for k,v in preprocessed_mathdata_results.items()}

    assert preprocessed_vote_matrix == preprocessed_mathdata_results





   
