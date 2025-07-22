# API Reference

## `reddwarf.implementations.base`

### ::: reddwarf.implementations.polis.run_pipeline
    options:
        show_root_heading: true

## `reddwarf.implementations.polis`

### ::: reddwarf.implementations.polis.run_pipeline
    options:
        show_root_heading: true

## `reddwarf.sklearn`

Various custom Scikit-Learn estimators to mimick aspects of Polis, suitable for
use in Scikit-Learn workflows, pipelines, and APIs.

### ::: reddwarf.sklearn.cluster.PolisKMeans
    options:
        show_root_heading: true
        docstring_style: numpy

### ::: reddwarf.sklearn.cluster.PolisKMeansDownsampler
    options:
        show_root_heading: true

### ::: reddwarf.sklearn.model_selection.GridSearchNonCV
    options:
        show_root_heading: true

### ::: reddwarf.sklearn.transformers.SparsityAwareScaler
    options:
        show_root_heading: true

## `reddwarf.utils.matrix`

### ::: reddwarf.utils.matrix.generate_raw_matrix
    options:
        show_root_heading: true

### ::: reddwarf.utils.matrix.simple_filter_matrix
    options:
        show_root_heading: true

### ::: reddwarf.utils.matrix.get_clusterable_participant_ids
    options:
        show_root_heading: true

## `reddwarf.utils.reducer`

### ::: reddwarf.utils.reducer.base.run_reducer
    options:
        show_root_heading: true

### ::: reddwarf.utils.reducer.base.get_reducer
    options:
        show_root_heading: true

## `reddwarf.utils.reducer.registry`

### ::: reddwarf.utils.reducer.registry.register_reducer
    options:
        show_root_heading: true

### ::: reddwarf.utils.reducer.registry.get_reducer
    options:
        show_root_heading: true

### ::: reddwarf.utils.reducer.registry.list_reducers
    options:
        show_root_heading: true

## `reddwarf.utils.clusterer`

### ::: reddwarf.utils.clusterer.base.run_clusterer
    options:
        show_root_heading: true

### ::: reddwarf.utils.clusterer.kmeans.find_best_kmeans
    options:
        show_root_heading: true

## `reddwarf.utils.consensus`

### ::: reddwarf.utils.consensus.select_consensus_statements
    options:
        show_root_heading: true

## `reddwarf.utils.stats`

### ::: reddwarf.utils.stats.select_representative_statements
    options:
        show_root_heading: true

### ::: reddwarf.utils.stats.calculate_comment_statistics
    options:
        show_root_heading: true

### ::: reddwarf.utils.stats.calculate_comment_statistics_dataframes
    options:
        show_root_heading: true

## `reddwarf.utils`

(These are in the process of being either moved or deprecated.)

### ::: reddwarf.utils.filter_votes
    options:
        show_root_heading: true

### ::: reddwarf.utils.filter_matrix
    options:
        show_root_heading: true

### ::: reddwarf.utils.get_unvoted_statement_ids
    options:
        show_root_heading: true

## `reddwarf.data_presenter`

### ::: reddwarf.data_presenter.generate_figure
    options:
        show_root_heading: true

### ::: reddwarf.data_presenter.generate_figure_polis
    options:
        show_root_heading: true


## Types

### ::: reddwarf.implementations.base.PolisClusteringResult
    options:
        show_root_heading: true

