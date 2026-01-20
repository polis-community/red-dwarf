# Changelog

## [Unreleased][] (YYYY-MM-DD)

_No changes yet._

## [0.4.0][] (2026-01-19)

### Changes

- Add `select_consensus_statements()` function, and wire into Polis implementation.
- Allow `calculate_comment_statistics()` to work without groups/labels.
- Generalize `format_comment_stats()` to work for group and consensus statements.
- Add `select_representative_statements()` to PolisClusteringResult as `repness` key.
- Rename arg `pick_n` to `pick_max` in `select_consensus_statements()`, for clarity and consistency.
- Slight change to PolisRepness type, so group IDs now returned as ints.
- Add `print_selected_statements()` presenter for inspecting `PolisClusteringResult`.
- Add `print_consensus_statements()` presenter for inspecting `PolisClusteringResult`.
- Allow `pick_max` and `confidence` interval args to be set in `polis.run_clustering()`.
- Allow `get_corrected_centroid_guesses()` to unflip each axis if correction not needed.
- Abstracted reducer and clusterer algorithm support.
  - Added support for pacmap/localmap beyond PCA.
  - Added support for HDBSCAN clustering beyond KMeans.
  - Allow passing of arbitary params into reducer/clusterer.
- Remove support for `polis_legacy` implementation (PolisClient).
- Added disagree variant of group-informed-consensus. (`group-informed-consensus-disagree`)
- Brought `group-informed-consensus` metrics to top-level result object.
- Renamed `run_clustering` function to `run_pipeline` and created base pipeline implementation.
- Add option to generate_figure_polis to configure showing pid labels (`show_pids`).
- Remove deprecated methods from doc website.
- Remove deprecated modules from prior import paths.
- Avoid using dataframes in a few low level util function, in favour of numpy arrays.
- Rename `projected_{participants,statements}` to `{participant,statement}_projections` in run_pipeline results. Also coords keyed to ID, instead of dataframes.
- Remove agora implementation and tests. ([#73](https://github.com/polis-community/red-dwarf/issues/74))
- Migrate from reference HDBSCAN module (in `scikit-learn`) to full-featured HDBSCAN* package.
- Add dependency groups to avoid installing everything. ([#11](https://github.com/polis-community/red-dwarf/issues/11))
- Add support for statement and participant IDs to be strings. ([`#92`](https://github.com/polis-community/red-dwarf/pull/92))
- Added `BestPolisKMeans` estimator. ([`#111`](https://github.com/polis-community/red-dwarf/pull/111))

### Fixes

- Handle when `is-meta` and `is-seed` columns arrive in CSV import.
  [`#55`](https://github.com/polis-community/red-dwarf/issues/55)
- Handle loading comments data from API when `is_meta` missing in CSV import.
- Only pass unique labels into `generate_figure()` colorbar.
- bugfix: `clusterer_kwargs` and `reducer_kwargs` were not being pass through `run_pipeline()`.
- bugfix: Ensure `run_pipeline()` passes `random_state` to reducer.
- bugfix: Fix overly constrained versions from [#80](https://github.com/polis-community/red-dwarf/issues/80).
- bugfix: Ensure we don't crash when a participant ID in `keep_participant_ids` doesn't exist in vote matrix. ([`#100`](https://github.com/polis-community/red-dwarf/pull/100))
- bugfix: Relaxed `weight_x_32767` field to accept alternative responses (voxit server). ([`#107`](https://github.com/polis-community/red-dwarf/pull/107))

### Chores

- Update the release process instructions.
- Added `simulate_api_response()` test helper for easier comparison with polismath output.
- Bumped minimum supported Python version from 3.8 to 3.10. ([`#94`](https://github.com/polis-community/red-dwarf/pull/94))
- Added CI testing for all of Python 3.10, 3.13. ([`#93`](https://github.com/polis-community/red-dwarf/pull/93))
- Enforced a build of website and notebooks in pull requests. ([`#96`](https://github.com/polis-community/red-dwarf/pull/96))
- Added support for test fixtures that pull data from live polis convos. ([`#98`](https://github.com/polis-community/red-dwarf/pull/98))
- Added data loader to docs website. ([`#101`](https://github.com/polis-community/red-dwarf/pull/101))
- Sync'd sanity-check API tests with new Pol.is server response fields. ([`#106`](https://github.com/polis-community/red-dwarf/pull/106))
- Resolved deprecated method in HDBSCAN. ([`#109`](https://github.com/polis-community/red-dwarf/pull/109))
- Removed CloudflareBypassHTTPAdapter that is no longer necessary. ([`#108`](https://github.com/polis-community/red-dwarf/pull/108))
- Extracted `Exporter` class out of data_loader into data_exporter. ([`#100`](https://github.com/polis-community/red-dwarf/pull/110))
- Updated HDBSCAN to allow using scikit-learn >=1.8. ([`#112`](https://github.com/polis-community/red-dwarf/issues/112))
- Set minimum for version for PaCMAP's Numba. Added back CI testing for Python 3.14. ([`#112`](https://github.com/polis-community/red-dwarf/pull/114))


## [0.3.0][] (2025-04-29)

### Fixes

- Allow `is_strict_moderation` to be inferred from not just API data, but file data.
- Better handle numpy divide-by-zero edge-cases in two-property test. ([#28](https://github.com/polis-community/red-dwarf/pull/28))
- Fix bug where `vote_matrix` was modified directly, leading to subtle side-effects.
- Fix bug in `select_representative_statements()` where mod-out statements weren't ignored.

### Changes

- Fixed participant projections to map more closely to Polis with `utils.pca.sparsity_aware_project_ptpt()`.
- Add simple Polis implementation in `reddwarf.implementations.polis`.
- Add singular `polis_id` arg as recommended way to download (auto-detect `report_id` vs `converation_id`).
- Calculate group-aware consensus stats. ([#28](https://github.com/polis-community/red-dwarf/pull/28))
- Removed `scale_projected_data()` in `PolisClient` (now happens in `run_pca()`).
- Deprecate `PolisClient()`.
- Add `inverse_transform()` to `SparsityAwareScaler`.
- Add data loader support for local math data files.
- Add support to easily flip signs in `generate_figure()`.
- Modify `generate_figure()` to accept more effective args.
  - Use numpy args of `coord_data`, `coord_labels` and `cluster_labels`
    individually, rather than using DataFrames.
  - Allow passing extra `coord_data` beyond what's labelled.
- Add automatic padding to polis implementation when cluster centroid guesses are provided.
- Add `PolisKMeans` scikit-learn estimator with:
  - cluster initialization strategy matching Polis,
  - new `init_centers` argument with more versatility for being given more/less guesses than needed, and
  - new instance variable `init_centers_used_` to allow inspection of guesses used.
- Allow passing KMeans `init` strategy into `find_optimal_k()`.
- Remove `pad_centroid_list_to_length` helper function.
- Add `GridSearchNonCV` to find optimal K via silhouette scores.
- For interal util functions, replace `max_group_count` args with `k_bounds` for upper and lower k bounds.
- Add `PolisKMeansDownsampler` transformer to support base clustering.
- Update `get_corrected_centroid_guesses()` to also extract from base clusters.
- Remove extraneous return values from `PolisClusteringResult`.
- Add `data_presenter.generate_figure_polis()` for making graphs from PolisClusteringResult.
- Add `group_aware_consensus` dataframe to PolisClusteringResult of polis implementation.
- Add group statement stats to MultiIndex DataFrame.
- Add `reddwarf.data_presenter.print_repress()` for printing representative statements.
- Add support for `Loader()` importing data from alternative Polis instances via `polis_instance_url` arg.
- Patch sklearn with a simple `PatchedPipeline`, to allow pipeline steps to access other steps.
- Modify `SparsityAwareScaler` to be able to use captured output from SparsityAware Capture.
- Remove ported Polis PCA functions that are no longer used.
- Remove old `impute_missing_votes()` function that's no longer used.
- In `PolisClusteringResult`, created new `statements_df` and `participants_df` with all raw calculation values.

### Chores

- Moved agora implementation from `reddwarf.agora` to `reddwarf.implementations.agora` (deprecation warning).
- Add missing `conversation.json` fixture file.
- Extract statement processing from polis class-based client to pure util function.
- Add types to fully describe polismath object. ([#28](https://github.com/polis-community/red-dwarf/pull/28))
- Add new fixture for large convo without meta statements. ([#28](https://github.com/polis-community/red-dwarf/pull/28))
- Add ability to filter unit tests and avoid running whole suite. ([#44](https://github.com/polis-community/red-dwarf/pull/44))
- Improve test fixture to download remote Polis data.
- Add helper to support simple sign-flips in Polis test data.
- Remove usage of PolisClient in tests, in favour of [data] Loader.
- Start storing `keep_participant_ids` in fixtures.
- Add solid unit test for expected variance, which is stablest measure we can derive.
- Use dataclasses for `polis_convo_data` test fixture.
- Add `utils.polismath.get_corrected_centroid_guesses()` to initiate centroid guesses from Polis API.
- Remove unused `init_cluster()` helper.

## [0.2.0][] (2025-03-24)

### Fixed

- Relax seaborn version constraint to be compatible with TabPFN. ([#16](https://github.com/polis-community/red-dwarf/issues/16))
- Data loader was not downloading last participant's votes, so most PCA results slightly off. ([#29](https://github.com/polis-community/red-dwarf/issues/29))

### Changes

- Implement `utils.calculate_representativeness()` function. ([#22](https://github.com/polis-community/red-dwarf/issues/22))
- Add color legend for labels in `data_presenter.generate_figure()`. [`d55f535`](https://github.com/polis-community/red-dwarf/pull/24/commits/d55f53588de72620abb984d7c1ac27f8a31d5478) ([#22](https://github.com/polis-community/red-dwarf/issues/22))
- Implement calculations of all comment statistics. ([#25](https://github.com/polis-community/red-dwarf/pull/25))
- Implement `utils.select_representative_statements()` to reproduce polismath output. ([#25](https://github.com/polis-community/red-dwarf/pull/25))
- Migrate from `red-dwarf-democracy` PyPI project namespace to `red-dwarf`.

### Chores

- Restructure `utils.py` into separate files. ([#26](https://github.com/polis-community/red-dwarf/pull/26))
- Add unit tests for `utils.run_pca()` to test against real polismath data.
- Add unit tests for `agora.run_clustering()`.
- Parametrize unit tests for real polis convo data.
- Add testing for notebook examples. ([#34](https://github.com/polis-community/red-dwarf/pull/34))

## [0.1.1][] (2025-03-04)

### Bugfixes

- Fix publishing issue with missing license file. ([#19](https://github.com/polis-community/red-dwarf/issues/19))
  - Workaround for [`pypa/setuptools#4769`](https://github.com/pypa/setuptools/issues/4759).
- Change package name from `reddwarf` to `red-dwarf-democracy`.

## [0.1.0][] (2025-03-04)

- Add Agora/ZKorum to README as sponsor.
- Add low-level stateless functions in `reddwarf.utils`.
  - Add first-pass unit test coverage, and CI.
- Add mid-level stateless `reddwarf.agora` implementation.
  - Add preferred types/modules and function definitions.
- Add code coverage reports for unit tests.
- Document library usage in Jupyter notebooks.
- Create documentation website using `mkdocs`.
- Add `make` targets for common development tasks, and default help target.
- Experimental
  - Add high-level stateful Polis client class implementation.
  - Add high-level stateful data loader class.
    - Support loading from JSON API, CSV export API, local files, and remote directory.
    - Support rate-limiting.
    - Support request caching.
    - Support bypass of Cloudflare for all API endpoints.
  - Add high-level stateful data presenter class.
  - Add integration tests.

<!-- Links -->

[Unreleased]: https://github.com/polis-community/red-dwarf/compare/v0.4.0...main
[0.4.0]: https://github.com/polis-community/red-dwarf/releases/tag/v0.4.0
[0.3.0]: https://github.com/polis-community/red-dwarf/releases/tag/v0.3.0
[0.2.0]: https://github.com/polis-community/red-dwarf/releases/tag/v0.2.0
[0.1.1]: https://github.com/polis-community/red-dwarf/releases/tag/v0.1.1
[0.1.0]: https://github.com/polis-community/red-dwarf/releases/tag/v0.1.0
