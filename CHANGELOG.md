# Changelog

## [Unreleased][] (YYYY-MM-DD)

### Fixes
- Allow `is_strict_moderation` to be inferred from not just API data, but file data.

### Changes
- Fixed participant projections to map more closely to Polis with `utils.pca.sparsity_aware_project_ptpt()`.
- Add simple Polis implementation in `reddwarf.implementations.polis`.

### Chores
- Moved agora implementation from `reddwarf.agora` to `reddwarf.implementations.agora` (deprecation warning).
- Add missing `conversation.json` fixture file.
- Extract statement processing from polis class-based client to pure util function.

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
   [Unreleased]: https://github.com/polis-community/red-dwarf/compare/v0.2.0...main
   [0.2.0]: https://github.com/polis-community/red-dwarf/releases/tag/v0.2.0
   [0.1.1]: https://github.com/polis-community/red-dwarf/releases/tag/v0.1.1
   [0.1.0]: https://github.com/polis-community/red-dwarf/releases/tag/v0.1.0
