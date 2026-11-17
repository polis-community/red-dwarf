# Red Dwarf: A Pol.is-like library

&nbsp;&nbsp;
⚫⋆✦⋆⭑⋆⋆⋆⋆⋆
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![PyPI - Version](https://img.shields.io/pypi/v/red-dwarf)][pypi]
[![Run Tests](https://github.com/polis-community/red-dwarf-democracy/actions/workflows/test.yml/badge.svg)](https://github.com/polis-community/red-dwarf-democracy/actions/workflows/test.yml)
[![Deploy Docs](https://github.com/polis-community/red-dwarf/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/polis-community/red-dwarf/actions/workflows/deploy-docs.yml)
--------

A <em>DIM</em>ensional <em>RED</em>uction library for reproducing and experimenting with Polis-like data pipelines.

> Red dwarf stars are dim red stars. They are hard to see, but are the most common
> type -- the quiet 70% majority. They burn slow, using fuel efficiently,
> making them the longest-living stars in the universe. They'll be around until
> the last light, even supporting habitable planets for billions of years.
> If there's persistant life out there, it's likely in a long slow dance with a red dwarf.

## Features

- Loads Polis conversation data from APIs, CSV exports, or saved JSON.
- Two pipeline implementations:
  - **"Polis" pipeline** = Legacy compatibility baseline: PCA, KMeans, and heuristic statement selection, tested against saved Polis API results within the limits below.
  - **"Agora" pipeline** = An alternative intended to address selection edge cases, using [Benjamini-Hochberg][bh-wiki] selection, effective agreement group-aware consensus, and [Simes'][simes-wiki] p-value combination.
- Alternative algorithms, aspiring for sensible defaults:
  - dimensional reduction: [PaCMAP & LocalMAP][pacmap]
    - Planned: [UMAP][umap], [TriMap][trimap], [PHATE][], [ivis][ivis], [LargeVis][largevis]
  - clustering: [HDBSCAN*][hdbscan]
    - Planned: [EVOC][evoc]
- Helpful visualizations via `matplotlib`
  - Planned: [Plotly][plotly]

## Polis vs Agora

Both pipelines share the same core (PCA + KMeans by default), but differ in
**statement selection** and **consensus scoring**. Polis preserves legacy behavior
for comparison; Agora aims to avoid the edge cases below rather than reproduce
every upstream heuristic.

| Feature | Polis | Agora |
|---------|-------|-------|
| **Statement selection** | Heuristics and fallbacks; `pick_max=5` by default | Benjamini-Hochberg selection; no fixed number per group |
| **Evidence** | z-score thresholds and best-agree heuristic | Simes combination of probability and representativeness p-values |
| **Group-aware consensus (agree)** | Legacy default: `prod(pa)` (raw product) | Top-level score: `prod(pa*(1-pd))^(1/n)`, discounting disagreement |
| **Output** | Selected statements only | Ranked candidates with `selected` flags; moderated-out statements excluded |

Here, `pa` and `pd` are smoothed agreement/disagreement probabilities per group,
and `n` is the number of groups. Agora's top-level `group_aware_consensus` is
recomputed from group statistics; its smoothing differs from the default standard-mode
consensus columns in `statements_df`, so these outputs should not be assumed interchangeable.

### Selection edge cases

The [Polis selector tests][polis-selector-tests] characterize inherited quirks.
The [paired pipeline tests][pipeline-selection-tests] run both public pipelines
on identical synthetic votes, seed two groups, and verify matching memberships:

| Case | Polis compatibility behavior | Agora behavior in these tests |
|------|------------------------------|-------------------------------|
| 0 agree / 20 disagree | Can receive `best-agree=True` despite a disagreement label | Selected as disagreement, with explicit agree/disagree counts and no best-agree flag |
| Unanimous agreement shared by both groups | Equal representativeness scores can format best-agree as disagreement | Labeled agreement |
| Pass-only or unvoted within a group | Fallback can return a statement without directional votes | Retained as a candidate, but `selected=False`; excluded from the representative-selection BH hypothesis count |

Both pipelines also retain strong disagreement without an agreement headline.
Red-Dwarf port bugs that dropped such results or misreported `n-agree` have been
fixed in the **Polis** path too; inherited quirks remain for compatibility.
Consumers of Agora's rankings must respect `selected`, not treat every candidate
as a recommendation.

These tests show targeted improvements, not overall statistical superiority or
end-to-end FDR calibration. Simes/BH guarantees depend on their assumptions;
Simes combines evidence, rather than requiring both a majority and between-group
distinctiveness. See also the [ranker and consensus unit tests][agora-stats-tests].

### How Polis compatibility is tested

The [small-conversation parity test][polis-parity-tests] recomputes results from
saved votes and statement metadata, comparing against raw Polis
`/api/v3/math/pca2` responses saved as `math-pca2.json`. The two
[fixtures][polis-fixtures] have 23 and 27 participants. Checks cover PCA,
projections, group memberships, extremity, priorities, representative statements,
consensus, and group-aware consensus, with numerical tolerances where appropriate.

This is a **controlled comparison**: PCA signs are aligned, group count and initial
centers come from the API, and retained participant IDs are supplied explicitly.
It does not establish automatic group-count/history parity or rerun the Clojure
implementation. [Live-loader tests][loader-tests] check fetching and data
compatibility, not full-pipeline parity on arbitrary live conversations.

### Base clusters and eventual display buckets

Polis first compresses participant projections into up to 100 **base clusters**
by default, then discovers opinion groups from their weighted centers. This is a
scaling approximation, not just a way to draw fewer dots.

Red-Dwarf currently discovers opinion groups directly from participant projections.
It does not need this intermediate compression to run analysis, and keeping
participants separate avoids losing that detail before group discovery. The
tradeoffs are potentially more computation and no claim of matching Polis's
larger-conversation results; those parity tests remain skipped.

The eventual plan is optional **display buckets**, created *after* opinion-group
discovery and before export/rendering. They would reduce the number of points
drawn without changing group memberships or analysis statistics. This is not
implemented yet, and would not reproduce Polis's computational base clustering.

Agora roadmap: cleaner input/output APIs, improved default algorithms, reduced DataFrame complexity, and richer outputs.

## Goals

Code that aspires to embody and support democratic values should be...

- **Legible.** It should be explorable and auditable not just to researchers, but to as many curious citizens as possible.
- **Re-usable.** It should be easily used in contexts in which its original creators did not anticipate, nor perhaps even desire.
- **Collectively stewarded.** It should grow and govern itself democratically and in a plural fashion.

## Roadmap

For now, see [this related issue](https://github.com/patcon/red-dwarf/issues/4)

## Sponsors

<p>
  <a href="https://www.agoracitizen.network" rel="noopener sponsored" target="_blank">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/agora-logo-white.png" />
      <img width="128" height="67" src="docs/agora-logo-gradient.png" alt="Agora Citizen Network" title="Where citizens converge to exchange and debate ideas" loading="lazy" />
    </picture>
  </a>
</p>

Red Dwarf is generously sponsored by [ZKorum SAS](https://zkorum.com), creators of the [Agora Citizen Network](https://www.agoracitizen.network).

Are you or your organization eager to see more platforms and community built around democracy-supporting algorithms like these? **Please consider [getting in touch on Discord](#get-involved) and supporting our continued work!** (ping @patcon)

## Usage

```
# OFFICIAL RELEASES

# For core, the classic polis pipeline: PCA + K-means
# (~60MB beyond scikit-learn disk space)
pip install red-dwarf

# For additional algorithms beyond classic polis: PaCMAP, LocalMAP, HDBSCAN, etc.
pip install red-dwarf[alt-algos]

# For additional packages for visualizing plots
pip install red-dwarf[plots]

# For everything (~60MB beyond core packages)
pip install red-dwarf[all]
# pip install red-dwarf[alt-algos,plots]
```

See [`docs/notebooks/agora-demo.ipynb`][agora-notebook] for the recommended quickstart using the Agora pipeline.

For the Polis compatibility baseline, see [`docs/notebooks/polis-implementation-demo.ipynb`][polis-notebook] and the [testing limits above](#how-polis-compatibility-is-tested).

| screenshot of library-generated notebook | screenshot of Polis-generated report |
|---|---|
| [![screen of the sample jupyter notebook](docs/notebook-screenshot.png)][polis-notebook] | ![screenshot of the polis report](https://imgur.com/blkIEtW.png) |

- [`docs/notebooks/agora-demo.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/agora-demo.ipynb) — Agora pipeline quickstart _(recommended)_
- [`docs/notebooks/polis-implementation-demo.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/polis-implementation-demo.ipynb) — Classic Polis pipeline baseline
- [`docs/notebooks/loading-data.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/loading-data.ipynb)
- [`docs/notebooks/heatmap.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/heatmap.ipynb)
- [`docs/notebooks/dump-downloaded-polis-data.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/dump-downloaded-polis-data.ipynb)
- Advanced
   - [`docs/notebooks/map-xids.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/map-xids.ipynb)
   - [`docs/notebooks/alternative-algorithms.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/alternative-algorithms.ipynb)
   - [`docs/notebooks/strip-pass-votes.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/strip-pass-votes.ipynb)
   - [`docs/notebooks/untested/tabpfn-experiment.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/untested/tabpfn-experiment.ipynb)

## Architecture

This is the generalized pipeline of Polis-like processes that we're aiming to accomodate. (See [issue #53](https://github.com/polis-community/red-dwarf/issues/53#issuecomment-2942923628) for details.)

![](https://github.com/user-attachments/assets/8b7b5bfc-7127-4a27-8316-7528209d7c8e)

## Get Involved

### Running it for local development

- Install python (preferrably virtual environment)
- Install uv (python package manager) (e.g. `pip install uv`)
- Install dependencies with `make install-dev`
- Run `make` command alone to see other helpful make subcommands
- Alternatively, run one of the ipynb notebooks; possibly replacing the install command at the top with `%pip install -e ../../` to use the local source instead.

### Contributing

- [Join][pug-discord] the _Polis User Group (PUG)_ **Discord** server.
- Open a **GitHub issue**.
- Submit a **GitHub pull request**.
- Review the [**_Awesome Polis_ directory**][awesome-polis] to learn about related projects, academic papers, and other resources.
   - Use the _"People"_ section to find other individuals and groups working in the field.


## Changelog

See [`CHANGELOG.md`][changelog].

## License

[MPL 2.0: Mozilla Public License 2.0][mplv2] (See [`LICENSE`][license])

<!-- Links -->
   [pypi]: https://pypi.org/project/red-dwarf/
   [stellarpunk]: https://www.youtube.com/watch?v=opnkQVZrhAw

   [bh-wiki]: https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure
   [simes-wiki]: https://en.wikipedia.org/wiki/Simes%27_test
   [polis-selector-tests]: https://github.com/polis-community/red-dwarf/blob/main/tests/utils/test_representative_statements.py
   [pipeline-selection-tests]: https://github.com/polis-community/red-dwarf/blob/main/tests/implementations/test_pipeline_representative_selection.py
   [agora-stats-tests]: https://github.com/polis-community/red-dwarf/blob/main/tests/utils/test_agora_stats.py
   [polis-parity-tests]: https://github.com/polis-community/red-dwarf/blob/main/tests/implementations/test_polis.py
   [polis-fixtures]: https://github.com/polis-community/red-dwarf/blob/main/tests/fixtures.py
   [loader-tests]: https://github.com/polis-community/red-dwarf/blob/main/tests/test_data_loader.py

   [pacmap]: https://github.com/YingfanWang/PaCMAP
   [umap]: https://github.com/lmcinnes/umap
   [trimap]: https://github.com/eamid/trimap
   [phate]: https://github.com/KrishnaswamyLab/PHATE
   [ivis]: https://github.com/beringresearch/ivis
   [largevis]: https://github.com/lferry007/LargeVis

   [hdbscan]: https://github.com/scikit-learn-contrib/hdbscan
   [evoc]: https://github.com/TutteInstitute/evoc

   [plotly]: https://plotly.com/python/

   [agora-notebook]: https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/agora-demo.ipynb
   [polis-notebook]: https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/polis-implementation-demo.ipynb
   [notebooks]: https://github.com/polis-community/red-dwarf/tree/main/docs/notebooks/
   [ZKorum]: https://github.com/zkorum
   [agora]: https://www.agoracitizen.network/
   [ngi-funding]: https://trustchain.ngi.eu/zkorum/
   [MPLv2]: https://choosealicense.com/licenses/mpl-2.0/
   [license]: https://github.com/polis-community/red-dwarf/blob/main/LICENSE
   [pug-discord]: https://discord.com/invite/wFWB8kzQpP
   [awesome-polis]: http://patcon.github.io/awesome-polis/
   [changelog]: CHANGELOG.md
