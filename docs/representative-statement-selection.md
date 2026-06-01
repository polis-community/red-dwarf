# Representative Statement Selection

After clustering, each group typically has some statements whose voting patterns are more characteristic of that group than of the others. These are the group’s representative statements.

A representative statement tells us something distinctive about a group:

- the group agrees with it much more than other groups
- the group disagrees with it much more than other groups
- or the group is internally divided on it in a way that is itself informative

Now there are two paths for the selection of representative statements in Red Dwarf after clustering: 

- a Polis-compatible path, which returns a shortlist
- an Agora path [NOTE: name to be changed], which ranks all statements

## Polis-Compatible Path

The Polis-compatible representative-selection pipeline returns a shortlist of representative statements per group (`pick_max`, default `5`).

- It uses a two-direction model of representativeness: each shortlisted statement is representative because the group is unusually likely to agree with it or disagree with it, so `repful_for` is only `agree` or `disagree`.
- Statements are kept only if they are statistically significant for at least one direction, using both within-group probability and cross-group representativeness tests.
- Surviving statements are ranked by a combined representativeness metric:

  ```text
  repness metric = repness * repness-test * p-success * p-test
  ```

- The selector also tries to keep at least one strong agreement-oriented statement near the top, using a dedicated "best agree" heuristic (`beats_best_of_agrees(...)`).
- If no statement survives significance filtering, it falls back to the single strongest overall representative statement for that group, using the strongest raw representativeness-test signal (`beats_best_by_repness_test(...)`).

The Polis-compatible path therefore behaves like a shortlist generator. It assigns `repful_for` to the rows it returns, but it does not produce a full ranking of all statements in the group.

## Agora Path

The Agora path ranks all statements per group instead of returning a shortlist. Representative selection is different here in several ways.

- It produces a full ranking of statements per group.
- It always assigns a representative label to ranked rows, not just to the subset that survives shortlist selection.
- It supports three representative directions:
  - `agree`
  - `disagree`
  - `divisive`
- It adds a statistical `selected` flag on top of the ranking. Each row gets a p-value tied to its winning label, and Benjamini-Hochberg / FDR is applied within the group.
- It uses whole-group prevalence in its ranking scores, so that statements supported by a broader share of the group can rank differently from statements that look strong only among a very small responding subset.
- It checks whether a label is locally plausible from the focal group’s own vote pattern before letting that label win the row.


## The Main Agora Output Fields

For each ranked representative row, the key fields are:

- `repful_for`
  Winning label: `agree`, `disagree`, or `divisive`.

- `effect_size`
  Ranking score of the winning label. It is the comparative representativeness score from Step 2 below.

- `p_value`
  P-value associated with the winning label.

- `selected`
  Whether the row passes BH/FDR selection within the group.

- `signal_strength`
  A stricter practical layer: `normal` or `strong`.

## How the agora path works

### Step 0: Start from grouped statement statistics

After clustering, we already have grouped per-statement stats such as:

- `na`, `nd`, `ns`
- `na_out`, `nd_out`, `ns_out`
- `group_size`, `out_group_size`
- `pa`, `pd`, `pat`, `pdt`
- `ra`, `rd`, `rat`, `rdt`

### Step 1: Check what labels are locally plausible

From the focal group’s own seen-vote pattern, what labels make sense? Compute three local scores:

```text
agree_local    = na / ns
disagree_local = nd / ns
divisive_local = 2 * min(na, nd) / ns
```

A label is a candidate if:

- `agree` if `agree_local >= 0.5`
- `disagree` if `disagree_local >= 0.5`
- `divisive` if `divisive_local >= 0.5` and `min(na, nd) >= 2`


### Step  2: Compute comparative scores on a common full-group scale

Among the plausible labels, which one is most distinctive for this group compared with the out-groups?

This uses whole-group prevalence for all three directions:

```text
agree_prevalence_in    = na / group_size
disagree_prevalence_in = nd / group_size
divisive_prevalence_in = 2 * min(na, nd) / group_size
```

The out-group uses the same formulas with `na_out`, `nd_out`, and `out_group_size`.

The score idea is:

1. how present is this pattern in the focal group?
2. how rare is the same pattern outside the group?
3. reward patterns that are both present internally and distinctive externally

In general form:

```text
effect = prevalence_in * (prevalence_in / max(prevalence_out, floor))
```

Where the floor is:

- `1 / out_group_size` for `agree` and `disagree`
- `2 / out_group_size` for `divisive`

For the directional labels, that is equivalent to:

```text
effect = prevalence_in² / max(prevalence_out, 1 / out_group_size)
```

This common scale lets `divisive` fairly compete with `agree` and `disagree`.

### Step 3: Compute p-values for the labels

For `agree` and `disagree`, this pipeline uses a Simes combination of two existing one-sided tests:

```text
p_agree    = simes(p_from_z(pat), p_from_z(rat))
p_disagree = simes(p_from_z(pdt), p_from_z(rdt))
```

For `divisive`,  a dedicated one-sided permutation test is used, which answers: is this group more internally split on this statement than the out-groups?

A `divisive_p_value` is computed for each row. However, statements with `min(na, nd) < 2` cannot win the `divisive` label, so that thin splits like 1-1 (which is perfectly 50-50) do not rise as divisive winners too early. 

### Step 4: Choose the winning label

For each statement in each group:

1. If one or more labels are eligible, only those labels compete and the highest comparative score wins.
2. If no labels are eligible, Agora falls back to the strongest local direction.
3. Ties break by lower p-value.
4. Final ties break by deterministic label order (`agree > disagree > divisive`).

The pipeline is therefore both:
- comparative across groups
- constrained by the focal group’s own vote pattern

### Step 5: Apply BH/FDR and signal strength

Once the winner is chosen:

- `effect_size` becomes the winner’s score. So once the statement has a winning label (agree, disagree, or divisive), effect_size is that label’s effect score carried forward into the final ranked row.
- `p_value` becomes the winner’s p-value
- BH/FDR is applied over chosen-label p-values within the group
- `selected` marks the rows that pass the inferential cutoff
- `signal_strength` is a stricter practical layer on top of selected, effect_size, p_value, and participation. So signals can either be "strong" or "normal". 

## Note on `pass` and missing votes

- `pass` is not a fourth representative direction, but it dilutes directional and divisive strength indirectly
- missing (`NaN`) still means not seen / no vote recorded
- participation matters through the full-group scoring model 
