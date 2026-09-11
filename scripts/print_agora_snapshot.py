#!/usr/bin/env python3
"""Print Agora pipeline snapshot for cram testing.

Runs the Agora pipeline on fixture data and prints human-readable output
showing representative opinions, group-aware consensus, and divisive
statements with cross-group comparisons.
"""
import sys

from reddwarf.data_loader import Loader
from reddwarf.implementations.agora import run_pipeline
from reddwarf.implementations.base import AnalysisOutcome
from reddwarf.utils.consensus import select_consensus_statements
from reddwarf.utils.statements import process_statements
from reddwarf.utils.stats import select_representative_statements


def truncate(text, max_len=30):
    if len(text) <= max_len:
        return f'"{text}"'
    return f'"{text[:max_len]}..."'


def format_votes(na, nd, ns):
    return f"{na:>2}/{nd:>2}/{ns:>2}"


def main():
    fixture_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/below-100-ptpts"

    loader = Loader(filepaths=[
        f"{fixture_dir}/votes.json",
        f"{fixture_dir}/comments.json",
        f"{fixture_dir}/conversation.json",
    ])

    _, _, mod_out_statement_ids, meta_statement_ids = process_statements(
        statement_data=loader.comments_data
    )

    stmt_text = {}
    for c in loader.comments_data:
        sid = c.get("tid", c.get("statement_id"))
        stmt_text[sid] = c.get("txt", "")

    result = run_pipeline(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        meta_statement_ids=meta_statement_ids,
        random_state=42,
    )
    if result.outcome != AnalysisOutcome.SUCCESS:
        raise SystemExit(f"Agora pipeline failed: {result.reason.value}")
    result = result.result

    group_ids = sorted(result.ranked_repness.keys())
    n_groups = len(group_ids)
    n_clustered = len(result.participants_df[result.participants_df["to_cluster"]])
    n_statements = len(result.statements_df)
    n_mod_out = len(mod_out_statement_ids)
    stats = result.group_comment_stats
    gac = result.group_aware_consensus

    # Header
    print("=== AGORA PIPELINE SNAPSHOT ===")
    print(f"Fixture: {fixture_dir}")
    print(f"Participants: {n_clustered} clustered, {n_groups} groups")
    print(f"Statements: {n_statements - n_mod_out} active ({n_mod_out} moderated out)")

    # Representative opinions
    print()
    print("=== REPRESENTATIVE OPINIONS ===")
    for gid in group_ids:
        print()
        print(f"--- Group {gid} ---")
        print(f" #  Sel  Dir    Stmt  Effect  Text")
        print(f"{'':>33}{' |'.join(f'  G{g} pa/pd (na/nd/ns)' for g in group_ids)}")
        for s in result.ranked_repness[gid]:
            sel = "[*]" if s.selected else "[ ]"
            direction = "agree" if s.repful_for == "agree" else "dis  "
            text = truncate(stmt_text.get(s.statement_id, "?"), 30)
            print(f"{s.rank:>2}  {sel}  {direction}  s{s.statement_id:<3}  {s.effect_size:.4f}  {text}")
            # Cross-group comparison line
            parts = []
            for g in group_ids:
                row = stats.loc[(g, s.statement_id)]
                pa, pd_val = row["pa"], row["pd"]
                na, nd, ns = int(row["na"]), int(row["nd"]), int(row["ns"])
                parts.append(f"  {pa:.2f}/{pd_val:.2f} ({format_votes(na, nd, ns)})")
            print(f"{'':>33}{' |'.join(parts)}")

    # Filter mod_out from GAC for display consistency with other sections
    active_gac_agree = [(sid, score) for sid, score in gac["agree"].items() if sid not in mod_out_statement_ids]
    active_gac_disagree = [(sid, score) for sid, score in gac["disagree"].items() if sid not in mod_out_statement_ids]

    # Group-aware consensus (agree)
    print()
    print("=== GROUP-AWARE CONSENSUS (AGREE) ===")
    sorted_agree = sorted(active_gac_agree, key=lambda x: x[1], reverse=True)
    print(f" #  Stmt  GAC     Text")
    print(f"{'':>24}{' |'.join(f'  G{g} pa (na/nd/ns)' for g in group_ids)}")
    for rank, (sid, score) in enumerate(sorted_agree, 1):
        text = truncate(stmt_text.get(sid, "?"), 30)
        print(f"{rank:>2}  s{sid:<3}  {score:.4f}  {text}")
        parts = []
        for g in group_ids:
            row = stats.loc[(g, sid)]
            pa = row["pa"]
            na, nd, ns = int(row["na"]), int(row["nd"]), int(row["ns"])
            parts.append(f"  {pa:.2f} ({format_votes(na, nd, ns)})")
        print(f"{'':>24}{' |'.join(parts)}")

    # Group-aware consensus (disagree)
    print()
    print("=== GROUP-AWARE CONSENSUS (DISAGREE) ===")
    sorted_disagree = sorted(active_gac_disagree, key=lambda x: x[1], reverse=True)
    print(f" #  Stmt  GAC     Text")
    print(f"{'':>24}{' |'.join(f'  G{g} pd (na/nd/ns)' for g in group_ids)}")
    for rank, (sid, score) in enumerate(sorted_disagree, 1):
        text = truncate(stmt_text.get(sid, "?"), 30)
        print(f"{rank:>2}  s{sid:<3}  {score:.4f}  {text}")
        parts = []
        for g in group_ids:
            row = stats.loc[(g, sid)]
            pd_val = row["pd"]
            na, nd, ns = int(row["na"]), int(row["nd"]), int(row["ns"])
            parts.append(f"  {pd_val:.2f} ({format_votes(na, nd, ns)})")
        print(f"{'':>24}{' |'.join(parts)}")

    # Consensus statements (agree)
    print()
    print("=== CONSENSUS STATEMENTS (AGREE) ===")
    print(f" #  Sel  Stmt  Effect  pa    na/nd/ns  p_adj    Text")
    for s in result.ranked_consensus.agree:
        sel = "[*]" if s.selected else "[ ]"
        text = truncate(stmt_text.get(s.statement_id, "?"), 30)
        print(f"{s.rank:>2}  {sel}  s{s.statement_id:<3}  {s.effect_size:.4f}  {s.pa:.2f}  {format_votes(s.na, s.nd, s.ns)}  {s.adjusted_p_value:.4f}  {text}")

    # Consensus statements (disagree)
    print()
    print("=== CONSENSUS STATEMENTS (DISAGREE) ===")
    print(f" #  Sel  Stmt  Effect  pd    na/nd/ns  p_adj    Text")
    for s in result.ranked_consensus.disagree:
        sel = "[*]" if s.selected else "[ ]"
        text = truncate(stmt_text.get(s.statement_id, "?"), 30)
        print(f"{s.rank:>2}  {sel}  s{s.statement_id:<3}  {s.effect_size:.4f}  {s.pd:.2f}  {format_votes(s.na, s.nd, s.ns)}  {s.adjusted_p_value:.4f}  {text}")

    # Divisive statements
    print()
    print("=== DIVISIVE STATEMENTS ===")
    # Compute divergence: max absolute difference in pa across any pair of groups
    statement_ids = [sid for sid in result.statements_df.index if sid not in mod_out_statement_ids]
    divergences = []
    for sid in statement_ids:
        pa_values = [stats.loc[(g, sid), "pa"] for g in group_ids]
        max_diff = max(pa_values) - min(pa_values)
        divergences.append((sid, max_diff))
    divergences.sort(key=lambda x: x[1], reverse=True)

    print(f" #  Stmt  Div     Text")
    print(f"{'':>24}{' |'.join(f'  G{g} pa/pd (na/nd/ns)' for g in group_ids)}")
    for rank, (sid, div) in enumerate(divergences, 1):
        text = truncate(stmt_text.get(sid, "?"), 30)
        print(f"{rank:>2}  s{sid:<3}  {div:.4f}  {text}")
        parts = []
        for g in group_ids:
            row = stats.loc[(g, sid)]
            pa, pd_val = row["pa"], row["pd"]
            na, nd, ns = int(row["na"]), int(row["nd"]), int(row["ns"])
            parts.append(f"  {pa:.2f}/{pd_val:.2f} ({format_votes(na, nd, ns)})")
        print(f"{'':>24}{' |'.join(parts)}")

    # Selection comparison: Polis vs Agora
    polis_repness = select_representative_statements(
        grouped_stats_df=result.group_comment_stats,
        mod_out_statement_ids=mod_out_statement_ids,
    )
    polis_consensus = select_consensus_statements(
        vote_matrix=result.raw_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    print()
    print("=== SELECTION COMPARISON ===")

    def fmt_stmt(sid, direction):
        d = "a" if direction == "agree" else "d"
        return f"s{sid}({d})"

    print()
    print("--- Repness ---")
    for gid in group_ids:
        polis_stmts = [fmt_stmt(s["tid"], s["repful-for"]) for s in polis_repness.get(gid, [])]
        agora_stmts = [fmt_stmt(s.statement_id, s.repful_for) for s in result.ranked_repness[gid] if s.selected]
        polis_str = " ".join(polis_stmts) if polis_stmts else "(none)"
        agora_str = " ".join(agora_stmts) if agora_stmts else "(none)"
        print(f"Group {gid}:")
        print(f"  Polis (pick_max=5): {polis_str}")
        print(f"  Agora (BH fdr=0.10): {agora_str}")

    print()
    print("--- Consensus (agree) ---")
    polis_agree = [f"s{s['tid']}" for s in polis_consensus["agree"]]
    agora_agree = [f"s{s.statement_id}" for s in result.ranked_consensus.agree if s.selected]
    print(f"  Polis (pick_max=5): {' '.join(polis_agree) if polis_agree else '(none)'}")
    print(f"  Agora (BH fdr=0.10): {' '.join(agora_agree) if agora_agree else '(none)'}")

    print()
    print("--- Consensus (disagree) ---")
    polis_disagree = [f"s{s['tid']}" for s in polis_consensus["disagree"]]
    agora_disagree = [f"s{s.statement_id}" for s in result.ranked_consensus.disagree if s.selected]
    print(f"  Polis (pick_max=5): {' '.join(polis_disagree) if polis_disagree else '(none)'}")
    print(f"  Agora (BH fdr=0.10): {' '.join(agora_disagree) if agora_disagree else '(none)'}")


if __name__ == "__main__":
    main()
