from dataclasses import dataclass
from typing import Literal


@dataclass
class RankedRepnessStatement:
    statement_id: int
    repful_for: Literal["agree", "disagree", "divisive"]
    signal_strength: Literal["normal", "strong"]
    na: int
    nd: int
    ns: int
    group_size: int
    out_group_size: int
    na_out: int
    nd_out: int
    ns_out: int
    pa: float
    pd: float
    pat: float
    pdt: float
    ra: float
    rd: float
    rat: float
    rdt: float
    divisiveness: float
    agree_effect: float
    disagree_effect: float
    divisive_effect: float
    agree_p_value: float
    disagree_p_value: float
    divisive_p_value: float
    effect_size: float  # winning thresholded ranking score
    p_value: float  # combined p-value (before BH)
    adjusted_p_value: float  # after BH
    selected: bool  # passes BH cutoff
    rank: int  # 1-based, by effect_size descending


@dataclass
class RankedConsensusStatement:
    statement_id: int
    direction: Literal["agree", "disagree"]
    na: int
    nd: int
    ns: int
    pa: float
    pd: float
    pat: float
    pdt: float
    effect_size: float  # pa or pd
    p_value: float
    adjusted_p_value: float
    selected: bool
    rank: int  # 1-based, by effect_size descending


@dataclass
class RankedConsensusResult:
    agree: list[RankedConsensusStatement]
    disagree: list[RankedConsensusStatement]
