import pandas as pd
from reddwarf.models import ModeratedEnum

def process_statements(
    statement_data: list[dict] = [],
    polis_backward_compat: bool = True,
    is_strict_moderation: bool = False,
) -> tuple[pd.DataFrame, list, list, list]:
    """
    Process raw statement data into a dataframe, and various lists of participant IDs.

    This is mainly used to help zero out vote columns for statements that are excluded via moderation.

    Args:
        statement_data (list[dict]): raw list of statement data dicts
        polis_backward_compat (bool): Whether to reproduce Polis behavior that disregards ambiguous unmoderated statements.
        is_strict_moderation (bool): Whether conversation follows strict moderation (No effect when polis_backward_compat=True)

    Returns:
        statements_df (pd.DataFrame): Dataframe of statements
        mod_in_statement_ids (list): List of statement IDs to moderate in (No current usage)
        mod_out_statement_ids (list): List of statement IDs to moderate out
        meta_statement_ids (list): List of meta statement IDs

    Raises:
        ValueError: Strict moderation is requested but moderation metadata is unavailable.

    """
    mod_in_statement_ids = []
    mod_out_statement_ids = []
    meta_statement_ids = []

    statements_df = (pd.DataFrame
        # TODO: See if both "moderated" and "mod" can end up in here. BUG?
        .from_records(statement_data)
        .set_index('statement_id')
        .sort_index()
    )

    if (
        not polis_backward_compat
        and is_strict_moderation
        and statements_df["moderated"].isna().any()
    ):
        raise ValueError(
            "Strict moderation requires moderation metadata for every statement; "
            "use a data source that includes moderation status."
        )

    if polis_backward_compat:
        mod_in_types =  [ ModeratedEnum.APPROVED ]
        mod_out_types = [ ModeratedEnum.REJECTED ]
    else:
        if is_strict_moderation:
            mod_in_types =  [ ModeratedEnum.APPROVED ]
            mod_out_types = [ ModeratedEnum.REJECTED, ModeratedEnum.UNMODERATED ]
        else:
            mod_in_types =  [ ModeratedEnum.APPROVED, ModeratedEnum.UNMODERATED ]
            mod_out_types = [ ModeratedEnum.REJECTED ]

    for i, row in statements_df.iterrows():
        # TODO: Why does is_meta make a statement mod-in? I'd assume it would exlude from mod-in.
        # Upstream commit messages say that mod-in was added to improve
        # customization of viz, but seem to never actually be used in front-end.
        # Note: mod-in doesn't seem to be actually used in the front-end, so a bug here wouldn't matter.
        # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L825-L842
        if row['is_meta'] or row['moderated'] in mod_in_types:
            mod_in_statement_ids.append(i)

        # May be used in the frontend.
        # In polismath, we only use mod-out to calculate repness, but it's acknowledged that it should react to strict moderation.
        # See: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L669
        if row['is_meta'] or row['moderated'] in mod_out_types:
            mod_out_statement_ids.append(i)

        # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L843-L850
        if row['is_meta']:
            meta_statement_ids.append(i)

    return statements_df, mod_in_statement_ids, mod_out_statement_ids, meta_statement_ids
