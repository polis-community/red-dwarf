import pandas as pd # For sake of clean diffs
from collections import defaultdict
from reddwarf.data_loader import Loader
from reddwarf.models import ModeratedEnum
from reddwarf import utils
from typing import Optional

class PolisClient():
    def __init__(self, is_strict_moderation: Optional[bool] = None) -> None:
        self.data_loader = None
        self.n_components = 2
        # Ref: https://hyp.is/8zUyWM5fEe-uIO-J34vbkg/gwern.net/doc/sociology/2021-small.pdf
        self.min_votes = 7
        self.max_group_count = 5
        self.votes = []
        self.statements_df = None
        self.participants_df = None
        # We sometimes want to keep some participant IDs that would otherwise be removed
        # (e.g., for not meeting vote threshold), to reproduce bugs in Polis codebase algorithms.
        self.keep_participant_ids = []
        # Ref: https://hyp.is/MV0Iws5fEe-k9BdY6UR1VQ/gwern.net/doc/sociology/2021-small.pdf
        self.matrix = None
        # Ref: https://gist.github.com/patcon/fd9079a5fbcd533160f8ae211e975307#file-math-pca2-json-L2
        # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L217-L225
        self.user_vote_counts = defaultdict(int)
        self.meta_tids = []
        self.mod_in = []
        self.mod_out = []
        self.last_vote_timestamp = 0
        self.group_clusters = []
        self.base_clusters = {}
        # TODO: Make accessor methods for these?
        self.statement_count = None
        self.participant_count = None
        self.is_strict_moderation = is_strict_moderation
        # Also know as PCA coords, PCA components, or embeddings.
        self.eigenvectors = []
        # Also know as PCA explained variance.
        self.eigenvalues = []
        self.optimal_k = None
        self.optimal_cluster_labels = None
        self.optimal_silhouette = None

    def get_participant_row_mask(self):
        raise NotImplementedError

    def get_is_strict_moderation(self):
        if self.is_strict_moderation == None:
            raise ValueError('is_strict_moderation cannot be auto-detected, and must be set manually')

        return self.is_strict_moderation

    def get_active_statement_ids(self):
        active_statements = self.statements_df[self.statements_df["is_shown"]]
        active_statement_ids = active_statements.index
        return active_statement_ids

    def get_unvoted_statement_ids(self):
        return self.matrix.pipe(utils.get_unvoted_statement_ids)

    def apply_masks(self, participant_rows=True, statement_cols=True):
        raise NotImplementedError

    def add_votes_batch(self, votes_df):
        """Add multiple votes from a DataFrame"""
        # TODO: Use tuples instead of named columns later, for perf.
        for _, row in votes_df.iterrows():
            self.add_vote(row)

    def add_vote(self, vote_row):
        """Add a single vote to the system"""
        # If this is a new vote (not an update)
        self.user_vote_counts[vote_row['participant_id']] += 1

        if self.last_vote_timestamp < int(vote_row['modified']):
            self.last_vote_timestamp = int(vote_row['modified'])

        self.votes.append({
            'participant_id': vote_row['participant_id'],
            'statement_id': vote_row['statement_id'],
            'vote': vote_row['vote'],
            'modified': vote_row['modified'],
        })
        # Matrix is now stale
        self.matrix = None
        self.statement_count = None
        self.participant_count = None

    def get_user_vote_counts(self):
        return self.user_vote_counts

    def get_meta_tids(self):
        return self.meta_tids

    def get_mod_in(self):
        return self.mod_in

    def get_mod_out(self):
        return self.mod_out

    def get_last_vote_timestamp(self):
        return self.last_vote_timestamp

    def get_group_clusters(self):
        return self.group_clusters

    def get_matrix(self, is_filtered=False, cutoff=None):
        # Only generate matrix when needed.
        if self.matrix is None:
            vote_matrix = utils.generate_raw_matrix(self.votes, cutoff=cutoff)
            self.statement_count = vote_matrix.notna().any().sum()
            self.participant_count = len(vote_matrix)

            if is_filtered:
                vote_matrix = utils.filter_matrix(
                    vote_matrix=vote_matrix,
                    min_user_vote_threshold=self.min_votes,
                    active_statement_ids=self.get_active_statement_ids(),
                    keep_participant_ids=self.keep_participant_ids,
                )

            self.matrix = vote_matrix

        return self.matrix

    def build_participants_dataframe(self, vote_matrix):
        # TODO: Drop statements from participants-votes when moderated out and no votes.
        def get_comment_count(pid):
            return (self
                .statements_df["participant_id"]
                .value_counts()
                .get(pid, 0)
            )

        def get_xid(pid):
            # TODO: Implemment this
            return None

        def get_group_id(pid):
            # TODO: Implement this.
            return None

        participants_df = (pd.DataFrame()
            .assign(participant_id=vote_matrix.index)
            .set_index('participant_id')
            .assign(xid=[get_xid(pid) for pid in vote_matrix.index])
            .assign(group_id=[get_group_id(pid) for pid in vote_matrix.index])
            .assign(n_comments=[get_comment_count(pid) for pid in vote_matrix.index])
            .assign(n_votes=vote_matrix.count(axis="columns"))
            .assign(n_agree=vote_matrix.apply(lambda row: row.eq(1).sum(), axis="columns"))
            .assign(n_disagree=vote_matrix.apply(lambda row: row.eq(-1).sum(), axis="columns"))
        )
        return participants_df

    def run_pca(self):
        projected_data, components, explained_variance, means, *_ = utils.run_pca(
            vote_matrix=self.matrix,
            n_components=self.n_components,
        )

        self.eigenvectors = components
        self.eigenvalues = explained_variance
        self.projected_data = projected_data
        self.means = means

    def scale_projected_data(self):
        scaled_data = utils.scale_projected_data(
            projected_data = self.projected_data,
            vote_matrix = self.matrix,
        )
        self.projected_data = scaled_data

    # Not working yet.
    def build_base_clusters(self):
        # TODO: Is this participant_count the correct one?
        n_clusters=min(self.base_cluster_count, self.participant_count)
        # Ref: Polis math values, base-iters
        print(self.eigenvectors)
        cluster_labels, cluster_centers = utils.run_kmeans(self.eigenvectors, n_clusters=n_clusters)
        # ptpt_to_cluster_mapping = dict(zip(self.matrix.index, range(len(kmeans.labels_))))
        # cluster_to_ptpt_mapping = {v: k for k, v in ptpt_to_cluster_mapping.items()}
        participant_df = pd.DataFrame.from_dict(
            {
                'participant_id': self.matrix.index,
                'cluster_id': cluster_labels.tolist(),
            },
        )
        cluster_df = participant_df.groupby('cluster_id')['participant_id'].apply(list).reset_index(name="participant_ids")
        self.base_clusters['id'] = list(range(100))
        self.base_clusters['members'] = cluster_df["participant_ids"].values.tolist()
        self.base_clusters['x'] = [xy[0] for xy in cluster_centers.tolist()]
        self.base_clusters['y'] = [xy[1] for xy in cluster_centers.tolist()]

    # Not working yet.
    def load_base_clusters_from_math(self):
        self.base_clusters = self.data_loader.math_data["base-clusters"]

    def find_optimal_k(self):
        best_k, silhouette_score, cluster_labels = utils.find_optimal_k(
            projected_data=self.projected_data,
            max_group_count=self.max_group_count,
            debug=True,
        )
        self.optimal_k = best_k
        self.optimal_silhouette = silhouette_score
        self.optimal_cluster_labels = cluster_labels

    def load_data(self, filepaths=[], conversation_id=None, report_id=None, directory_url=None, data_source="api", is_ip_rotation_enabled=False, is_cache_enabled=True):
        if conversation_id or report_id or filepaths or directory_url:
            self.data_loader = Loader(conversation_id=conversation_id, report_id=report_id, filepaths=filepaths, directory_url=directory_url, data_source=data_source, is_ip_rotation_enabled=is_ip_rotation_enabled, is_cache_enabled=is_cache_enabled)

            # Infer moderation type from API conversation_data when available.
            if self.data_loader.conversation_data:
                self.is_strict_moderation = self.data_loader.conversation_data["strict_moderation"]

            if self.data_loader.comments_data:
                self.load_comments_data(data=self.data_loader.comments_data)
            if self.data_loader.votes_data:
                self.load_votes_data(data=self.data_loader.votes_data)

    def load_votes_data(self, data=None):
        votes_df = pd.DataFrame.from_records(data).astype({'modified': 'int64'})
        self.add_votes_batch(votes_df)

    def load_comments_data(self, data=None):
        statements_df = (pd.DataFrame
            .from_records(data)
            .set_index('statement_id')
            .sort_index()
        )

        def is_shown(statement):
            if self.get_is_strict_moderation():
                active_mod_states = [ModeratedEnum.APPROVED]
            else:
                active_mod_states = [ModeratedEnum.APPROVED, ModeratedEnum.UNMODERATED]
            is_shown = statement["moderated"] in active_mod_states
            return is_shown

        statements_df["is_shown"] = statements_df.apply(is_shown, axis="columns")
        for i, row in statements_df.iterrows():
            # TODO: Why does is_meta make this mod-in.
            # Maybe I don't understand what mod-in does...
            # Note: mod-in/mod-out doesn't seem to be actually used in the front-end, so a bug here wouldn't matter.
            # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L825-L842
            if row['is_meta'] or row['moderated'] == 1:
                self.mod_in.append(i)

            if row['is_meta'] or row['moderated'] == -1:
                self.mod_out.append(i)

            # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L843-L850
            if row['is_meta']:
                self.meta_tids.append(i)

        self.statements_df = statements_df
