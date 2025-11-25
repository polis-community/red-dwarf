import json
import os
from typing_extensions import deprecated
import pandas as pd
from fake_useragent import UserAgent
from datetime import datetime, timezone, timedelta
from requests_ratelimiter import SQLiteBucket, LimiterSession
import csv
from io import StringIO
from reddwarf.models import Vote, Statement
from reddwarf.helpers import CachedLimiterSession, CloudflareBypassHTTPAdapter
from reddwarf.utils.matrix import KeepType, deduplicate_votes

ua = UserAgent()


class Loader:
    """
    A comprehensive data loader for Polis conversation data.

    The Loader class provides a unified interface for loading Polis conversation data
    from multiple sources including API endpoints, CSV exports, and local JSON files.
    It handles data validation, caching, rate limiting, and export functionality.

    The class automatically determines the appropriate loading strategy based on the
    provided parameters and can export data in both JSON and CSV formats compatible
    with the Polis platform.

    Args:
        polis_instance_url (str, optional): Base URL of the Polis instance. Defaults to "https://pol.is".
        filepaths (list[str], optional): List of local file paths to load data from. Defaults to [].
        polis_id (str, optional): Generic Polis identifier (report ID starting with 'r' or conversation ID).
        conversation_id (str, optional): Specific conversation ID for API requests.
        report_id (str, optional): Specific report ID for API requests or CSV exports.
        is_cache_enabled (bool, optional): Enable HTTP request caching. Defaults to True.
        output_dir (str, optional): Directory path for exporting loaded data. If provided, data is automatically exported.
        data_source (str, optional): Data source type ("api" or "csv_export"). Defaults to "api".
        directory_url (str, optional): Direct URL to CSV export directory. If provided, forces csv_export mode.

    Attributes:
        votes_data (list[dict]): Loaded vote data with participant_id, statement_id, vote, and modified fields.
        comments_data (list[dict]): Loaded statement/comment data with text, metadata, and statistics.
        math_data (dict): Mathematical analysis data including PCA projections and group clusters.
        conversation_data (dict): Conversation metadata including topic, description, and settings.
        report_data (dict): Report metadata when loaded via report_id.
        skipped_dup_votes (list[dict]): Duplicate votes that were filtered out during processing.

    Examples:
        Load from API using conversation ID:
        >>> loader = Loader(conversation_id="12345")

        Load from CSV export using report ID:
        >>> loader = Loader(report_id="r67890", data_source="csv_export")

        Load from local files:
        >>> loader = Loader(filepaths=["votes.json", "comments.json", "math-pca2.json"])

        Load and export to directory:
        >>> loader = Loader(conversation_id="12345", output_dir="./exported_data")
    """

    def __init__(
        self,
        polis_instance_url=None,
        filepaths=[],
        polis_id=None,
        conversation_id=None,
        report_id=None,
        is_cache_enabled=True,
        output_dir=None,
        data_source="api",
        directory_url=None,
    ):
        self.polis_instance_url = polis_instance_url or "https://pol.is"
        self.polis_id = report_id or conversation_id or polis_id
        self.conversation_id = conversation_id
        self.report_id = report_id
        self.is_cache_enabled = is_cache_enabled
        self.output_dir = output_dir
        self.data_source = data_source
        self.filepaths = filepaths
        self.directory_url = directory_url

        self.votes_data = []
        self.comments_data = []
        self.math_data = {}
        self.conversation_data = {}
        self.report_data = {}
        self.skipped_dup_votes = []

        if self.filepaths:
            self.load_file_data()
        elif (
            self.conversation_id
            or self.report_id
            or self.polis_id
            or self.directory_url
        ):
            self.populate_polis_ids()
            self.init_http_client()
            if self.directory_url:
                self.data_source = "csv_export"

            if self.data_source == "csv_export":
                self.load_remote_export_data()
            elif self.data_source == "api":
                self.load_api_data()
            else:
                raise ValueError("Unknown data_source: {}".format(self.data_source))

        if self.output_dir:
            self.dump_data(self.output_dir)

    def populate_polis_ids(self):
        """
        Normalize and populate Polis ID fields from the provided identifiers.

        This method handles the logic for determining conversation_id and report_id
        from the generic polis_id parameter. (Report IDs start with 'r', while
        conversation IDs start with a number.)
        """
        if self.polis_id:
            # If polis_id set, set report or conversation ID.
            if self.polis_id[0] == "r":
                self.report_id = self.polis_id
            elif self.polis_id[0].isdigit():
                self.conversation_id = self.polis_id
        else:
            # If not set, write it from what's provided.
            self.polis_id = self.report_id or self.conversation_id

    # Deprecated.
    def dump_data(self, output_dir):
        """
        Export loaded data to JSON files in the specified directory.

        Args:
            output_dir (str): Directory path where JSON files will be written.

        Note:
            This method is deprecated. Use export_data() instead.
        """
        self.export_data(output_dir, format="json")

    def export_data(self, output_dir, format="csv"):
        """
        Export loaded data to files in the specified format.

        Args:
            output_dir (str): Directory path where files will be written.
            format (str): Export format, either "json" or "csv". Defaults to "csv".

        The CSV format exports multiple files compatible with Polis platform:
        - votes.csv: Individual vote records
        - comments.csv: Statement/comment data with metadata
        - comment-groups.csv: Group-specific voting statistics per statement
        - participant-votes.csv: Participant voting patterns and group assignments
        - summary.csv: Conversation summary statistics
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if format == "json":
            self._export_data_json(output_dir)
        elif format == "csv":
            self._export_data_csv(output_dir)

    def _export_data_json(self, output_dir):
        if self.votes_data:
            with open(output_dir + "/votes.json", "w") as f:
                f.write(json.dumps(self.votes_data, indent=4))

        if self.comments_data:
            with open(output_dir + "/comments.json", "w") as f:
                f.write(json.dumps(self.comments_data, indent=4))

        if self.math_data:
            with open(output_dir + "/math-pca2.json", "w") as f:
                f.write(json.dumps(self.math_data, indent=4))

        if self.conversation_data:
            with open(output_dir + "/conversation.json", "w") as f:
                f.write(json.dumps(self.conversation_data, indent=4))

    def _export_data_csv(self, output_dir):
        self._write_polis_votes(output_dir)
        self._write_polis_comments(output_dir)
        self._write_polis_comment_groups(output_dir)
        self._write_polis_participant_votes(output_dir)
        self._write_polis_summary(output_dir)

    def _write_polis_votes(self, output_dir):
        """
        POLIS format:
            timestamp,datetime,comment-id,voter-id,vote
        """
        if not self.votes_data:
            return

        sorted_votes_data = sorted(
            self.votes_data, key=lambda x: (x["statement_id"], x["participant_id"])
        )
        with open(output_dir + "/votes.csv", "w") as f:
            writer = csv.writer(f)
            headers = ["timestamp", "datetime", "comment-id", "voter-id", "vote"]
            writer.writerow(headers)
            for entry in sorted_votes_data:
                ts, dt_str = self._format_polis_times(entry["modified"])
                row = [
                    ts,
                    dt_str,
                    entry["statement_id"],
                    entry["participant_id"],
                    entry["vote"],
                ]
                writer.writerow(row)

    def _write_polis_comments(self, output_dir):
        """
        POLIS format:
            timestamp,datetime,comment-id,author-id,agrees,disagrees,moderated,comment-body
        """
        if not self.comments_data:
            return

        with open(output_dir + "/comments.csv", "w") as f:
            headers = [
                "timestamp",
                "datetime",
                "comment-id",
                "author-id",
                "agrees",
                "disagrees",
                "moderated",
                "comment-body",
            ]
            f.write(",".join(headers) + "\n")
            # Sort comments_data by 'created' timestamp before writing
            sorted_comments = sorted(
                self.comments_data,
                key=lambda x: (x["statement_id"], x["participant_id"]),
            )
            for entry in sorted_comments:
                ts, dt_str = self._format_polis_times(entry["created"])
                single_quote = '"'
                double_quote = '""'
                row = [
                    ts,
                    dt_str,
                    entry["statement_id"],
                    entry["participant_id"],
                    entry["agree_count"],
                    entry["disagree_count"],
                    entry["moderated"],
                    f'"{str(entry["txt"]).replace(single_quote, double_quote)}"',
                ]
                f.write(",".join([str(item) for item in row]) + "\n")

    def _format_polis_times(self, time):
        """Convert timestamp or ISO string to Polis datetime format."""
        from dateutil import parser

        try:
            if isinstance(time, (int, float)):
                # Handle timestamps
                timestamp = int(str(time)[:10]) if time > 10**10 else time
                date_obj = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                # Handle string inputs in ISO format
                date_obj = parser.parse(time)
                if date_obj.tzinfo is None:
                    date_obj = date_obj.replace(tzinfo=timezone.utc)

            date_obj = date_obj.astimezone(timezone.utc)
            dt_str = date_obj.strftime(
                "%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)"
            )

            return int(date_obj.timestamp()), dt_str
        except (ValueError, OSError) as error:
            raise ValueError(f"Timestamp is not in a recognizable format: {error}")

    def _write_polis_comment_groups(self, output_dir):
        """
        POLIS format:
            comment-id,comment,total-votes,total-agrees,total-disagrees,total-passes,group-a-votes,group-a-agrees,group-a-disagrees,group-a-passes,group-[next alphabetic identifier (b)]-votes,[repeat 'votes/agrees/disagrees/passes' with alphabetic identifier...]

        Each row represents a comment with total votes & votes by group
        """
        if not self.comments_data or not self.math_data:
            return

        group_votes = self.math_data.get("group-votes", {})
        group_clusters = self.math_data.get("group-clusters", [])
        group_ids = [group["id"] for group in group_clusters]
        # Map group indices to letters: 0 -> 'a', 1 -> 'b', etc.
        group_letters = [chr(ord("a") + i) for i in range(len(group_ids))]

        with open(output_dir + "/comment-groups.csv", "w") as f:
            # Build header dynamically based on available groups
            header = [
                "comment-id",
                "comment",
                "total-votes",
                "total-agrees",
                "total-disagrees",
                "total-passes",
            ]
            for i, group in enumerate(group_clusters):
                if i < len(group_letters):
                    group_letter = group_letters[i]
                    header.extend(
                        [
                            f"group-{group_letter}-votes",
                            f"group-{group_letter}-agrees",
                            f"group-{group_letter}-disagrees",
                            f"group-{group_letter}-passes",
                        ]
                    )
            f.write(",".join(header))
            f.write("\n")
            rows = []
            sorted_comments_data = sorted(
                self.comments_data, key=lambda x: x["statement_id"]
            )
            for comment in sorted_comments_data:
                comment_id = str(comment["statement_id"])
                row = [
                    comment_id,
                    comment["txt"]
                    if comment["txt"][0] == '"'
                    else '"' + comment["txt"] + '"',
                    comment["count"],
                    comment["agree_count"],
                    comment["disagree_count"],
                    comment["pass_count"],
                ]

                # Add group-specific data
                for i, group in enumerate(group_clusters):
                    if i < len(group_letters):
                        group_id = str(group["id"])
                        if (
                            group_id in group_votes
                            and comment_id in group_votes[group_id]["votes"]
                        ):
                            vote_data = group_votes[group_id]["votes"][comment_id]
                            total_votes = (
                                vote_data["A"] + vote_data["D"] + vote_data["S"]
                            )
                            row.extend(
                                [
                                    total_votes,
                                    vote_data["A"],  # agrees
                                    vote_data["D"],  # disagrees
                                    vote_data["S"],  # passes (skips)
                                ]
                            )
                        else:
                            # No votes from this group for this comment
                            row.extend([0, 0, 0, 0])
                rows.append(row)
                f.write(",".join([str(item) for item in row]) + "\n")

    def _write_polis_participant_votes(self, output_dir):
        """
        POLIS format:
            participant,group-id,n-comments,n-votes,n-agree,n-disagree,0,1,2,3,...

        Each row represents a participant with:
        - participant: participant ID
        - group-id: which group they belong to (if any)
        - n-comments: number of comments they made
        - n-votes: total number of votes they cast
        - n-agree: number of agree votes
        - n-disagree: number of disagree votes
        - 0,1,2,3...: their vote on each comment (1=agree, -1=disagree, 0=pass, empty=no vote)
        """
        if not self.votes_data:
            return

        # Get all unique participant IDs and statement IDs
        participant_ids = set()
        statement_ids = set()
        for vote in self.votes_data:
            participant_ids.add(vote["participant_id"])
            statement_ids.add(vote["statement_id"])

        # Sort to ensure consistent order
        sorted_participant_ids = sorted(participant_ids)
        sorted_statement_ids = sorted(statement_ids)

        # Build participant vote matrix
        participant_votes = {}
        for vote in self.votes_data:
            pid = vote["participant_id"]
            sid = vote["statement_id"]
            if pid not in participant_votes:
                participant_votes[pid] = {}
            participant_votes[pid][sid] = vote["vote"]

        # Get participant group assignments from math data
        participant_groups = {}
        if self.math_data and "group-clusters" in self.math_data:
            for group in self.math_data["group-clusters"]:
                group_id = group["id"]
                for member in group["members"]:
                    participant_groups[member] = group_id

        # Count comments per participant
        participant_comment_counts = {}
        if self.comments_data:
            for comment in self.comments_data:
                pid = comment["participant_id"]
                participant_comment_counts[pid] = (
                    participant_comment_counts.get(pid, 0) + 1
                )

        with open(output_dir + "/participant-votes.csv", "w") as f:
            # Build header
            header = [
                "participant",
                "group-id",
                "n-comments",
                "n-votes",
                "n-agree",
                "n-disagree",
            ]
            header.extend([str(sid) for sid in sorted_statement_ids])
            f.write(",".join(header) + "\n")

            # Write participant data
            for pid in sorted_participant_ids:
                participant_vote_data = participant_votes.get(pid, {})

                # Count votes
                n_votes = len(participant_vote_data)
                n_agree = sum(1 for v in participant_vote_data.values() if v == 1)
                n_disagree = sum(1 for v in participant_vote_data.values() if v == -1)

                # Get group assignment
                group_id = participant_groups.get(pid, "")

                # Get comment count
                n_comments = participant_comment_counts.get(pid, 0)

                row = [pid, group_id, n_comments, n_votes, n_agree, n_disagree]

                # Add vote for each statement
                for sid in sorted_statement_ids:
                    vote = participant_vote_data.get(sid, "")
                    row.append(vote)

                f.write(",".join([str(item) for item in row]) + "\n")

    def _write_polis_summary(self, output_dir):
        """
        POLIS format:
            topic,[string]
            url,http://pol.is/[report_id]
            voters,[num]
            voters-in-conv,[num]
            commenters,[num]
            comments,[num]
            groups,[num]
            conversation-description,[string]
        """
        if not self.conversation_data:
            return

        # Calculate summary statistics
        total_voters = (
            len(set(vote["participant_id"] for vote in self.votes_data))
            if self.votes_data
            else 0
        )
        total_commenters = (
            len(set(comment["participant_id"] for comment in self.comments_data))
            if self.comments_data
            else 0
        )
        total_comments = len(self.comments_data) if self.comments_data else 0
        total_groups = (
            len(self.math_data.get("group-clusters", [])) if self.math_data else 0
        )

        # Get conversation details
        topic = self.conversation_data.get("topic", "")
        description = self.conversation_data.get("description", "")
        if description:
            description = (
                description.replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )

        # Build URL
        url = (
            f"{self.polis_instance_url}/{self.conversation_id}"
            if self.conversation_id
            else self.polis_id
            if self.polis_id
            else self.report_id
        )

        with open(output_dir + "/summary.csv", "w") as f:
            f.write(f'topic,"{topic}"\n')
            f.write(f"url,{url}\n")
            f.write(f"voters,{total_voters}\n")
            f.write(f"voters-in-conv,{total_voters}\n")
            f.write(f"commenters,{total_commenters}\n")
            f.write(f"comments,{total_comments}\n")
            f.write(f"groups,{total_groups}\n")
            f.write(f'conversation-description,"{description}"\n')

    def init_http_client(self):
        """
        Initialize HTTP session with rate limiting, caching, and Cloudflare bypass.

        Sets up a requests session with:
        - Rate limiting (5 requests per second)
        - Optional SQLite-based response caching (1 hour expiration)
        - Cloudflare bypass adapter for the Polis instance
        - Random user agent headers
        """
        # Throttle requests, but disable when response is already cached.
        if self.is_cache_enabled:
            # Source: https://github.com/JWCook/requests-ratelimiter/tree/main?tab=readme-ov-file#custom-session-example-requests-cache
            self.session = CachedLimiterSession(
                per_second=5,
                expire_after=timedelta(hours=1),
                cache_name="test_cache.sqlite",
                bucket_class=SQLiteBucket,
                bucket_kwargs={
                    "path": "test_cache.sqlite",
                    "isolation_level": "EXCLUSIVE",
                    "check_same_thread": False,
                },
            )
        else:
            self.session = LimiterSession(per_second=5)
        adapter = CloudflareBypassHTTPAdapter()
        self.session.mount(self.polis_instance_url, adapter)
        self.session.headers = {
            "User-Agent": ua.random,
        }

    def get_polis_export_directory_url(self, report_id):
        """
        Generate the CSV export directory URL for a given report ID.

        Args:
            report_id (str): The report ID (typically starts with 'r').

        Returns:
            str: Full URL to the CSV export directory endpoint.
        """
        return f"{self.polis_instance_url}/api/v3/reportExport/{report_id}/"

    def _is_statement_meta_field_missing(self):
        if self.comments_data:
            return self.comments_data[0]["is_meta"] is None
        else:
            # No statements loaded, so can't say.
            return False

    def load_remote_export_data(self):
        """
        Load data from remote CSV export endpoints.

        Downloads and processes CSV files from Polis export directory, including:
        - comments.csv: Statement data
        - votes.csv: Vote records

        Handles missing is_meta field by falling back to API data when necessary.
        Automatically filters duplicate votes, keeping the most recent.

        Raises:
            ValueError: If CSV export URL cannot be determined or API fallback fails.
        """
        if self.directory_url:
            directory_url = self.directory_url
        elif self.report_id:
            directory_url = self.get_polis_export_directory_url(self.report_id)
        else:
            raise ValueError(
                "Cannot determine CSV export URL without report_id or directory_url"
            )

        self.load_remote_export_data_comments(directory_url)
        self.load_remote_export_data_votes(directory_url)

        # Supplement is_meta statement field via API if missing.
        # See: https://github.com/polis-community/red-dwarf/issues/55
        if self._is_statement_meta_field_missing():
            import warnings

            warnings.warn(
                "CSV import is missing is_meta field. Attempting to load comments data from API instead..."
            )
            try:
                if self.report_id and not self.conversation_id:
                    self.load_api_data_report()
                    self.conversation_id = self.report_data["conversation_id"]
                self.load_api_data_comments()
            except Exception:
                raise ValueError(
                    " ".join(
                        [
                            "Due to an upstream bug, we must patch CSV exports using the API,",
                            "so conversation_id or report_id is required.",
                            "See: https://github.com/polis-community/red-dwarf/issues/56",
                        ]
                    )
                )

        # When multiple votes (same tid and pid), keep only most recent (vs first).
        self.filter_duplicate_votes(keep="last")
        # self.load_remote_export_data_summary()
        # self.load_remote_export_data_participant_votes()
        # self.load_remote_export_data_comment_groups()

    def load_remote_export_data_comments(self, directory_url):
        """
        Load statement/comment data from remote CSV export.

        Args:
            directory_url (str): Base URL of the CSV export directory.
        """
        r = self.session.get(directory_url + "comments.csv")
        comments_csv = r.text
        reader = csv.DictReader(StringIO(comments_csv))
        self.comments_data = [
            Statement(**c).model_dump(mode="json") for c in list(reader)
        ]

    def load_remote_export_data_votes(self, directory_url):
        """
        Load vote data from remote CSV export.

        Args:
            directory_url (str): Base URL of the CSV export directory.
        """
        r = self.session.get(directory_url + "votes.csv")
        votes_csv = r.text
        reader = csv.DictReader(StringIO(votes_csv))
        self.votes_data = [
            Vote(**vote).model_dump(mode="json") for vote in list(reader)
        ]

    def filter_duplicate_votes(self, keep: KeepType = "last"):
        votes_df = pd.DataFrame(self.votes_data)
        deduped, skipped = deduplicate_votes(votes=votes_df, keep=keep)

        self.skipped_dup_votes = skipped.to_dict(orient="records")
        self.votes_data = deduped.to_dict(orient="records")

    def load_remote_export_data_summary(self):
        # r = self.session.get(self.polis_instance_url + "/api/v3/reportExport/{}/summary.csv".format(self.report_id))
        # summary_csv = r.text
        # print(summary_csv)
        raise NotImplementedError

    def load_remote_export_data_participant_votes(self):
        # r = self.session.get(self.polis_instance_url + "/api/v3/reportExport/{}/participant-votes.csv".format(self.report_id))
        # participant_votes_csv = r.text
        # print(participant_votes_csv)
        raise NotImplementedError

    def load_remote_export_data_comment_groups(self):
        # r = self.session.get(self.polis_instance_url + "/api/v3/reportExport/{}/comment-groups.csv".format(self.report_id))
        # comment_groups_csv = r.text
        # print(comment_groups_csv)
        raise NotImplementedError

    def load_file_data(self):
        """
        Load data from local JSON files specified in self.filepaths.

        Automatically detects file types based on filename patterns:
        - votes.json: Vote records
        - comments.json: Statement/comment data
        - conversation.json: Conversation metadata
        - math-pca2.json: Mathematical analysis results

        Raises:
            ValueError: If a file type cannot be determined from its name.
        """
        for f in self.filepaths:
            if f.endswith("votes.json"):
                self.load_file_data_votes(file=f)
            elif f.endswith("comments.json"):
                self.load_file_data_comments(file=f)
            elif f.endswith("conversation.json"):
                self.load_file_data_conversation(file=f)
            elif f.endswith("math-pca2.json"):
                self.load_file_data_math(file=f)
            else:
                raise ValueError("Unknown file type")

    def load_file_data_votes(self, file=None):
        """
        Load vote data from a local JSON file.

        Args:
            file (str): Path to the votes JSON file.
        """
        with open(file) as f:
            votes_data = json.load(f)

        votes_data = [Vote(**vote).model_dump(mode="json") for vote in votes_data]
        self.votes_data = votes_data

    def load_file_data_comments(self, file=None):
        """
        Load statement/comment data from a local JSON file.

        Args:
            file (str): Path to the comments JSON file.
        """
        with open(file) as f:
            comments_data = json.load(f)

        comments_data = [Statement(**c).model_dump(mode="json") for c in comments_data]
        self.comments_data = comments_data

    def load_file_data_conversation(self, file=None):
        """
        Load conversation metadata from a local JSON file.

        Args:
            file (str): Path to the conversation JSON file.
        """
        with open(file) as f:
            convo_data = json.load(f)

        self.conversation_data = convo_data

    def load_file_data_math(self, file=None):
        """
        Load mathematical analysis data from a local JSON file.

        Args:
            file (str): Path to the math-pca2 JSON file.
        """
        with open(file) as f:
            math_data = json.load(f)

        self.math_data = math_data

    def load_api_data(self):
        """
        Load complete dataset from Polis API endpoints.

        Loads data in the following order:
        1. Report data (if report_id provided) to get conversation_id
        2. Conversation metadata
        3. Comments/statements data
        4. Mathematical analysis data (PCA, clustering)
        5. Individual participant votes (up to participant count from math data)

        Automatically handles vote sign correction for API data and resolves
        any conflicts between report_id and conversation_id parameters.

        Raises:
            ValueError: If report_id conflicts with conversation_id.
        """
        if self.report_id:
            self.load_api_data_report()
            convo_id_from_report_id = self.report_data["conversation_id"]
            if self.conversation_id and (
                self.conversation_id != convo_id_from_report_id
            ):
                raise ValueError("report_id conflicts with conversation_id")
            self.conversation_id = convo_id_from_report_id

        self.load_api_data_conversation()
        self.load_api_data_comments()
        self.load_api_data_math()
        # TODO: Add a way to do this without math data, for example
        # by checking until 5 empty responses in a row.
        # This is the best place to check though, as `voters`
        # in summary.csv omits some participants.
        participant_count = self.math_data["n"]
        # DANGER: This is potentially an issue that throws everything off by missing some participants.
        self.load_api_data_votes(last_participant_id=participant_count)

    def load_api_data_report(self):
        """
        Load report metadata from the Polis API.

        Uses the report_id to fetch report information and extract the associated
        conversation_id for subsequent API calls.
        """
        params = {
            "report_id": self.report_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/reports", params=params)
        reports = json.loads(r.text)
        self.report_data = reports[0]

    def load_api_data_conversation(self):
        """
        Load conversation metadata from the Polis API.

        Fetches conversation details including topic, description, and settings
        using the conversation_id.
        """
        params = {
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(
            self.polis_instance_url + "/api/v3/conversations", params=params
        )
        convo = json.loads(r.text)
        self.conversation_data = convo

    def load_api_data_math(self):
        """
        Load mathematical analysis data from the Polis API.

        Fetches PCA projections, clustering results, and group statistics
        from the math/pca2 endpoint.
        """
        params = {
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(
            self.polis_instance_url + "/api/v3/math/pca2", params=params
        )
        math = json.loads(r.text)
        self.math_data = math

    def load_api_data_comments(self):
        """
        Load statement/comment data from the Polis API.

        Fetches all statements with moderation status and voting patterns
        included in the response.
        """
        params = {
            "conversation_id": self.conversation_id,
            "moderation": "true",
            "include_voting_patterns": "true",
        }
        r = self.session.get(
            self.polis_instance_url + "/api/v3/comments", params=params
        )
        comments = json.loads(r.text)
        comments = [Statement(**c).model_dump(mode="json") for c in comments]
        self.comments_data = comments

    def fix_participant_vote_sign(self):
        """
        Correct vote sign inversion in API data.

        The Polis API returns votes with inverted signs compared to the expected
        format (e.g., agree votes come as -1 instead of 1). This method fixes
        the inversion by negating all vote values.
        """
        """For data coming from the API, vote signs are inverted (e.g., agree is -1)"""
        for item in self.votes_data:
            item["vote"] = -item["vote"]

    def load_api_data_votes(self, last_participant_id=None):
        """
        Load individual participant votes from the Polis API.

        Args:
            last_participant_id (int): Maximum participant ID to fetch votes for.
                                     Typically obtained from math data participant count.

        Iterates through all participant IDs from 0 to last_participant_id and
        fetches their vote records. Automatically applies vote sign correction.
        """
        for pid in range(0, last_participant_id + 1):
            params = {
                "pid": pid,
                "conversation_id": self.conversation_id,
            }
            r = self.session.get(
                self.polis_instance_url + "/api/v3/votes", params=params
            )
            participant_votes = json.loads(r.text)
            participant_votes = [
                Vote(**vote).model_dump(mode="json") for vote in participant_votes
            ]
            self.votes_data.extend(participant_votes)

        self.fix_participant_vote_sign()

    def fetch_pid(self, xid):
        """
        Fetch internal participant ID (pid) for a given external ID (xid).

        Args:
            xid (str): External participant identifier.

        Returns:
            int: Internal participant ID used by Polis system.
        """
        params = {
            "pid": "mypid",
            "xid": xid,
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(
            self.polis_instance_url + "/api/v3/participationInit", params=params
        )
        data = json.loads(r.text)

        return data["ptpt"]["pid"]

    def fetch_xid_to_pid_mappings(self, xids=[]):
        """
        Create mapping dictionary from external IDs to internal participant IDs.

        Args:
            xids (list[str]): List of external participant identifiers.

        Returns:
            dict: Mapping of external IDs to internal participant IDs.
        """
        mappings = {}
        for xid in xids:
            pid = self.fetch_pid(xid)
            mappings[xid] = pid

        return mappings
