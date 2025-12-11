import os
import json
import csv
from datetime import datetime, timezone
from dateutil import parser


class Exporter:
    """
    Handles all JSON/CSV export formats for Polis-compatible data.
    """

    def __init__(self, votes: list, comments: list, math_data: dict, conversation_data: dict, polis_instance_url: str):
        self.votes = votes
        self.comments = comments
        self.math = math_data
        self.conversation = conversation_data
        self.polis_instance_url = polis_instance_url

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def export(self, output_dir, format="csv"):
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
        os.makedirs(output_dir, exist_ok=True)

        if format == "json":
            self._export_json(output_dir)
        elif format == "csv":
            self._export_csv(output_dir)
        else:
            raise ValueError(f"Unknown format: {format}")

    # ---------------------------------------------------------
    # JSON
    # ---------------------------------------------------------
    def _export_json(self, output_dir):
        self._write_json(output_dir, "votes.json", self.votes)
        self._write_json(output_dir, "comments.json", self.comments)
        self._write_json(output_dir, "math-pca2.json", self.math)
        self._write_json(output_dir, "conversation.json", self.conversation)

    def _write_json(self, output_dir, filename, data):
        if not data:
            return
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------
    def _export_csv(self, output_dir):
        self._write_votes_csv(output_dir)
        self._write_comments_csv(output_dir)
        self._write_comment_groups_csv(output_dir)
        self._write_participant_votes_csv(output_dir)
        self._write_summary_csv(output_dir)

    # ---------------------------------------------------------
    # Shared time formatter
    # ---------------------------------------------------------
    def _format_polis_times(self, value):
        try:
            if isinstance(value, (int, float)):
                ts = int(str(value)[:10])
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = parser.parse(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

            dt = dt.astimezone(timezone.utc)
            formatted = dt.strftime(
                "%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)"
            )
            return int(dt.timestamp()), formatted
        except Exception as e:
            raise ValueError(f"Invalid timestamp: {value}: {e}")

    # ---------------------------------------------------------
    # Votes CSV
    # ---------------------------------------------------------
    def _write_votes_csv(self, output_dir):
        """
        POLIS format:
            timestamp,datetime,comment-id,voter-id,vote
        """
        if not self.votes:
            return

        path = os.path.join(output_dir, "votes.csv")
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "datetime", "comment-id", "voter-id", "vote"])

            for v in sorted(self.votes, key=lambda x: (x["statement_id"], x["participant_id"])):
                ts, dt = self._format_polis_times(v["modified"])
                writer.writerow([ts, dt, v["statement_id"], v["participant_id"], v["vote"]])

    # ---------------------------------------------------------
    # Comments CSV
    # ---------------------------------------------------------
    def _write_comments_csv(self, output_dir):
        if not self.comments:
            return

        path = os.path.join(output_dir, "comments.csv")
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

        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for c in sorted(self.comments, key=lambda x: (x["statement_id"], x["participant_id"])):
                ts, dt = self._format_polis_times(c["created"])
                body = c["txt"].replace('"', '""')
                writer.writerow([
                    ts,
                    dt,
                    c["statement_id"],
                    c["participant_id"],
                    c["agree_count"],
                    c["disagree_count"],
                    c["moderated"],
                    f'"{body}"',
                ])

    # ---------------------------------------------------------
    # Comment Groups CSV
    # ---------------------------------------------------------
    def _write_comment_groups_csv(self, output_dir):
        """
        POLIS format:
            comment-id,comment,total-votes,total-agrees,total-disagrees,total-passes,group-a-votes,group-a-agrees,group-a-disagrees,group-a-passes,group-[next alphabetic identifier (b)]-votes,[repeat 'votes/agrees/disagrees/passes' with alphabetic identifier...]

        Each row represents a comment with total votes & votes by group
        """
        if not self.comments or not self.math:
            return

        group_votes = self.math.get("group-votes", {})
        group_clusters = self.math.get("group-clusters", [])
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
                self.comments, key=lambda x: x["statement_id"]
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

    # ---------------------------------------------------------
    # Participant Votes CSV
    # ---------------------------------------------------------
    def _write_participant_votes_csv(self, output_dir):
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
        if not self.votes:
            return

        # Get all unique participant IDs and statement IDs
        participant_ids = set()
        statement_ids = set()
        for vote in self.votes:
            participant_ids.add(vote["participant_id"])
            statement_ids.add(vote["statement_id"])

        # Sort to ensure consistent order
        sorted_participant_ids = sorted(participant_ids)
        sorted_statement_ids = sorted(statement_ids)

        # Build participant vote matrix
        participant_votes = {}
        for vote in self.votes:
            pid = vote["participant_id"]
            sid = vote["statement_id"]
            if pid not in participant_votes:
                participant_votes[pid] = {}
            participant_votes[pid][sid] = vote["vote"]

        # Get participant group assignments from math data
        participant_groups = {}
        if self.math and "group-clusters" in self.math:
            for group in self.math["group-clusters"]:
                group_id = group["id"]
                for member in group["members"]:
                    participant_groups[member] = group_id

        # Count comments per participant
        participant_comment_counts = {}
        if self.comments:
            for comment in self.comments:
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

    # ---------------------------------------------------------
    # Summary CSV
    # ---------------------------------------------------------
    def _write_summary_csv(self, output_dir):
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
        if not self.conversation:
            return

        # Calculate summary statistics
        total_voters = (
            len(set(vote["participant_id"] for vote in self.votes))
            if self.votes
            else 0
        )
        total_commenters = (
            len(set(comment["participant_id"] for comment in self.comments))
            if self.comments
            else 0
        )
        total_comments = len(self.comments) if self.comments else 0
        total_groups = (
            len(self.math.get("group-clusters", [])) if self.math else 0
        )

        # Get conversation details
        topic = self.conversation.get("topic", "")
        description = self.conversation.get("description", "")
        if description:
            description = (
                description.replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )

        # Build URL
        conversation_id = self.conversation.get("conversation_id", "")
        url = f"{self.polis_instance_url}/{conversation_id}"

        with open(output_dir + "/summary.csv", "w") as f:
            f.write(f'topic,"{topic}"\n')
            f.write(f"url,{url}\n")
            f.write(f"voters,{total_voters}\n")
            f.write(f"voters-in-conv,{total_voters}\n")
            f.write(f"commenters,{total_commenters}\n")
            f.write(f"comments,{total_comments}\n")
            f.write(f"groups,{total_groups}\n")
            f.write(f'conversation-description,"{description}"\n')