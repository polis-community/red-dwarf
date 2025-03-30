import json
import os
from fake_useragent import UserAgent
from datetime import timedelta
from requests_ratelimiter import SQLiteBucket, LimiterSession
import csv
from io import StringIO
from reddwarf.models import Vote, Statement
from reddwarf.helpers import RedDwarfSession, CloudflareBypassHTTPAdapter

ua = UserAgent()

class Loader():
    def __init__(self, filepaths=[], conversation_id=None, report_id=None, is_cache_enabled=True, is_ip_rotation_enabled=False, output_dir=None, data_source="api", directory_url=None):
        self.polis_instance_url = "https://pol.is"
        self.conversation_id = conversation_id
        self.report_id = report_id
        self.is_cache_enabled = is_cache_enabled
        self.is_ip_rotation_enabled = is_ip_rotation_enabled
        self.output_dir = output_dir
        self.data_source = data_source
        self.filepaths = filepaths
        self.directory_url = directory_url

        self.votes_data = []
        self.comments_data = []
        self.math_data = {}
        self.conversation_data = {}

        if self.filepaths:
            self.load_file_data()
        elif self.conversation_id or self.report_id or self.directory_url:
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

    def dump_data(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.votes_data:
            with open(output_dir + "/votes.json", 'w') as f:
                f.write(json.dumps(self.votes_data, indent=4))

        if self.comments_data:
            with open(output_dir + "/comments.json", 'w') as f:
                f.write(json.dumps(self.comments_data, indent=4))

        if self.math_data:
            with open(output_dir + "/math-pca2.json", 'w') as f:
                f.write(json.dumps(self.math_data, indent=4))

        if self.conversation_data:
            with open(output_dir + "/conversation.json", 'w') as f:
                f.write(json.dumps(self.conversation_data, indent=4))


    def init_http_client(self):
        # Throttle requests, but disable when response is already cached.
        if self.is_cache_enabled:
            # Source: https://github.com/JWCook/requests-ratelimiter/tree/main?tab=readme-ov-file#custom-session-example-requests-cache
            self.session = RedDwarfSession(
                per_second=5,
                expire_after=timedelta(hours=1),
                cache_name="test_cache.sqlite",
                bucket_class=SQLiteBucket,
                bucket_kwargs={
                    "path": "test_cache.sqlite",
                    'isolation_level': "EXCLUSIVE",
                    'check_same_thread': False,
                },
            )

        else:
            self.session = RedDwarfSession(per_second=5)
            self.session.cache._settings.disabled = True

        if self.is_ip_rotation_enabled:
            self.session.use_ip_rotation(self.polis_instance_url)

        adapter = CloudflareBypassHTTPAdapter()
        self.session.mount(self.polis_instance_url, adapter)
        self.session.headers = {
            'User-Agent': ua.random,
        }

    def get_polis_export_directory_url(self, report_id):
        return f"{self.polis_instance_url}/api/v3/reportExport/{report_id}/"

    def load_remote_export_data(self):
        if self.directory_url:
            directory_url = self.directory_url
        elif self.report_id:
            directory_url = self.get_polis_export_directory_url(self.report_id)
        else:
            raise ValueError("Cannot determine CSV export URL without report_id or directory_url")

        self.load_remote_export_data_comments(directory_url)
        self.load_remote_export_data_votes(directory_url)


        # When multiple votes (same tid and pid), keep only most recent (vs first).
        self.filter_duplicate_votes(keep="recent")
        # self.load_remote_export_data_summary()
        # self.load_remote_export_data_participant_votes()
        # self.load_remote_export_data_comment_groups()

    def load_remote_export_data_comments(self, directory_url):
        r = self.session.get(directory_url + "comments.csv")
        comments_csv = r.text
        reader = csv.DictReader(StringIO(comments_csv))
        self.comments_data = [Statement(**c).model_dump(mode='json') for c in list(reader)]

    def load_remote_export_data_votes(self, directory_url):
        r = self.session.get(directory_url + "votes.csv")
        votes_csv = r.text
        reader = csv.DictReader(StringIO(votes_csv))
        self.votes_data = [Vote(**vote).model_dump(mode='json') for vote in list(reader)]

    def filter_duplicate_votes(self, keep="recent"):
        if keep not in {"recent", "first"}:
            raise ValueError("Invalid value for 'keep'. Use 'recent' or 'first'.")

        # Sort by modified time (descending for "recent", ascending for "first")
        if keep == "recent":
            reverse_sort = True
        else:
            reverse_sort = False
        sorted_votes = sorted(self.votes_data, key=lambda x: x["modified"], reverse=reverse_sort)

        filtered_dict = {}
        for v in sorted_votes:
            key = (v["participant_id"], v["statement_id"])
            if key not in filtered_dict:
                filtered_dict[key] = v
            else:
                print("Removing duplicate vote: {}".format(v))

        self.votes_data = list(filtered_dict.values())


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
        for f in self.filepaths:
            if f.endswith("votes.json"):
                self.load_file_data_votes(file=f)
            elif f.endswith("comments.json"):
                self.load_file_data_comments(file=f)
            else:
                raise ValueError("Unknown file type")

    def load_file_data_votes(self, file=None):
        with open(file) as f:
            votes_data = json.load(f)

        votes_data = [Vote(**vote).model_dump(mode='json') for vote in votes_data]
        self.votes_data = votes_data

    def load_file_data_comments(self, file=None):
        with open(file) as f:
            comments_data = json.load(f)

        comments_data = [Statement(**c).model_dump(mode='json') for c in comments_data]
        self.comments_data = comments_data

    def load_api_data(self):
        if self.report_id:
            self.load_api_data_report()
            convo_id_from_report_id = self.report_data["conversation_id"]
            if self.conversation_id and (self.conversation_id != convo_id_from_report_id):
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
        params = {
            "report_id": self.report_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/reports", params=params)
        reports = json.loads(r.text)
        self.report_data = reports[0]

    def load_api_data_conversation(self):
        params = {
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/conversations", params=params)
        convo = json.loads(r.text)
        self.conversation_data = convo

    def load_api_data_math(self):
        params = {
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/math/pca2", params=params)
        math = json.loads(r.text)
        self.math_data = math

    def load_api_data_comments(self):
        params = {
            "conversation_id": self.conversation_id,
            "moderation": "true",
            "include_voting_patterns": "true",
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/comments", params=params)
        comments = json.loads(r.text)
        comments = [Statement(**c).model_dump(mode='json') for c in comments]
        self.comments_data = comments

    def fix_participant_vote_sign(self):
        """For data coming from the API, vote signs are inverted (e.g., agree is -1)"""
        for item in self.votes_data:
            item["vote"] = -item["vote"]

    def load_api_data_votes(self, last_participant_id=None):
        for pid in range(0, last_participant_id+1):
            params = {
                "pid": pid,
                "conversation_id": self.conversation_id,
            }
            r = self.session.get(self.polis_instance_url + "/api/v3/votes", params=params)
            participant_votes = json.loads(r.text)
            participant_votes = [Vote(**vote).model_dump(mode='json') for vote in participant_votes]
            self.votes_data.extend(participant_votes)

        self.fix_participant_vote_sign()

    def fetch_pid(self, xid):
        params = {
            "pid": "mypid",
            "xid": xid,
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/participationInit", params=params)
        data = json.loads(r.text)

        return data["ptpt"]["pid"]

    def fetch_xid_to_pid_mappings(self, xids=[]):
        mappings = {}
        for xid in xids:
            pid = self.fetch_pid(xid)
            mappings[xid] = pid

        return mappings
