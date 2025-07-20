def preprocess_votes(votes: list) -> list:
    """Convert integer statement and participant IDs to string IDs and append "p" """
    for vote in votes:
        vote["participant_id"] = str(vote["participant_id"]) + "p"
        vote["statement_id"] = str(vote["statement_id"]) + "p"
    return votes