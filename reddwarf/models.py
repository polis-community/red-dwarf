from pydantic import (
    BaseModel,
    NonNegativeInt,
    Field,
    AliasChoices,
    field_serializer,
    field_validator,
)
from typing import Literal, Optional, TypeAlias
from datetime import datetime
from enum import IntEnum

class VoteEnum(IntEnum):
    AGREE = 1
    PASS = 0
    DISAGREE = -1

class ModeratedEnum(IntEnum):
    APPROVED = 1
    UNMODERATED = 0
    REJECTED = -1

IncrementingId: TypeAlias = NonNegativeInt

class Vote(BaseModel):
    @field_serializer('modified')
    def serialize_modified(self, modified: datetime, _info):
        return modified.timestamp() * 1000

    participant_id: IncrementingId = Field(
        validation_alias=AliasChoices('participant_id', 'pid', 'voter-id'),
        serialization_alias="participant_id",
    )
    statement_id: IncrementingId = Field(
        validation_alias=AliasChoices('statement_id', 'tid', 'comment-id'),
    )
    vote: VoteEnum
    weight_x_32767: Optional[int] = None
    modified: datetime = Field(
        validation_alias=AliasChoices('modified', 'timestamp'),
        serialization_alias="modified",
    )
    conversation_id: Optional[str] = None
    datetime: Optional[str] = Field(default=None, exclude=True)

class Statement(BaseModel):
    txt: str = Field(
        validation_alias=AliasChoices('txt', 'comment-body'),
    )
    statement_id: IncrementingId = Field(
        validation_alias=AliasChoices('statement_id', 'tid', 'comment-id'),
    )
    created: datetime = Field(
        validation_alias=AliasChoices('timestamp', 'created'),
    )
    tweet_id: Optional[int] = None
    quote_src_url: Optional[str] = None
    is_seed: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices('is_seed', 'is-seed'),
    )
    is_meta: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices('is_meta', 'is-meta'),
    )
    lang: Optional[str] = None
    participant_id: IncrementingId = Field(
        validation_alias=AliasChoices('participant_id', 'pid', 'author-id'),
        serialization_alias="participant_id",
    )
    velocity: Optional[Literal[1]] = None
    # Public API responses can omit moderation metadata; absence is not approval.
    moderated: Optional[ModeratedEnum] = Field(
        default=None,
        validation_alias=AliasChoices('moderated', 'mod'),
    )

    @field_validator("moderated", mode="before")
    @classmethod
    def parse_empty_moderation(cls, value):
        # CSV exports represent unavailable moderation metadata as an empty cell.
        return None if value == "" else value

    active: Optional[bool] = None
    agree_count: Optional[NonNegativeInt] = Field(
        validation_alias=AliasChoices('agree_count', 'agrees'),
        default=None,
    )
    disagree_count: Optional[NonNegativeInt] = Field(
        validation_alias=AliasChoices('disagree_count', 'disagrees'),
        default=None,
    )
    pass_count: Optional[NonNegativeInt] = None
    count: Optional[NonNegativeInt] = None
    conversation_id: Optional[str] = None
    datetime: Optional[str] = Field(default=None, exclude=True)
