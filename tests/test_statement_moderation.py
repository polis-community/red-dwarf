import pytest
from pydantic import ValidationError

from reddwarf.models import ModeratedEnum, Statement
from reddwarf.utils.statements import process_statements


def _statement_data():
    return {
        "txt": "A statement",
        "tid": 7,
        "created": 1700000000000,
        "pid": 0,
        "is_meta": False,
    }


@pytest.mark.parametrize("moderation_fields", [{}, {"mod": None}, {"moderated": ""}])
def test_statement_preserves_unknown_moderation(moderation_fields):
    statement = Statement.model_validate({**_statement_data(), **moderation_fields})

    assert statement.moderated is None
    assert statement.model_dump(mode="json")["moderated"] is None


@pytest.mark.parametrize("field", ["mod", "moderated"])
@pytest.mark.parametrize("status", list(ModeratedEnum))
def test_statement_preserves_known_moderation(field, status):
    statement = Statement.model_validate({**_statement_data(), field: status.value})

    assert statement.moderated == status


def test_statement_rejects_invalid_moderation():
    with pytest.raises(ValidationError):
        Statement.model_validate({**_statement_data(), "mod": 2})


@pytest.mark.parametrize("polis_backward_compat", [True, False])
def test_unknown_moderation_does_not_manufacture_approval(polis_backward_compat):
    statement = Statement.model_validate(_statement_data()).model_dump(mode="json")

    _, mod_in, mod_out, _ = process_statements(
        statement_data=[statement],
        polis_backward_compat=polis_backward_compat,
    )

    assert mod_in == []
    assert mod_out == []


def test_strict_moderation_requires_known_status():
    statement = Statement.model_validate(_statement_data()).model_dump(mode="json")

    with pytest.raises(ValueError, match="Strict moderation requires moderation metadata"):
        process_statements(
            statement_data=[statement],
            polis_backward_compat=False,
            is_strict_moderation=True,
        )
