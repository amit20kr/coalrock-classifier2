import pytest

from tests import require_checkpoint

require_checkpoint(allow_module_level=True)

import backend.main as main

@pytest.mark.parametrize("config", [{}, {"review_unvalidated_input": None}, {"review_unvalidated_input": "false"}, {"review_unvalidated_input": 0}])
def test_missing_or_malformed_review_setting_is_safe(config):
    assert main._review_unvalidated(config) is True

def test_explicit_false_refuses_startup():
    with pytest.raises(RuntimeError, match="review_unvalidated_input"):
        main._review_unvalidated({"review_unvalidated_input": False})

def test_deployed_review_setting_is_enabled():
    assert main.REVIEW_UNVALIDATED_INPUT is True
