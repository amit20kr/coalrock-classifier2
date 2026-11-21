from pathlib import Path
import pytest

from tests import require_checkpoint

require_checkpoint(allow_module_level=True)

import backend.main as main

def test_modified_checkpoint_refused(tmp_path):
    source = Path(main._MODEL_PATH).read_bytes()
    changed = bytearray(source)
    changed[-1] ^= 1
    copy = tmp_path / "changed.pth"
    copy.write_bytes(changed)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        main._verify_weights(copy)
    assert Path(main._MODEL_PATH).read_bytes() == source

def test_original_checkpoint_accepted():
    assert main._verify_weights(main._MODEL_PATH) == main.WEIGHTS_SHA256
