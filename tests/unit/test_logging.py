import json
import logging

from awo.logging_utils import JsonFormatter


def test_json_formatter() -> None:
    record = logging.LogRecord(
        name="awo.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="ready",
        args=(),
        exc_info=None,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "awo.test"
    assert payload["message"] == "ready"
