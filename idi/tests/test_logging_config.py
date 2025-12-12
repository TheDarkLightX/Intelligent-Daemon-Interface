import logging

from idi.logging import LoggingOptions, configure_logging


def test_logging_redacts_secrets_in_message(capfd) -> None:
    configure_logging(LoggingOptions(level="INFO", format="text", redact=True))
    logger = logging.getLogger("idi.test")
    logger.info("api_key=SUPERSECRET")

    captured = capfd.readouterr()
    assert "SUPERSECRET" not in captured.err
    assert "[REDACTED]" in captured.err


def test_logging_redacts_secrets_in_context(capfd) -> None:
    configure_logging(LoggingOptions(level="INFO", format="json", redact=True))
    logger = logging.getLogger("idi.test")
    logger.info("hello", extra={"context": {"token": "SUPERSECRET"}})

    captured = capfd.readouterr()
    assert "SUPERSECRET" not in captured.err
    assert "[REDACTED]" in captured.err
