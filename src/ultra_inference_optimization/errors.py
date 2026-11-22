"""Named failures. Exit 2 = refused (a precondition said no), 3 = failed."""


class UioError(Exception):
    exit_code = 3

    def __init__(self, cause: str, message: str = ""):
        super().__init__(f"{cause}: {message}" if message else cause)
        self.cause = cause
        self.message = message or cause


class Refused(UioError):
    exit_code = 2


class Failed(UioError):
    exit_code = 3
