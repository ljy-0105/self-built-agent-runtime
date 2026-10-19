class SessionNotFoundError(LookupError):
    """Raised when a session is missing or is not owned by the requested user."""
