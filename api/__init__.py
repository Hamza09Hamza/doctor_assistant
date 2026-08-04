"""Phase 1 service boundary: a FastAPI wrapper around `pipeline.Pipeline`.

Nothing in `core/`, `experts/`, `reporting/`, or `pipeline.py` changes to support this —
the API layer only adds persistence (`api/models.py`) and HTTP plumbing around the
engine that already exists.
"""
