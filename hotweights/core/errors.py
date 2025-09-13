"""Common exceptions for hotweights library."""
from __future__ import annotations


class HotweightsError(Exception):
    pass


class ValidationError(HotweightsError):
    pass


class TransportError(HotweightsError):
    pass


class CoordinatorError(HotweightsError):
    pass

