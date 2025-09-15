from __future__ import annotations

import os

from hotweights.utils.env import env_bool, env_int, env_mb, env_list_int


def test_env_bool_parsing(monkeypatch):
    monkeypatch.setenv("X_BOOL", "1")
    assert env_bool("X_BOOL") is True
    monkeypatch.setenv("X_BOOL", "true")
    assert env_bool("X_BOOL") is True
    monkeypatch.setenv("X_BOOL", "no")
    assert env_bool("X_BOOL", True) is False
    monkeypatch.delenv("X_BOOL", raising=False)
    assert env_bool("X_BOOL", False) is False


def test_env_int_and_mb(monkeypatch):
    monkeypatch.setenv("X_INT", "5")
    assert env_int("X_INT", 1) == 5
    assert env_int("X_MISSING", 7) == 7
    assert env_int("X_NEG", -3, minimum=0) == 0
    monkeypatch.setenv("X_MB", "2")
    assert env_mb("X_MB", 1) == (2 << 20)


def test_env_list_int(monkeypatch):
    monkeypatch.setenv("X_LIST", "1, 2, 3")
    assert env_list_int("X_LIST", [9]) == [1, 2, 3]
    monkeypatch.setenv("X_LIST", "")
    assert env_list_int("X_LIST", [9]) == [9]
    monkeypatch.setenv("X_LIST", "a, b")
    assert env_list_int("X_LIST", [4, 5]) == [4, 5]

