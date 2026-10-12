"""Two invariants the whole suite runs under: no network, no ambient configuration.

`adk-python`'s contribution guide asks for unit tests "free of external dependencies",
and both ways of failing that are quiet. A test that opens a socket passes on a laptop
with a network and fails in a sandbox. A test that reads `SMART_ROUTER_BQ_PROJECT` passes
in CI, where nothing is exported, and behaves differently in the shell where
`scripts/deploy_sql.sh` was just run, where everything is.

Neither is a live bug today: the suite is green with a full developer environment
exported, and nothing in it reaches the network. These fixtures keep it that way, and
turn the next violation into a failure at the offending line instead of a mystery in
somebody else's sandbox.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest

# Loopback and the empty host are the addresses a test may legitimately reach: the MCP
# stub server in tests/mcp_stub_server.py is a real process on this machine, and asyncio
# uses socketpairs on some platforms.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "localhost.localdomain", "", "0.0.0.0"})

# Cleared before every test. The first two prefixes are this project's own knobs and the
# credentials that would let a client actually connect; the rest are what the GCP client
# libraries and gcloud leave behind in a shell that has deployed something.
_CLEARED_PREFIXES = (
    "SMART_ROUTER_",
    "GOOGLE_GENAI_",
    "GOOGLE_CLOUD_",
    "CLOUDSDK_",
)
_CLEARED_VARS = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_API_KEY",
    "GCLOUD_PROJECT",
    "GCP_PROJECT",
)


class NetworkUseInTest(RuntimeError):
    """A test tried to reach something that is not on this machine."""


def _host_of(address: Any) -> str | None:
    """The hostname in a socket address, or None when there is not one.

    AF_UNIX addresses are strings and AF_NETLINK ones are tuples of ints; neither leaves
    the machine, so both come back as None and are allowed through.
    """
    if isinstance(address, tuple) and address and isinstance(address[0], str):
        return address[0]
    return None


def _check(address: Any, call: str) -> None:
    host = _host_of(address)
    if host is not None and host not in _LOCAL_HOSTS:
        raise NetworkUseInTest(
            f"{call} to {host!r} during a test. Unit tests must not touch the network: "
            f"mock the client, or move the check to scripts/ where a live call belongs."
        )


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail loudly on any connection or name lookup that leaves this machine."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        _check(address, "socket.connect")
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        _check(address, "socket.connect_ex")
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        _check((host, port), "socket.getaddrinfo")
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield


@pytest.fixture(autouse=True)
def no_ambient_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test in an empty environment, whatever the developer's shell holds.

    `RouterConfig.from_env` reads the operational knobs straight off `os.environ`, so a
    test that builds a config without passing one explicitly would otherwise inherit
    whichever project, bucket and mode the last script export left behind.
    """
    for name in list(os.environ):
        if name.startswith(_CLEARED_PREFIXES) or name in _CLEARED_VARS:
            monkeypatch.delenv(name, raising=False)
    yield
