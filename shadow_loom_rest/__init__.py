# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shadow-Loom REST adapter.

A thin FastAPI transport that mirrors the MCP service. It does not
reimplement any tool logic: every endpoint dispatches to the exact same
tool functions registered on the MCP server, passing a transport-neutral
:class:`shadow_loom_mcp.auth.Principal` in the ``ctx`` slot so that all
authentication, scope/role checks, rate limits, world-state loading, and
pipeline execution are shared verbatim with the MCP transport.
"""

from shadow_loom_rest.app import app, create_app  # noqa: F401
