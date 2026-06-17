# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the Shadow-Loom MCP server: python -m shadow_loom_mcp

Transport selection (env-driven so the Bearer-token auth model actually
works):

  * ``MCP_TRANSPORT=stdio`` (default) — local / Claude Desktop. There is
    no HTTP ``Authorization`` header over stdio, so Bearer auth cannot
    run; use ``MCP_ALLOW_OPEN_MODE=true`` for trusted local use.
  * ``MCP_TRANSPORT=http`` — networked deployments. Required for the
    Bearer-token auth configured on the server to take effect. Honours
    ``MCP_HOST`` (default ``127.0.0.1``) and ``MCP_PORT``/``PORT``
    (default ``8000``).
"""

import os

from shadow_loom_mcp.server import mcp

transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()

if transport == "http":
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT") or os.environ.get("PORT") or "8000")
    mcp.run(transport="http", host=host, port=port)
else:
    mcp.run()
