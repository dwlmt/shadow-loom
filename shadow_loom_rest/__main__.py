# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the Shadow-Loom REST API: ``python -m shadow_loom_rest``.

Honours ``REST_HOST`` (default ``127.0.0.1``) and ``REST_PORT`` /
``PORT`` (default ``8100``). Authentication uses the same Bearer API
keys as the MCP server; in production set ``AUTH_REQUIRED=true`` so the
open-mode fallback stays disabled.
"""

import os

import uvicorn

host = os.environ.get("REST_HOST", "127.0.0.1")
port = int(os.environ.get("REST_PORT") or os.environ.get("PORT") or "8100")

uvicorn.run("shadow_loom_rest.app:app", host=host, port=port)
