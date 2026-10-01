"""
verto/api/client.py
~~~~~~~~~~~~~~~~~~~
HTTP client for communicating with a remote VERTO server.
Built on the standard library, so the CLI can query a server without PyTorch,
model checkpoints, or Spider dataset files installed locally.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

# A question can take several beam-search generations, which is slow on a CPU server.
DEFAULT_TIMEOUT = 300.0


def _detail(err: urllib.error.HTTPError) -> str:
    """The server's error message: FastAPI's "detail" field, or the raw body."""
    body = err.read().decode("utf-8", "replace")
    try:
        return str(json.loads(body).get("detail", body))
    except (ValueError, AttributeError):
        return body


class VERTOClient:
    """Remote HTTP Client matching the query and database listing interface of NLDIEngine."""

    def __init__(self, server_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.server_url}{path}", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"VERTO server at {self.server_url} returned HTTP {e.code}: {_detail(e)}") from e
        except TimeoutError as e:
            raise TimeoutError(f"VERTO server at {self.server_url} did not answer within {self.timeout:g} s") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Could not connect to VERTO server at {self.server_url}: {e.reason}") from e

    def health(self) -> dict[str, Any]:
        """Perform liveness check on the remote VERTO server."""
        return self._get("/api/health")

    def list_databases(self) -> list[str]:
        """Fetch available database IDs from the remote server."""
        return self._get("/api/databases").get("databases", [])

    def query(
        self,
        question: str,
        db_id: str,
        use_schema_linking: bool = True,
        use_error_recovery: bool = True,
    ) -> dict[str, Any]:
        """Send a question to the remote server.

        Failures come back in the result's ``error`` field, never as an exception.
        """
        payload = json.dumps({
            "question": question,
            "db_id": db_id,
            "use_schema_linking": use_schema_linking,
            "use_error_recovery": use_error_recovery,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.server_url}/api/query",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {_detail(e)}"
        except TimeoutError:
            error = f"No answer from {self.server_url} within {self.timeout:g} s"
        except (urllib.error.URLError, OSError) as e:
            error = f"Connection error: {getattr(e, 'reason', e)}"
        return {"sql": None, "columns": None, "result": None, "attempts": 0, "error": error}
