"""Package entry point: enables ``python -m trans``.

Replaces the legacy top-level ``trans_cli.py`` shim. The PyPI
``console_scripts`` entry already routes ``trans`` directly to
``trans.cli:app``; this file lets the dev invocation work without
the shim.
"""

from trans.cli import app

if __name__ == "__main__":
    app()
