"""Thin compatibility shim — the real code lives in trans/."""
from trans.cli import app

if __name__ == '__main__':
    app()
