# Four Mountains procedural stimulus generation and layout package
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("procedural", "layout", "rendering"):
    _p = os.path.join(_HERE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
