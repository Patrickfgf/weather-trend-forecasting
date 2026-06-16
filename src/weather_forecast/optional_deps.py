"""Graceful import guards for optional dependencies.

The CORE stack (pandas, numpy, scipy, scikit-learn, statsmodels, xgboost,
matplotlib, seaborn, plotly, pyarrow) is required. Everything probed here may be
absent on a given interpreter (e.g. no wheel for a brand-new Python). Import via
this module so a missing wheel degrades to a logged skip instead of crashing a
Restart-Run-All notebook run.
"""
from __future__ import annotations

import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)


def _try(name: str) -> tuple[ModuleType | None, bool]:
    try:
        return importlib.import_module(name), True
    except Exception as exc:  # ImportError, or a build/runtime error on import
        logger.info("optional dependency %r unavailable (%s)", name, type(exc).__name__)
        return None, False


lightgbm, HAS_LIGHTGBM = _try("lightgbm")
prophet, HAS_PROPHET = _try("prophet")
shap, HAS_SHAP = _try("shap")
pandera, HAS_PANDERA = _try("pandera")
kaleido, HAS_KALEIDO = _try("kaleido")
folium, HAS_FOLIUM = _try("folium")
pycountry_convert, HAS_PYCOUNTRY_CONVERT = _try("pycountry_convert")
statsforecast, HAS_STATSFORECAST = _try("statsforecast")


def summary() -> dict[str, bool]:
    """Mapping of optional dependency -> availability (for logging in the pipeline)."""
    return {
        "lightgbm": HAS_LIGHTGBM,
        "prophet": HAS_PROPHET,
        "shap": HAS_SHAP,
        "pandera": HAS_PANDERA,
        "kaleido": HAS_KALEIDO,
        "folium": HAS_FOLIUM,
        "pycountry_convert": HAS_PYCOUNTRY_CONVERT,
        "statsforecast": HAS_STATSFORECAST,
    }
