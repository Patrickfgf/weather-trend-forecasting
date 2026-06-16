"""Country -> continent and latitude -> climate-zone enrichment.

Geographic enrichment used by ``climate.py`` (two-stage region means) and
``spatial.py`` (choropleths). Two derivations, both pure and vectorized:

- ``continent``: prefer ``pycountry_convert`` when available (authoritative
  ISO-3166 -> continent), else a static alpha-2 fallback dict, else ``"Unknown"``.
  Country *names* in the dataset are messy ("USA", "Russia", "South Korea"), so we
  normalize through an ``ALIASES`` map to ISO alpha-2 BEFORE any lookup.
- ``climate_zone``: a coarse latitude banding (tropics / subtropics / temperate /
  polar). This is a *geographic* zone from |latitude|, NOT a Koppen climate
  classification and NOT inferred from the (short, daily-snapshot) temperature
  history — so it carries no within-sample circularity.

Design choices:
- WHY a UNIQUE-country lookup + ``.map`` instead of row-wise ``.apply``: the frame
  has tens of thousands of rows but ~200 distinct countries; resolving each country
  once and broadcasting via ``.map`` is O(n_unique) not O(n_rows), and avoids the
  Python-level per-row call overhead our conventions forbid.
- WHY ``ALIASES`` is module-level and reused: ``spatial.py`` needs the same
  name->canonical mapping to match plotly choropleth ``locationmode='country names'``
  geometries, so the alias table lives here as the single source of truth.
- Graceful degradation: if ``pycountry_convert`` is missing we ``warnings.warn``
  ONCE and silently use the static dict; we never raise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from weather_forecast import optional_deps

# --------------------------------------------------------------------------- #
# Climate-zone latitude bands (degrees, absolute latitude). Tropic of Cancer/
# Capricorn ~= 23.5; the 35/55 cuts are the conventional subtropical/temperate
# and temperate/polar boundaries used in coarse geographic banding.
# --------------------------------------------------------------------------- #
TROPICAL_MAX_ABS_LAT: float = 23.5
SUBTROPICAL_MAX_ABS_LAT: float = 35.0
TEMPERATE_MAX_ABS_LAT: float = 55.0

# Continent display names (kept stable so categories are consistent across modules).
_CONTINENT_NAMES: dict[str, str] = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "OC": "Oceania",
    "AN": "Antarctica",
}

# --------------------------------------------------------------------------- #
# ALIASES: free-text country name (as it appears in the dataset / common usage)
# -> ISO-3166 alpha-2 code. Lowercased keys; we lowercase the input before lookup
# so matching is case-insensitive. This same table is reused by spatial.py to
# reconcile names with plotly's choropleth geometries.
# --------------------------------------------------------------------------- #
ALIASES: dict[str, str] = {
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "united states": "US",
    "united states of america": "US",
    "america": "US",
    "uk": "GB",
    "u.k.": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "britain": "GB",
    "england": "GB",
    "russia": "RU",
    "russian federation": "RU",
    "south korea": "KR",
    "korea, south": "KR",
    "republic of korea": "KR",
    "north korea": "KP",
    "korea, north": "KP",
    "czech republic": "CZ",
    "czechia": "CZ",
    "slovak republic": "SK",
    "slovakia": "SK",
    "iran": "IR",
    "iran, islamic republic of": "IR",
    "syria": "SY",
    "syrian arab republic": "SY",
    "vietnam": "VN",
    "viet nam": "VN",
    "laos": "LA",
    "lao people's democratic republic": "LA",
    "venezuela": "VE",
    "bolivia": "BO",
    "tanzania": "TZ",
    "united republic of tanzania": "TZ",
    "moldova": "MD",
    "republic of moldova": "MD",
    "macedonia": "MK",
    "north macedonia": "MK",
    "brunei": "BN",
    "brunei darussalam": "BN",
    "ivory coast": "CI",
    "cote d'ivoire": "CI",
    "côte d'ivoire": "CI",
    "cape verde": "CV",
    "cabo verde": "CV",
    "democratic republic of the congo": "CD",
    "dr congo": "CD",
    "congo (kinshasa)": "CD",
    "congo": "CG",
    "republic of the congo": "CG",
    "congo (brazzaville)": "CG",
    "myanmar": "MM",
    "burma": "MM",
    "swaziland": "SZ",
    "eswatini": "SZ",
    "the gambia": "GM",
    "gambia": "GM",
    "the bahamas": "BS",
    "bahamas": "BS",
    "east timor": "TL",
    "timor-leste": "TL",
    "palestine": "PS",
    "palestinian territory": "PS",
    "vatican": "VA",
    "vatican city": "VA",
    "holy see": "VA",
    "taiwan": "TW",
    "hong kong": "HK",
    "macau": "MO",
    "macao": "MO",
    "south sudan": "SS",
    "kosovo": "XK",
}

# --------------------------------------------------------------------------- #
# CONTINENT_BY_ALPHA2: static ISO-3166 alpha-2 -> continent name. Robust fallback
# used when pycountry_convert is unavailable. Antarctica (AQ) intentionally maps to
# Antarctica; transcontinental countries follow pycountry_convert's convention
# (RU/TR -> Asia, etc.) so both code paths agree.
# --------------------------------------------------------------------------- #
CONTINENT_BY_ALPHA2: dict[str, str] = {
    # Africa
    "DZ": "Africa", "AO": "Africa", "BJ": "Africa", "BW": "Africa", "BF": "Africa",
    "BI": "Africa", "CV": "Africa", "CM": "Africa", "CF": "Africa", "TD": "Africa",
    "KM": "Africa", "CG": "Africa", "CD": "Africa", "CI": "Africa", "DJ": "Africa",
    "EG": "Africa", "GQ": "Africa", "ER": "Africa", "SZ": "Africa", "ET": "Africa",
    "GA": "Africa", "GM": "Africa", "GH": "Africa", "GN": "Africa", "GW": "Africa",
    "KE": "Africa", "LS": "Africa", "LR": "Africa", "LY": "Africa", "MG": "Africa",
    "MW": "Africa", "ML": "Africa", "MR": "Africa", "MU": "Africa", "YT": "Africa",
    "MA": "Africa", "MZ": "Africa", "NA": "Africa", "NE": "Africa", "NG": "Africa",
    "RE": "Africa", "RW": "Africa", "SH": "Africa", "ST": "Africa", "SN": "Africa",
    "SC": "Africa", "SL": "Africa", "SO": "Africa", "ZA": "Africa", "SS": "Africa",
    "SD": "Africa", "TZ": "Africa", "TG": "Africa", "TN": "Africa", "UG": "Africa",
    "EH": "Africa", "ZM": "Africa", "ZW": "Africa",
    # Asia
    "AF": "Asia", "AM": "Asia", "AZ": "Asia", "BH": "Asia", "BD": "Asia",
    "BT": "Asia", "BN": "Asia", "KH": "Asia", "CN": "Asia", "CY": "Asia",
    "GE": "Asia", "HK": "Asia", "IN": "Asia", "ID": "Asia", "IR": "Asia",
    "IQ": "Asia", "IL": "Asia", "JP": "Asia", "JO": "Asia", "KZ": "Asia",
    "KW": "Asia", "KG": "Asia", "LA": "Asia", "LB": "Asia", "MO": "Asia",
    "MY": "Asia", "MV": "Asia", "MN": "Asia", "MM": "Asia", "NP": "Asia",
    "KP": "Asia", "OM": "Asia", "PK": "Asia", "PS": "Asia", "PH": "Asia",
    "QA": "Asia", "RU": "Asia", "SA": "Asia", "SG": "Asia", "KR": "Asia",
    "LK": "Asia", "SY": "Asia", "TW": "Asia", "TJ": "Asia", "TH": "Asia",
    "TL": "Asia", "TR": "Asia", "TM": "Asia", "AE": "Asia", "UZ": "Asia",
    "VN": "Asia", "YE": "Asia",
    # Europe
    "AL": "Europe", "AD": "Europe", "AT": "Europe", "BY": "Europe", "BE": "Europe",
    "BA": "Europe", "BG": "Europe", "HR": "Europe", "CZ": "Europe", "DK": "Europe",
    "EE": "Europe", "FO": "Europe", "FI": "Europe", "FR": "Europe", "DE": "Europe",
    "GI": "Europe", "GR": "Europe", "GG": "Europe", "HU": "Europe", "IS": "Europe",
    "IE": "Europe", "IM": "Europe", "IT": "Europe", "JE": "Europe", "XK": "Europe",
    "LV": "Europe", "LI": "Europe", "LT": "Europe", "LU": "Europe", "MT": "Europe",
    "MD": "Europe", "MC": "Europe", "ME": "Europe", "NL": "Europe", "MK": "Europe",
    "NO": "Europe", "PL": "Europe", "PT": "Europe", "RO": "Europe", "SM": "Europe",
    "RS": "Europe", "SK": "Europe", "SI": "Europe", "ES": "Europe", "SE": "Europe",
    "CH": "Europe", "UA": "Europe", "GB": "Europe", "VA": "Europe",
    # North America
    "AI": "North America", "AG": "North America", "AW": "North America",
    "BS": "North America", "BB": "North America", "BZ": "North America",
    "BM": "North America", "CA": "North America", "KY": "North America",
    "CR": "North America", "CU": "North America", "CW": "North America",
    "DM": "North America", "DO": "North America", "SV": "North America",
    "GL": "North America", "GD": "North America", "GP": "North America",
    "GT": "North America", "HT": "North America", "HN": "North America",
    "JM": "North America", "MQ": "North America", "MX": "North America",
    "MS": "North America", "NI": "North America", "PA": "North America",
    "PR": "North America", "BL": "North America", "KN": "North America",
    "LC": "North America", "MF": "North America", "PM": "North America",
    "VC": "North America", "TT": "North America", "TC": "North America",
    "US": "North America", "VG": "North America", "VI": "North America",
    # South America
    "AR": "South America", "BO": "South America", "BR": "South America",
    "CL": "South America", "CO": "South America", "EC": "South America",
    "FK": "South America", "GF": "South America", "GY": "South America",
    "PY": "South America", "PE": "South America", "SR": "South America",
    "UY": "South America", "VE": "South America",
    # Oceania
    "AS": "Oceania", "AU": "Oceania", "CK": "Oceania", "FJ": "Oceania",
    "PF": "Oceania", "GU": "Oceania", "KI": "Oceania", "MH": "Oceania",
    "FM": "Oceania", "NR": "Oceania", "NC": "Oceania", "NZ": "Oceania",
    "NU": "Oceania", "NF": "Oceania", "MP": "Oceania", "PW": "Oceania",
    "PG": "Oceania", "PN": "Oceania", "WS": "Oceania", "SB": "Oceania",
    "TK": "Oceania", "TO": "Oceania", "TV": "Oceania", "VU": "Oceania",
    "WF": "Oceania",
    # Antarctica
    "AQ": "Antarctica", "BV": "Antarctica", "GS": "Antarctica",
    "HM": "Antarctica", "TF": "Antarctica",
}

_UNKNOWN: str = "Unknown"
_warned_missing_pcc: bool = False


def _normalize_to_alpha2(name: str) -> str | None:
    """Map a raw country name to an ISO alpha-2 code (None if unresolved).

    WHY first check ALIASES then a 2-letter passthrough: the dataset uses common
    names, but already-alpha2 values (and exact uppercase codes) should pass through
    so we never lose a valid code to a missing alias entry.
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    if not key:
        return None
    if key in ALIASES:
        return ALIASES[key]
    # Already an alpha-2 code (e.g. "BR", "us") -> normalize to uppercase.
    if len(key) == 2 and key.isalpha():
        return key.upper()
    # Fuzzy fallback: resolve full country names ("Australia", "Brazil", "India", ...)
    # via pycountry_convert. Without this, most full names fall through to "Unknown".
    if optional_deps.HAS_PYCOUNTRY_CONVERT:
        try:
            # NOTE: this pycountry_convert build's signature is (name) with no cn_fuzzy kwarg.
            return optional_deps.pycountry_convert.country_name_to_country_alpha2(name)
        except Exception:
            return None
    return None


def _continent_via_pycountry(alpha2: str) -> str | None:
    """Resolve continent name from alpha-2 via pycountry_convert (None on failure)."""
    pcc = optional_deps.pycountry_convert
    try:
        code = pcc.country_alpha2_to_continent_code(alpha2)
    except Exception:
        # KeyError for codes pycountry_convert doesn't know (e.g. some territories).
        return None
    return _CONTINENT_NAMES.get(code, _UNKNOWN)


def _resolve_continent(name: str, *, use_pcc: bool) -> str:
    """Resolve one country name to a continent, preferring pycountry_convert.

    Resolution order: name -> alpha-2 (alias-aware) -> continent (library, then
    static dict, then "Unknown"). If the name itself maps to a known continent name
    directly (already a continent), that is left to the caller's static dict.
    """
    alpha2 = _normalize_to_alpha2(name)
    if alpha2 is None:
        return _UNKNOWN
    if use_pcc:
        via_lib = _continent_via_pycountry(alpha2)
        if via_lib is not None:
            return via_lib
    return CONTINENT_BY_ALPHA2.get(alpha2, _UNKNOWN)


def _build_continent_lookup(countries: pd.Index) -> dict[str, str]:
    """Build a {raw_country -> continent} lookup over DISTINCT countries only.

    WHY a unique lookup: resolving ~200 distinct countries once and broadcasting via
    ``Series.map`` is far cheaper than a row-wise ``apply`` over the full frame, and
    keeps the function vectorized per our conventions.
    """
    global _warned_missing_pcc
    use_pcc = optional_deps.HAS_PYCOUNTRY_CONVERT
    if not use_pcc and not _warned_missing_pcc:
        warnings.warn(
            "pycountry_convert not installed; falling back to the static "
            "CONTINENT_BY_ALPHA2 map for continent enrichment.",
            RuntimeWarning,
            stacklevel=2,
        )
        _warned_missing_pcc = True
    return {c: _resolve_continent(c, use_pcc=use_pcc) for c in countries}


def _climate_zone_from_abs_lat(abs_lat: pd.Series) -> pd.Series:
    """Vectorized latitude -> climate-zone band; NaN latitude -> "Unknown".

    WHY ``np.select`` over chained ``.where``: a single pass with explicit ordered
    conditions reads as the banding table it represents and stays NaN-safe (NaN
    fails every ``<=`` comparison and falls through to the default).
    """
    conditions = [
        abs_lat <= TROPICAL_MAX_ABS_LAT,
        abs_lat <= SUBTROPICAL_MAX_ABS_LAT,
        abs_lat <= TEMPERATE_MAX_ABS_LAT,
        abs_lat > TEMPERATE_MAX_ABS_LAT,
    ]
    choices = ["Tropical", "Subtropical", "Temperate", "Polar"]
    return pd.Series(
        np.select(conditions, choices, default=_UNKNOWN),
        index=abs_lat.index,
        dtype="object",
    )


def add_region_columns(
    df: pd.DataFrame,
    country_col: str = "country",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``continent``, ``climate_zone`` and ``hemisphere``.

    Pure: never mutates ``df``. Columns are categoricals so downstream groupbys are
    cheap and category sets stay explicit. Missing columns or unresolved values
    degrade to "Unknown" rather than raising — the enrichment is best-effort metadata.
    """
    out = df.copy()

    # --- continent (unique-country lookup, then broadcast via .map) -----------
    if country_col in out.columns:
        countries = out[country_col].astype("string")
        lookup = _build_continent_lookup(countries.dropna().unique())
        continent = countries.map(lookup).fillna(_UNKNOWN)
    else:
        continent = pd.Series(_UNKNOWN, index=out.index)

    # --- climate_zone + hemisphere from latitude ------------------------------
    if lat_col in out.columns:
        lat = pd.to_numeric(out[lat_col], errors="coerce")
        abs_lat = lat.abs()
        climate_zone = _climate_zone_from_abs_lat(abs_lat)
        # sign(lat) > 0 -> N, < 0 -> S, NaN/0 handled: 0 is on the equator -> "N".
        hemisphere = pd.Series(
            np.where(lat.isna(), _UNKNOWN, np.where(lat < 0, "S", "N")),
            index=out.index,
            dtype="object",
        )
    else:
        climate_zone = pd.Series(_UNKNOWN, index=out.index)
        hemisphere = pd.Series(_UNKNOWN, index=out.index)

    # Respect an upstream hemisphere definition (engineering.add_calendar owns it) so the
    # two modules never disagree on the convention for the same column.
    if "hemisphere" in df.columns:
        hemisphere = out["hemisphere"]

    return out.assign(
        continent=pd.Categorical(continent),
        climate_zone=pd.Categorical(climate_zone),
        hemisphere=pd.Categorical(hemisphere),
    )
