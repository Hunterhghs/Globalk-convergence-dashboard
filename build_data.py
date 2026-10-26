#!/usr/bin/env python3
"""
Global Convergence Observatory — data pipeline.

Fetches World Bank open data (GDP per capita, population, and diffusion
proxies: electricity, internet, mobile, literacy, urbanization, WGI
governance) and computes the convergence statistics used by the dashboard:

  * sigma-convergence  — cross-country dispersion of log income per year
  * beta-convergence   — OLS of growth on initial log income (1995 -> 2023)
  * catch-up half-life — ln(2) / lambda
  * D-coefficient      — simplified Medium Coefficient (diffusion proxy, 0-100)
  * convergence clubs  — 2x2 of initial income vs growth residual

Outputs:
  data/global-convergence.js   (const GLOBAL_CONVERGENCE = {...})
  data/world-geo.js            (const WORLD_GEOJSON = {...})  [downloaded]

Usage:
  python3 build_data.py
"""

import json, math, os, sys, time, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

API = "https://api.worldbank.org/v2/country/all/indicator"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
)

# (indicator, key, dates, mode) — mode "panel" keeps yearly series, "latest" keeps last non-null
INDICATORS = [
    ("NY.GDP.PCAP.KD", "gdp", "1990:2023", "panel"),
    ("SP.POP.TOTL", "pop", "1990:2023", "panel"),
    ("EG.ELC.ACCS.ZS", "electricity", "2015:2023", "latest"),
    ("IT.NET.USER.ZS", "internet", "2015:2023", "latest"),
    ("IT.CEL.SETS.P2", "mobile", "2015:2023", "latest"),
    ("SE.ADT.LITR.ZS", "literacy", "2015:2023", "latest"),
    ("SE.SEC.ENRR", "enroll", "2015:2023", "latest"),
    ("SP.URB.TOTL.IN.ZS", "urban", "2015:2023", "latest"),
]

T0, T1 = 1995, 2023
YEARS = list(range(1990, 2024))


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hheuristics/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  !! fetch failed: {url} -> {exc}")
                raise
            time.sleep(2 + attempt * 2)


def fetch_indicator(code, dates):
    """Return {iso3: {year_str: value}} from the World Bank API."""
    out = {}
    page = 1
    while True:
        url = f"{API}/{code}?format=json&date={dates}&per_page=20000&page={page}"
        payload = fetch_json(url)
        if not isinstance(payload, list) or len(payload) < 2:
            break
        rows = payload[1]
        if not rows:
            break
        for row in rows:
            iso3 = row.get("countryiso3code")
            value = row.get("value")
            year = row.get("date")
            if iso3 and value is not None and year:
                out.setdefault(iso3, {})[year] = float(value)
        total = int(payload[0].get("total", 0) or 0)
        pages = int(payload[0].get("pages", 0) or 0)
        if page >= pages:
            break
        page += 1
    return out


def latest(series):
    """Most recent non-null value from a {year: value} dict."""
    if not series:
        return None
    for year in sorted(series.keys(), reverse=True):
        if series[year] is not None:
            return series[year]
    return None


def clamp(v, lo=0.0, hi=1.0):
    if v is None:
        return None
    return max(lo, min(hi, v))


def d_coefficient(d):
    """Simplified Medium Coefficient (0-100): diffusion capacity proxy.

    Weights follow the Veles D-Coefficient Engine's digital-era profile,
    renormalised over the four components observable in World Bank WDI data
    (WGI governance estimates are not exposed via api.worldbank.org, so the
    institutional component is folded into the other weights).
    """
    def pct(v):
        return clamp((v / 100.0) if v is not None else None)

    elec = pct(d.get("electricity"))
    mob = clamp((d.get("mobile") or 0) / 120.0)
    net = pct(d.get("internet"))
    lit = pct(d.get("literacy"))
    enr = pct(d.get("enroll"))
    urb = pct(d.get("urban"))

    infra = 0.45 * (elec if elec is not None else 0.8) + 0.30 * mob + 0.25 * (net if net is not None else 0.0)
    # human capital: blend literacy with secondary enrollment, with sane fallbacks
    lit_eff = lit if lit is not None else (enr if enr is not None else 0.7)
    enr_eff = enr if enr is not None else lit_eff
    human = 0.70 * lit_eff + 0.30 * enr_eff
    info = net if net is not None else 0.0
    geo = urb if urb is not None else 0.5

    score = 0.35 * infra + 0.30 * human + 0.20 * info + 0.15 * geo
    return round(score * 100, 1)


def ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return None
    b = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    a = float(y.mean() - b * x.mean())
    resid = y - (a + b * x)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": a, "b": b, "r2": r2, "n": n}


def main():
    print("◆ Global Convergence Observatory — data pipeline")
    print(f"  output dir: {DATA_DIR}")

    # ---- 1. fetch indicators ------------------------------------------------
    raw = {}
    for code, key, dates, _mode in INDICATORS:
        print(f"  fetching {code} ({key}) {dates} ...")
        raw[key] = fetch_indicator(code, dates)
        print(f"    -> {len(raw[key])} countries")

    # ---- 2. assemble country records ----------------------------------------
    countries = {}
    iso3s = sorted(set(raw["gdp"]) & set(raw["pop"]))
    print(f"  panel countries (gdp + pop): {len(iso3s)}")

    for iso3 in iso3s:
        gdp = {y: raw["gdp"][iso3].get(str(y)) for y in YEARS}
        pop = {y: raw["pop"][iso3].get(str(y)) for y in YEARS}
        # drop years with no gdp
        gdp = {y: v for y, v in gdp.items() if v is not None}
        if len(gdp) < 8:
            continue
        d_in = {
            k: latest(raw[k].get(iso3, {}))
            for k in ("electricity", "internet", "mobile", "literacy", "enroll", "urban")
        }
        # name fallback: use iso3
        name = iso3
        countries[iso3] = {
            "name": name,
            "gdp": {str(y): round(v, 1) for y, v in sorted(gdp.items())},
            "pop": {str(y): round(v) for y, v in sorted(pop.items()) if v is not None},
            "d_inputs": {k: (round(v, 2) if v is not None else None) for k, v in d_in.items()},
            "d_coeff": d_coefficient(d_in),
        }

    # ---- 3. names for countries ---------------------------------------------
    # fetch country list once for display names (cheap)
    print("  fetching country metadata ...")
    meta_payload = fetch_json(
        "https://api.worldbank.org/v2/country?format=json&per_page=500"
    )
    for row in (meta_payload[1] if isinstance(meta_payload, list) and len(meta_payload) > 1 else []):
        iso3 = row.get("id")
        if iso3 in countries:
            countries[iso3]["name"] = row.get("name", iso3)
            countries[iso3]["region"] = (row.get("region") or {}).get("value", "")
            countries[iso3]["incomeGroup"] = (row.get("incomeLevel") or {}).get("value", "")

    # ---- 4. per-country growth / gap / ln_y0 --------------------------------
    for iso3, c in countries.items():
        ys = sorted(int(y) for y in c["gdp"])
        y0 = c["gdp"].get(str(T0))
        y1 = c["gdp"].get(str(T1))
        if y0 is None or y1 is None:
            y0 = c["gdp"].get(str(min(ys)))
            y1 = c["gdp"].get(str(max(ys)))
        ln_y0 = math.log(y0) if y0 and y0 > 0 else None
        ln_y1 = math.log(y1) if y1 and y1 > 0 else None
        span = T1 - T0 if (str(T0) in c["gdp"] and str(T1) in c["gdp"]) else (max(ys) - min(ys)) or 1
        growth = ((ln_y1 - ln_y0) / span * 100.0) if ln_y0 is not None and ln_y1 is not None else None
        c["ln_y0"] = round(ln_y0, 4) if ln_y0 is not None else None
        c["growth"] = round(growth, 2) if growth is not None else None
        c["gap_frontier"] = None

    # gap to USA frontier (per year handled client-side; store 2023 here)
    us_gdp_2023 = countries.get("USA", {}).get("gdp", {}).get("2023")
    for iso3, c in countries.items():
        y23 = c["gdp"].get("2023")
        if us_gdp_2023 and y23:
            c["gap_2023"] = round(y23 / us_gdp_2023 * 100.0, 1)
        else:
            c["gap_2023"] = None

    # ---- 5. beta-convergence regression (1995 -> 2023) ----------------------
    beta_rows = [
        (c["ln_y0"], c["growth"])
        for c in countries.values()
        if c["ln_y0"] is not None and c["growth"] is not None
    ]
    beta = None
    if len(beta_rows) >= 10:
        x = [r[0] for r in beta_rows]
        y = [r[1] for r in beta_rows]
        fit = ols(x, y)
        if fit:
            T = T1 - T0
            b = fit["b"]
            lam = None
            half_life = None
            # g is measured in %/yr, so convert the slope to a fraction for the
            # standard speed formula: lambda = -ln(1 + b*T) / T
            b_frac = b / 100.0
            if (1 + b_frac * T) > 0:
                lam = -math.log(1 + b_frac * T) / T
                if lam > 0:
                    half_life = math.log(2) / lam
            beta = {
                "t0": T0, "t1": T1, "a": round(fit["a"], 4), "b": round(b, 4),
                "r2": round(fit["r2"], 4), "n": fit["n"],
                "lambda": round(lam, 5) if lam else None,
                "half_life_years": round(half_life, 1) if half_life else None,
            }
            # club labels via residual around the fitted line
            median_lny = float(np.median(x))
            for iso3, c in countries.items():
                if c["ln_y0"] is None or c["growth"] is None:
                    c["club"] = None
                    continue
                pred = fit["a"] + fit["b"] * c["ln_y0"]
                resid = c["growth"] - pred
                c["residual"] = round(resid, 3)
                poor = c["ln_y0"] < median_lny
                if poor and resid > 0:
                    c["club"] = "Catch-up engine"
                elif poor and resid <= 0:
                    c["club"] = "Lagging"
                elif (not poor) and resid > 0:
                    c["club"] = "Frontier outperformer"
                else:
                    c["club"] = "Frontier steady"

    # ---- 6. sigma-convergence -----------------------------------------------
    sigma = {}
    for year in YEARS:
        ys = str(year)
        vals = [
            math.log(c["gdp"][ys])
            for c in countries.values()
            if ys in c["gdp"] and c["gdp"][ys] and c["gdp"][ys] > 0
        ]
        if len(vals) >= 60:
            sigma[ys] = round(float(np.std(vals, ddof=1)), 4)

    # ---- 7. KPI snapshot -----------------------------------------------------
    n_converging = sum(1 for c in countries.values() if c.get("club") == "Catch-up engine")
    n_lagging = sum(1 for c in countries.values() if c.get("club") == "Lagging")
    n_sample = len(countries)
    bottom_half = [c for c in countries.values() if c.get("club") in ("Catch-up engine", "Lagging")]
    median_growth_bottom = float(np.median([c["growth"] for c in bottom_half if c["growth"] is not None])) if bottom_half else None

    # population-weighted global D
    d_weighted, pop_tot = 0.0, 0.0
    for c in countries.values():
        pop = latest(c["pop"])
        if pop and c.get("d_coeff") is not None:
            d_weighted += c["d_coeff"] * pop
            pop_tot += pop
    d_global = round(d_weighted / pop_tot, 1) if pop_tot else None

    sigma_1995 = sigma.get("1995")
    sigma_2023 = sigma.get("2023")
    kpis = {
        "sigma_1995": sigma_1995,
        "sigma_2023": sigma_2023,
        "sigma_change_pct": round((sigma_2023 - sigma_1995) / sigma_1995 * 100, 1) if sigma_1995 else None,
        "beta_speed_pct": round(beta["lambda"] * 100, 2) if beta and beta.get("lambda") else None,
        "half_life_years": beta.get("half_life_years") if beta else None,
        "d_global": d_global,
        "n_converging": n_converging,
        "n_lagging": n_lagging,
        "n_sample": n_sample,
        "median_growth_bottom_pct": round(median_growth_bottom, 2) if median_growth_bottom is not None else None,
    }

    # ---- 8. write data JS ----------------------------------------------------
    payload = {
        "meta": {
            "title": "Global Convergence Observatory",
            "subtitle": "Beta, sigma & the Medium Coefficient — 1990-2023",
            "source": "World Bank World Development Indicators (via api.worldbank.org)",
            "author": "H Heuristics",
            "last_updated": time.strftime("%Y-%m-%d"),
            "note": "D-coefficient is a simplified Medium Coefficient proxy built from electricity, mobile, internet, literacy and urbanization. GDP per capita is constant 2015 US$.",
        },
        "years": [str(y) for y in YEARS],
        "t0": T0,
        "t1": T1,
        "countries": countries,
        "stats": {"sigma": sigma, "beta": beta, "kpis": kpis},
    }
    out_js = DATA_DIR / "global-convergence.js"
    out_js.write_text(
        "// Generated by build_data.py — do not edit by hand.\n"
        "const GLOBAL_CONVERGENCE = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"  wrote {out_js} ({out_js.stat().st_size/1024:.0f} KB, {n_sample} countries)")

    # ---- 9. world geojson ----------------------------------------------------
    geo_path = DATA_DIR / "world-geo.js"
    if not geo_path.exists():
        print(f"  downloading world GeoJSON ...")
        geo = fetch_json(GEOJSON_URL)
        # slim: keep only what Leaflet needs
        for feat in geo.get("features", []):
            props = feat.get("properties", {})
            keep = {
                "name": props.get("name"),
                "iso_a3": props.get("iso_a3") or props.get("ISO_A3") or props.get("ADM0_A3"),
            }
            feat["properties"] = keep
            if feat.get("geometry"):
                coords = feat["geometry"].get("coordinates")
                if coords:
                    feat["geometry"]["coordinates"] = _round_coords(coords)
        geo_path.write_text(
            "// World GeoJSON (110m, slimmed) — source: github.com/johan/world.geo.json\n"
            "const WORLD_GEOJSON = " + json.dumps(geo, separators=(",", ":")) + ";\n",
            encoding="utf-8",
        )
        print(f"  wrote {geo_path} ({geo_path.stat().st_size/1024:.0f} KB)")
    else:
        print(f"  {geo_path} exists — skipping download")

    # ---- 10. sanity -----------------------------------------------------------
    checks = {
        "USA in panel": "USA" in countries,
        "CHN in panel": "CHN" in countries,
        "NGA in panel": "NGA" in countries,
        "beta exists": beta is not None,
        "sigma 2023 exists": sigma_2023 is not None,
    }
    for label, ok in checks.items():
        print(f"  {'✓' if ok else '✗'} {label}")
    if not all(checks.values()):
        sys.exit(1)
    print("◆ done.")


def _round_coords(obj, ndigits=3):
    if isinstance(obj, list):
        return [_round_coords(v, ndigits) for v in obj]
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


if __name__ == "__main__":
    main()
