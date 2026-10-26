# Global Convergence Observatory

Interactive world dashboard for global economic convergence — β-convergence,
σ-convergence, catch-up half-life, convergence clubs, and H Heuristics'
Medium Coefficient (D-coefficient). Built with the Veles pipeline
(Fable-5 × DeepSeek).

## Data

- **GDP per capita** — `NY.GDP.PCAP.KD` (constant 2015 US$), 1990–2023
- **Population** — `SP.POP.TOTL`
- **Diffusion proxies** — `EG.ELC.ACCS.ZS` (electricity), `IT.NET.USER.ZS`
  (internet), `IT.CEL.SETS.P2` (mobile), `SE.ADT.LITR.ZS` (literacy),
  `SE.SEC.ENRR` (secondary enrollment), `SP.URB.TOTL.IN.ZS` (urbanization)

Source: World Bank World Development Indicators via `api.worldbank.org`.

## Convergence math

- **σ-convergence** — cross-country standard deviation of log GDP per capita,
  per year (panel of countries with data in that year, n ≥ 60).
- **β-convergence** — OLS of annualized growth 1995→2023 on log initial
  income. Speed `λ = −ln(1 + b·T)/T` with `b` as a fraction; half-life
  `ln(2)/λ`.
- **D-coefficient** — simplified Medium Coefficient (0–100), a weighted
  composite of infrastructure, human capital, information, and geography
  proxies (digital-era profile of the Veles D-Coefficient Engine,
  renormalised over observable components).
- **Convergence clubs** — 2×2 of initial income (above/below sample median)
  vs β-regression residual: *Catch-up engine*, *Lagging*,
  *Frontier outperformer*, *Frontier steady*.

## Files

```
index.html                       dashboard (self-contained UI)
data/global-convergence.js       computed panel + stats (generated)
data/world-geo.js                world GeoJSON, 110m (generated)
build_data.py                    pipeline: fetch → compute → write data files
```

## Rebuild data

```bash
python3 build_data.py
```

## Deploy

Static site — push to GitHub and enable GitHub Pages from the repo root
(Settings → Pages → Deploy from branch → `/ (root)`).

Built by H Heuristics · Veles pipeline (Fable-5 × DeepSeek)
