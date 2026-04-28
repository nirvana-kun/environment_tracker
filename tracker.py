"""
Environment & ESG Data Tracker
--------------------------------
Pulls free environmental and economic data, computes correlations,
and renders terminal heatmaps and charts.

All data sources are 100% free:
  - World Bank Open Data API (no key needed)
  - Our World in Data (CSV downloads)
  - Global Forest Watch (public API)

Indicators tracked:
  - Forest cover % and change
  - CO2 emissions per capita
  - Renewable energy % of total
  - GDP per capita
  - Population
  - Electric power consumption
  - Access to clean fuels

Requirements:
    pip install pandas numpy matplotlib requests

Usage:
    python tracker.py                  # Full dashboard
    python tracker.py --chart forest   # Just forest cover
    python tracker.py --chart co2      # Just CO2 trends
    python tracker.py --chart corr     # Just correlation heatmap
    python tracker.py --countries "USA,CHN,IND,DEU,BRA,NGA"
"""

import argparse
import json
import time
import urllib.request
import urllib.parse
from typing import Optional

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
except ImportError:
    raise ImportError("Run: pip install pandas numpy matplotlib")


# ── World Bank indicator codes ─────────────────────────────────────────────────

WB_INDICATORS = {
    "forest_pct":       ("AG.LND.FRST.ZS",  "Forest area (% of land)"),
    "co2_per_capita":   ("EN.ATM.CO2E.PC",   "CO₂ emissions (t per capita)"),
    "renewable_pct":    ("EG.FEC.RNEW.ZS",   "Renewable energy (% of total)"),
    "gdp_per_capita":   ("NY.GDP.PCAP.CD",   "GDP per capita (USD)"),
    "population":       ("SP.POP.TOTL",       "Population"),
    "electricity_kwh":  ("EG.USE.ELEC.KH.PC","Electric power (kWh per capita)"),
    "clean_fuels_pct":  ("EG.CFT.ACCS.ZS",   "Access to clean fuels (%)"),
    "methane_pc":       ("EN.ATM.METH.PC",    "Methane emissions (t CO₂e per capita)"),
}

DEFAULT_COUNTRIES = ["USA", "CHN", "IND", "DEU", "BRA", "NGA", "FRA", "GBR", "JPN", "AUS"]
WB_API = "https://api.worldbank.org/v2"


# ── Data fetcher ───────────────────────────────────────────────────────────────

class EnvDataFetcher:

    def _wb_fetch(self, indicator: str, countries: list[str], years: int = 20) -> pd.DataFrame:
        """Fetch a World Bank indicator for multiple countries."""
        country_str = ";".join(countries)
        url = (
            f"{WB_API}/country/{country_str}/indicator/{indicator}"
            f"?format=json&per_page=500&mrv={years}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "env-tracker/1.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read().decode())

            if len(data) < 2 or not data[1]:
                return pd.DataFrame()

            rows = []
            for entry in data[1]:
                if entry.get("value") is not None:
                    rows.append({
                        "country": entry["country"]["value"],
                        "iso":     entry["countryiso3code"],
                        "year":    int(entry["date"]),
                        "value":   float(entry["value"]),
                    })

            df = pd.DataFrame(rows)
            return df

        except Exception as e:
            print(f"  Warning: could not fetch {indicator}: {e}")
            return pd.DataFrame()

    def fetch_all(
        self,
        countries: list[str] = DEFAULT_COUNTRIES,
        years: int = 20,
    ) -> dict[str, pd.DataFrame]:
        """Fetch all indicators. Returns dict of {indicator_name: DataFrame}."""
        data = {}
        for name, (code, label) in WB_INDICATORS.items():
            print(f"  Fetching: {label}...")
            df = self._wb_fetch(code, countries, years)
            if not df.empty:
                data[name] = df
            time.sleep(0.3)   # Be polite to the API
        return data

    def pivot_latest(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build a country × indicator table using most recent available year per series."""
        frames = []
        for name, df in data.items():
            if df.empty:
                continue
            # Most recent value per country
            latest = df.sort_values("year", ascending=False).groupby("iso").first().reset_index()
            latest = latest.rename(columns={"value": name})[["iso", "country", name]]
            frames.append(latest.set_index("iso"))

        if not frames:
            return pd.DataFrame()

        result = frames[0]
        for f in frames[1:]:
            result = result.join(f.drop(columns=["country"], errors="ignore"), how="outer")
        return result.reset_index()

    def pivot_timeseries(self, data: dict[str, pd.DataFrame], indicator: str) -> pd.DataFrame:
        """Wide table: year × country for a single indicator."""
        if indicator not in data or data[indicator].empty:
            return pd.DataFrame()
        df = data[indicator]
        return df.pivot_table(index="year", columns="iso", values="value")


# ── Visualisations ─────────────────────────────────────────────────────────────

class EnvVisualiser:

    GREENS  = ["#1a3a1a", "#2d6a2d", "#4caf50", "#81c784", "#c8e6c9"]
    REDS    = ["#b71c1c", "#e53935", "#ef9a9a", "#ffcdd2"]
    DIVERG  = "RdYlGn"

    def _label(self, key: str) -> str:
        return WB_INDICATORS.get(key, (key, key))[1]

    # ── Correlation heatmap ────────────────────────────────────────────────────

    def plot_correlation(self, latest: pd.DataFrame, save: Optional[str] = None) -> None:
        numeric = latest.select_dtypes(include=[float, int]).dropna(axis=1, how="all")
        corr    = numeric.corr()

        labels  = [self._label(c) for c in corr.columns]

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr.values, cmap=self.DIVERG, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        ax.set_title("Environment & Economy — Correlation Matrix\n(latest available year per indicator)",
                     fontsize=12, pad=15)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()

    # ── Forest cover heatmap ───────────────────────────────────────────────────

    def plot_forest(self, ts: pd.DataFrame, save: Optional[str] = None) -> None:
        if ts.empty:
            print("No forest data available.")
            return

        ts_clean = ts.dropna(axis=1, thresh=5).sort_index()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Top: line chart
        ax = axes[0]
        for col in ts_clean.columns:
            ax.plot(ts_clean.index, ts_clean[col], marker="o", markersize=2, label=col)
        ax.set_title("Forest Area (% of land area) over time", fontsize=12)
        ax.set_ylabel("% of land area")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)

        # Bottom: heatmap (country × year)
        ax2 = axes[1]
        data_T = ts_clean.T   # countries as rows
        im = ax2.imshow(data_T.values, cmap="Greens", aspect="auto",
                        vmin=0, vmax=data_T.values.max())
        plt.colorbar(im, ax=ax2, label="% forest cover")
        ax2.set_xticks(range(len(ts_clean.index)))
        ax2.set_xticklabels(ts_clean.index, rotation=45, ha="right", fontsize=7)
        ax2.set_yticks(range(len(data_T.index)))
        ax2.set_yticklabels(data_T.index, fontsize=8)
        ax2.set_title("Forest Cover Heatmap (darker = more forest)", fontsize=11)

        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()

    # ── CO2 chart ─────────────────────────────────────────────────────────────

    def plot_co2(self, ts: pd.DataFrame, save: Optional[str] = None) -> None:
        if ts.empty:
            print("No CO2 data available.")
            return

        ts_clean = ts.dropna(axis=1, thresh=5).sort_index()
        fig, ax  = plt.subplots(figsize=(13, 6))

        for col in ts_clean.columns:
            ax.plot(ts_clean.index, ts_clean[col], marker="o", markersize=3, label=col)

        ax.set_title("CO₂ Emissions per Capita (tonnes) over time", fontsize=12)
        ax.set_ylabel("Tonnes CO₂ per capita")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()

    # ── GDP vs ESG scatter ─────────────────────────────────────────────────────

    def plot_gdp_vs_env(self, latest: pd.DataFrame, save: Optional[str] = None) -> None:
        needed = ["gdp_per_capita", "co2_per_capita", "renewable_pct", "forest_pct", "country"]
        df = latest[[c for c in needed if c in latest.columns]].dropna(subset=["gdp_per_capita"])

        if df.empty:
            print("Insufficient data for GDP vs Environment scatter.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle("GDP per Capita vs Environmental Indicators", fontsize=13)

        pairs = [
            ("co2_per_capita",  "CO₂ per Capita (t)", "Reds"),
            ("renewable_pct",   "Renewable Energy (%)", "Greens"),
            ("forest_pct",      "Forest Cover (%)",     "YlGn"),
        ]

        for ax, (col, ylabel, cmap) in zip(axes, pairs):
            if col not in df.columns:
                ax.set_visible(False)
                continue
            sub = df[["gdp_per_capita", col, "country"]].dropna()
            sc = ax.scatter(
                sub["gdp_per_capita"], sub[col],
                c=sub[col], cmap=cmap, s=80, alpha=0.8, edgecolors="gray", linewidths=0.4,
            )
            for _, row in sub.iterrows():
                ax.annotate(row["country"][:3], (row["gdp_per_capita"], row[col]),
                            fontsize=6, alpha=0.7)
            ax.set_xlabel("GDP per Capita (USD)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"GDP vs {ylabel}", fontsize=10)
            ax.grid(alpha=0.3)
            plt.colorbar(sc, ax=ax)

        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()

    # ── Full dashboard ─────────────────────────────────────────────────────────

    def terminal_summary(self, latest: pd.DataFrame) -> None:
        """Print a quick text table to terminal."""
        cols = ["country", "gdp_per_capita", "co2_per_capita", "forest_pct", "renewable_pct"]
        available = [c for c in cols if c in latest.columns]
        df = latest[available].dropna(subset=["gdp_per_capita"]).sort_values(
            "gdp_per_capita", ascending=False
        )

        print(f"\n{'═'*78}")
        print(f"  ENVIRONMENT & ECONOMY SNAPSHOT  (World Bank latest available data)")
        print(f"{'═'*78}")
        header = f"  {'Country':<22} {'GDP/cap':>10} {'CO₂/cap':>10} {'Forest%':>9} {'Renew%':>9}"
        print(header)
        print(f"  {'-'*70}")

        for _, row in df.iterrows():
            gdp  = f"${row['gdp_per_capita']:>9,.0f}" if "gdp_per_capita" in row else "N/A"
            co2  = f"{row['co2_per_capita']:>8.1f}t"  if "co2_per_capita"  in row else "N/A"
            frst = f"{row['forest_pct']:>8.1f}%"      if "forest_pct"      in row else "N/A"
            renw = f"{row['renewable_pct']:>8.1f}%"   if "renewable_pct"   in row else "N/A"
            name = str(row.get("country", ""))[:22]
            print(f"  {name:<22} {gdp:>10} {co2:>10} {frst:>9} {renw:>9}")

        print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Environment & ESG Data Tracker")
    parser.add_argument("--chart",     choices=["forest", "co2", "corr", "gdp", "all"],
                        default="all", help="Which chart to show")
    parser.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES),
                        help="Comma-separated ISO3 country codes")
    parser.add_argument("--years",     type=int, default=20, help="Years of history")
    parser.add_argument("--save",      action="store_true", help="Save charts as PNG files")
    args = parser.parse_args()

    countries = [c.strip().upper() for c in args.countries.split(",")]

    fetcher = EnvDataFetcher()
    vis     = EnvVisualiser()

    print(f"Fetching data for: {', '.join(countries)}")
    data = fetcher.fetch_all(countries, years=args.years)

    latest = fetcher.pivot_latest(data)
    vis.terminal_summary(latest)

    chart = args.chart
    save  = args.save

    if chart in ("forest", "all"):
        ts = fetcher.pivot_timeseries(data, "forest_pct")
        vis.plot_forest(ts, save="forest.png" if save else None)

    if chart in ("co2", "all"):
        ts = fetcher.pivot_timeseries(data, "co2_per_capita")
        vis.plot_co2(ts, save="co2.png" if save else None)

    if chart in ("corr", "all"):
        vis.plot_correlation(latest, save="correlation.png" if save else None)

    if chart in ("gdp", "all"):
        vis.plot_gdp_vs_env(latest, save="gdp_env.png" if save else None)


if __name__ == "__main__":
    main()
