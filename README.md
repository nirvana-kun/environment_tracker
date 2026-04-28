# Environment & ESG Data Tracker

Pulls free environmental and economic data from the World Bank Open Data API and generates correlation heatmaps, trend charts, and GDP vs environment scatter plots.

## Setup

```bash
pip install pandas numpy matplotlib
```

No API keys needed — World Bank data is fully public.

## Usage

```bash
python tracker.py                            # Full dashboard (all charts)
python tracker.py --chart forest             # Forest cover only
python tracker.py --chart co2               # CO2 emissions only
python tracker.py --chart corr              # Correlation heatmap only
python tracker.py --chart gdp               # GDP vs environment scatter
python tracker.py --save                    # Save charts as PNG files
python tracker.py --countries "USA,CHN,IND,DEU,BRA,NGA,FRA"
python tracker.py --years 30               # 30 years of history
```

## Indicators tracked (all from World Bank, free)

| Indicator | Code |
|-----------|------|
| Forest area (% of land) | AG.LND.FRST.ZS |
| CO₂ emissions per capita | EN.ATM.CO2E.PC |
| Renewable energy % | EG.FEC.RNEW.ZS |
| GDP per capita | NY.GDP.PCAP.CD |
| Electric power (kWh/capita) | EG.USE.ELEC.KH.PC |
| Access to clean fuels % | EG.CFT.ACCS.ZS |
| Methane emissions per capita | EN.ATM.METH.PC |

## Charts produced

- **Normalised price chart** — all countries rebased to 100
- **Rolling volatility** — 30-day annualised vol per country
- **Drawdown chart** — peak-to-trough declines
- **Correlation heatmap** — economy vs environment cross-correlations
- **GDP vs environment scatter** — CO2, renewables, and forest vs wealth
