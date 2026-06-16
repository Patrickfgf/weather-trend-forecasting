# Weather Trend Forecasting — Global Weather Repository

> 🇬🇧 English version: [`README.md`](README.md)

Previsão de **temperatura** diária de cidades do mundo a partir do dataset do Kaggle
[Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository),
com um pipeline de modelagem *leakage-safe*, detecção de anomalias avançada e cinco
análises únicas (clima, qualidade do ar, importância de features, espacial e geográfica).

> ### PM Accelerator — Missão
> O Product Manager Accelerator (PMA), liderado pela Dra. Nancy Li, é uma comunidade
> de desenvolvimento profissional em product management. Sua missão é apoiar profissionais
> de PM em todas as fases da carreira — de aspirantes e iniciantes a líderes seniores rumo
> a posições de Diretoria e executivas — por meio de treinamento prático e real (inclusive
> construir e lançar produtos de IA reais), coaching de carreira e uma comunidade baseada
> numa cultura de compartilhar e elevar os outros. Pela iniciativa sem fins lucrativos
> PMA Kids, o PMA também oferece educação gratuita de product management a adolescentes de
> famílias de baixa renda, derrubando barreiras financeiras e promovendo equidade educacional.
> _Fonte: <https://www.pmaccelerator.io/>_

---

## O que este projeto faz

- **Alvo / grão:** prever `temperature_celsius`; 1 linha = 1 snapshot diário de uma cidade,
  com chave `(country, location_name)` no instante `last_updated`.
- **Dois tracks de forecasting, um backtest leakage-safe:**
  - **Track A** — modelos clássicos por cidade (seasonal-naive, drift, climatology,
    Holt-Winters **ETS**, **SARIMAX**) sobre um conjunto de cidades representativas
    diversas em clima.
  - **Track B** — um modelo **global em painel** de gradient boosting (XGBoost; LightGBM
    se disponível) que agrega sinal entre todas as cidades, mais um **ensemble por erro
    inverso** e **stacking OOF**.
- **EDA avançada + detecção de anomalias:** z-scores robustos por (cidade, mês), resíduos
  STL, IsolationForest + LocalOutlierFactor, com análise das anomalias (condition lift,
  detecção de artefato de ingestão).
- **Cinco análises únicas:** padrões climáticos within-sample por zona climática,
  correlações qualidade-do-ar ↔ clima, importância de features (permutation / native / SHAP),
  e geografia espacial + cross-país/continente.

## Início rápido

```bash
# 1) Python 3.11+ (desenvolvido e verificado em 3.14)
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate no macOS/Linux
pip install -r requirements.txt
pip install -r requirements-optional.txt   # aceleradores opcionais; degradam graciosamente se pulados
pip install -e .                  # faz `import weather_forecast` funcionar em qualquer lugar

# 2) (opcional) obter os dados reais — senão um fallback sintético roda automaticamente
#    Baixe "GlobalWeatherRepository.csv" do Kaggle e coloque em:
#    data/raw/GlobalWeatherRepository.csv

# 3) rodar o pipeline inteiro (produz todos os artefatos em data/ e reports/)
python run_pipeline.py

# 4) abrir a narrativa
jupyter lab notebooks/01_weather_trend_forecasting.ipynb

# 5) testes
pytest -q
```

Se nenhum CSV estiver em `data/raw/`, o pipeline loga `SYNTHETIC fallback` e roda sobre um
dataset sintético determinístico e fiel ao schema, para que um revisor execute tudo sem
nenhum download manual. Coloque o CSV real do Kaggle e re-rode para resultados reais.

## Estrutura do projeto

```
weather-trend-forecasting/
├── run_pipeline.py              # entrypoint único -> todos os artefatos
├── requirements.txt             # stack core pinada
├── requirements-optional.txt    # extras opcionais (degradam graciosamente)
├── pyproject.toml               # editable install + config pytest/ruff
├── data/{raw,interim,processed} # raw (gitignored) -> parquet intermediários/saídas
├── reports/{figures,metrics}    # figuras PNG/HTML + métricas CSV/JSON
├── notebooks/                   # notebook narrativo (importa de src/)
├── tests/                       # suíte pytest (fixtures sintéticas, testes de leakage)
├── docs/TECHNICAL_SPEC.md       # metodologia & decisões de design
├── project-brief.html           # brief técnico (para CV / portfólio)
└── src/weather_forecast/
    ├── config.py                # single source of truth (colunas, grão, seeds, missão)
    ├── ingest.py                # borda de I/O: carga + normalização de schema + validação + fallback
    ├── synthetic.py             # gerador sintético fiel ao GWR
    ├── cleaning.py              # dedup, gating físico, imputação per-city, flags de outlier
    ├── engineering.py           # calendário/cíclicas + lags/rollings leakage-safe
    ├── profiling.py             # EDA defensiva + seleção de cidades representativas
    ├── plots.py                 # figuras matplotlib (EDA/anomalia/forecast)
    ├── anomaly.py               # detecção de anomalia univariada + multivariada & análise
    ├── metrics.py               # MAE/RMSE/sMAPE/MAPE/MASE + baselines
    ├── backtest.py              # CV rolling-origin leakage-safe (per-city + global)
    ├── models.py                # ETS/SARIMAX + XGBoost/LightGBM/Prophet
    ├── ensemble.py              # pesos por erro inverso + stacking OOF
    ├── regions.py               # país->continente + zonas climáticas por latitude
    ├── climate.py | air_quality.py | feature_importance.py | spatial.py  # análises únicas
    └── pipeline.py              # orquestração
```

## Como evita data leakage (a história central de correção)

1. **Backtest só cronológico** — CV rolling-origin / expanding-window dividindo numa data
   de calendário global, aplicada identicamente a cada cidade. Sem `KFold`, sem shuffle.
   Todo fold assegura `train.max_day < test.min_day`.
2. **Features causais** — lags usam `groupby(city).shift(k)`; estatísticas rolling usam
   `shift(1)` *antes* do `rolling`, então o valor da própria linha nunca entra na janela.
3. **Clima exógeno é defasado, nunca contemporâneo** — você não conhece a umidade de amanhã
   ao prever a temperatura de amanhã.
4. **Alvo & gêmeos vazantes excluídos** — `temperature_fahrenheit` (o alvo disfarçado) e
   `feels_like_*` são removidos das features e da importância.
5. **Steps com estado ajustam só no treino** — scalers / stackers veem apenas dados out-of-fold.
6. **Multi-step direto** — o modelo global prevê cada horizonte *a partir da origem*
   (label deslocado h dias; features só até o corte), nunca lendo o real intra-janela —
   um verdadeiro forecaster h-step, justamente comparável aos baselines.

Tudo isso é enforçado por `assert`s no código e por testes dedicados em `tests/test_features.py`.

## Reprodutibilidade

- Seeds fixas (`config.SEED = 42`), `xgboost` determinístico (`tree_method="hist"`).
- Intermediários em Parquet (tipados, preservam timezone). Idempotente: mesmo input → mesmo output.
- Sem paths absolutos; tudo ancorado na raiz do projeto.

## Snapshot de resultados (execução real sobre os dados do Kaggle)

Macro-média sobre (cidade × fold), cidades representativas, ordenado por MASE:

| modelo | MASE ↓ | MAE (°C) | nota |
|---|---|---|---|
| **sarimax** | **0.786** | 2.63 | melhor clássico (per-city) |
| xgb_global | 0.794 | 2.67 | empate técnico com o SARIMAX |
| lgbm_global | 0.800 | 2.67 | painel global |
| naive | 0.884 | 2.96 | persistência — a barra real a bater |
| seasonal_naive | 1.115 | 3.67 | falha out-of-sample (> 1) |
| climatology | 1.527 | 4.79 | fraco em histórico sub-anual |

MASE < 1 significa "bate o seasonal-naive". Nas cidades representativas é um **empate
técnico** entre o SARIMAX clássico e o modelo ML global; no **painel completo de 268
cidades** o global se descola (XGBoost MASE **0.734**) e é a escolha de produção — um único
modelo que generaliza para cidades novas sem refit. Um achado real interessante: o
seasonal-naive **falha** out-of-sample (o ciclo semanal não se sustenta), então a
persistência simples é a barra a bater — confirmando que isto é um problema de persistência
(`temperature_celsius_lag1` domina a importância de features). Narrativa completa e figuras:
[`REPORT.md`](REPORT.md). Os números regeneram a partir do CSV real quando presente.

> **Brief técnico para CV/portfólio:** veja [`project-brief.html`](project-brief.html) — um
> resumo detalhado de decisões, stack e competências demonstradas.

## Licença & contato

Para avaliação: mantenha o repositório **público** durante o review, ou conceda acesso a
`community@pmaccelerator.io` e `hr@pmaccelerator.io`.
