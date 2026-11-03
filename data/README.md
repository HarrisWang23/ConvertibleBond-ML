## Data source and access

The 2017–2022 weekly convertible bond data used in this study were obtained from the China Stock Market & Accounting Research Database (CSMAR) under a student license provided by The Chinese University of Hong Kong, Shenzhen (CUHK-Shenzhen).

If you have your own valid CSMAR license or institutional access, open the [CSMAR China Bond Market data query portal](https://data.csmar.com/). After signing in, follow **Get Data → Data Query → China Bond Market Series → Bond Market**, then select the convertible-bond table and the fields and date range you need. CSMAR's query page requires you to choose a table after login, so this link opens the official query portal rather than a preselected table. The datasets available to you depend on your license.

## Predictors

We computed and used 56 predictors in total, classified as bond predictors and stock predictors. Some of them are shown here as an example.

### A. Convertible bond predictors 

| No. | Predictor | Brief definition |
|---:|---|---|
| 1 | `Size` | Convertible bond market capitalization at week-end |
| 2 | `Age` | Bond age in years |
| 3 | `W_ret1` | Convertible bond return over the past 1 week |
| 4 | `W_ret2` | Convertible bond return over the past 2 weeks |
| 5 | `W_ret16` | Convertible bond return over the past 16 weeks |
| 6 | `Price` | Raw convertible bond price |
| 7 | `W_ret4_1` | Convertible bond return over weeks 1–4 |
| 8 | `Vol` | Log average daily trading volume during the week |
| 9 | `Volscaled` | Log average daily volume × price, scaled by market capitalization |
| 10 | `Turnover` | Convertible bond turnover during the week |
| 11 | `Conversion_premium` | (Bond price − conversion value) / bond price |
| 12 | `IV` | Implied volatility of the convertible bond |


### B. Stock predictors

| No. | Predictor | Brief definition |
|---:|---|---|
| 1 | `Debt_EBITDA` | Long- and short-term debt divided by EBITDA |
| 2 | `W_stock_ret1` | Underlying stock return over the past 1 week |
| 3 | `W_stock_ret16` | Underlying stock return over the past 16 weeks |
| 4 | `W_stock_ret6_1` | Underlying stock return over weeks 1–6 |
| 5 | `EP` | Earnings per share divided by stock price |
| 6 | `Mktcap` | Underlying stock market capitalization |
| 7 | `Market_leverage` | (Market cap + debt + minority interest + preferred stock − cash − inventories) / market cap |
| 8 | `Turnover_vol` | Quarterly standard deviation of sales divided by assets |
