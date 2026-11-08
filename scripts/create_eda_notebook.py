"""Create the reproducible EDA notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_eda.ipynb"


def main() -> None:
    """Write the notebook with its sections, calculations, and figures."""
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Step 4 — Exploratory analysis\n"
            "All conclusions below are calculated from the project files. "
            "The partial day 2011-12-09 is excluded."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import statsmodels\n"
            "import matplotlib\n"
            "print('sys.executable:', sys.executable)\n"
            "print('sys.version:', sys.version)\n"
            "print('pandas:', pd.__version__)\n"
            "print('numpy:', np.__version__)\n"
            "print('statsmodels:', statsmodels.__version__)\n"
            "print('matplotlib:', matplotlib.__version__)"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from statsmodels.tsa.seasonal import seasonal_decompose\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from retail_forecast.data.eda import (eda_summary, weekday_profile, "
            "decompose_weekly, anomaly_details, cleaning_impact, cleaning_impact_steps)\n"
            "from retail_forecast.data.load import load_config\n"
            "config = load_config(ROOT / 'configs' / 'config.yaml')\n"
            "clean = pd.read_csv(ROOT / 'data/interim/clean_transactions.csv', "
            "dtype={'InvoiceNo':'string','StockCode':'string','CustomerID':'string'}, "
            "parse_dates=['InvoiceDate'])\n"
            "raw = pd.read_csv(ROOT / 'data/raw/online_retail.csv', "
            "dtype={'InvoiceNo':'string','StockCode':'string','CustomerID':'string'}, "
            "parse_dates=['InvoiceDate'])\n"
            "daily = pd.read_csv(ROOT / 'data/processed/daily_revenue.csv', parse_dates=['date'])\n"
            "weekly = pd.read_csv(ROOT / 'data/processed/weekly_revenue.csv', parse_dates=['date'])\n"
            "anomalous = pd.read_csv(ROOT / 'data/processed/anomalous_days.csv', parse_dates=['date'])\n"
            "transaction_counts = clean.assign(date=clean.InvoiceDate.dt.normalize()).groupby('date').size()\n"
            "daily['n_transactions'] = daily.date.map(transaction_counts).fillna(0).astype(int)\n"
            "summary = eda_summary(clean, daily)\n"
            "profile = weekday_profile(daily)\n"
            "impact = cleaning_impact(raw, clean)\n"
            "cleaning_steps = cleaning_impact_steps(raw, pd.read_csv(ROOT / 'data/interim/cleaning_summary.csv'), config)\n"
            "details = anomaly_details(clean, anomalous)\n"
            "figures = ROOT / 'reports/figures'; figures.mkdir(parents=True, exist_ok=True)\n"
            "sns.set_theme(style='whitegrid')"
        ),
        nbf.v4.new_markdown_cell("## a. Context and cleaning pipeline"),
        nbf.v4.new_code_cell(
            "cleaning_summary = pd.read_csv(ROOT / 'data/interim/cleaning_summary.csv')\n"
            "cleaning_summary.loc[cleaning_summary['step'].eq('winsorize'), 'note'] = '0 caps'\n"
            "cleaning_summary.loc[cleaning_summary['step'].eq('add_revenue'), 'note'] = '0 rows removed'\n"
            "display(cleaning_summary)\n"
            "display(impact)\n"
            "display(cleaning_steps)\n"
            'print(f"Cleaning retains {len(clean):,} rows out of {len(raw):,}, or {len(clean)/len(raw):.1%}; cleaned revenue is £{clean.Revenue.sum():,.2f}.")'
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — cleaning\n\n"
            "Cleaning retains **519,726 rows out of 541,909**, or **95.9%**, "
            "for calculated cleaned revenue of **£9,821,479.21**. Cancellations and "
            "non-product codes are therefore handled before modeling."
        ),
        nbf.v4.new_markdown_cell("## b. Daily and weekly revenue"),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(daily.date, daily.revenue)\n"
            "ax.fill_between(daily.date, 0, daily.revenue, where=~daily.is_trading_day, alpha=.25, label='closed')\n"
            "ax.set(title='Daily revenue', ylabel='£'); ax.legend(); fig.tight_layout(); fig.savefig(figures/'daily_revenue.png'); plt.show()\n"
            "fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(weekly.date, weekly.revenue)\n"
            "partial = weekly.loc[weekly.is_partial_week]; ax.plot(partial.date, partial.revenue, linestyle='None', marker='o', label='partial week')\n"
            "ax.set(title='Weekly revenue', ylabel='£'); ax.legend(); fig.tight_layout(); fig.savefig(figures/'weekly_revenue.png'); plt.show()\n"
            "max_day = daily.loc[daily.revenue.idxmax()]\n"
            "top5_days = daily.nlargest(5, 'revenue')[['date', 'revenue']]\n"
            "active_eda = daily.loc[daily.is_trading_day & ~daily.date.between('2010-12-24', '2011-01-03')].copy()\n"
            "active_eda['month'] = active_eda.date.dt.to_period('M').astype(str)\n"
            "complete_months = active_eda.loc[active_eda.month.between('2011-01', '2011-11')].copy()\n"
            "monthly_stats = complete_months.groupby('month').agg(total=('revenue', 'sum'), mean=('revenue', 'mean'), n_days=('date', 'size'))\n"
            "nov_stats = monthly_stats.loc['2011-11']\n"
            "nov_total_rank = int(monthly_stats.total.rank(ascending=False, method='min').loc['2011-11'])\n"
            "nov_mean_rank = int(monthly_stats['mean'].rank(ascending=False, method='min').loc['2011-11'])\n"
            "nov_total_ratio = nov_stats.total / monthly_stats.total.median()\n"
            "nov_mean_ratio = nov_stats['mean'] / monthly_stats['mean'].median()\n"
            "jan_mean = monthly_stats.loc['2011-01', 'mean']; oct_mean = monthly_stats.loc['2011-10', 'mean']\n"
            "jan_oct = complete_months.loc[complete_months.month.between('2011-01', '2011-10')]\n"
            "trend_slope = float(__import__('numpy').polyfit(range(len(jan_oct)), jan_oct.revenue.to_numpy(float), 1)[0])\n"
            "print(f\"Series: {daily.date.min().date()} to {daily.date.max().date()}, {len(daily)} days.\")\n"
            "print(f\"IDXMAX: date={max_day.date.date()}, revenue=£{max_day.revenue:,.2f}\")\n"
            "print('TOP5_DAYS:'); print(top5_days.to_string(index=False))\n"
            "print('MONTHLY_ACTIVE_MEANS_COMPLETE='); print(monthly_stats[['mean', 'n_days']].to_string())\n"
            "print(f\"JAN_OCT_TREND: January=£{jan_mean:,.2f}, October=£{oct_mean:,.2f}, change=£{oct_mean-jan_mean:,.2f} ({oct_mean/jan_mean-1:.2%}); linear January-October slope=£{trend_slope:,.2f}/day. With one annual cycle, the underlying trend cannot be identified reliably.\")"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — trend\n\n"
            "The maximum calculated by `daily.revenue.idxmax()` is **£110,815.68 on "
            "2011-11-14**. Across complete months from January to November 2011, "
            "the active-day average rises from **£23,530.54 in January** to "
            "**£41,190.79 in October**, or **+75.05%** excluding November; the "
            "January-October slope is **£73.63/day**. With one annual cycle, "
            "the underlying trend cannot be identified reliably."
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — observed closure\n\n"
            "The window from **December 24, 2010 to January 3, 2011** contains "
            "**0 transactions** and **£0.00** in revenue. For Christmas 2011, "
            "this closure is a displayed calendar assumption because only one Christmas is observed."
        ),
        nbf.v4.new_code_cell(
            "closure_window = daily.loc[daily.date.between('2010-12-15', '2011-01-10')]\n"
            "fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(closure_window.date, closure_window.revenue, marker='o')\n"
            "ax.axvspan(pd.Timestamp('2010-12-24'), pd.Timestamp('2011-01-03'), alpha=.2, color='red', label='closure Dec 24-Jan 3')\n"
            "ax.set(title='Christmas closure window', ylabel='£'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures/'christmas_closure_window.png'); plt.show()\n"
            "print(f\"Window 2010-12-15 -> 2011-01-10: {len(closure_window)} days, {int(closure_window.n_transactions.sum())} transactions during closure, £{closure_window.loc[closure_window.date.between('2010-12-24','2011-01-03'),'revenue'].sum():,.2f} during closure.\")"
        ),
        nbf.v4.new_markdown_cell("## c. Weekday profile"),
        nbf.v4.new_code_cell(
            "display(profile)\n"
            "fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(profile, x='weekday', y='mean', ax=ax)\n"
            "ax.set(title='Average active-day revenue', ylabel='£'); fig.tight_layout(); fig.savefig(figures/'weekday_profile.png'); plt.show()\n"
            'print(f"Saturday accounts for {int((daily.date.dt.dayofweek==5).sum())} days in the series and {int(daily.loc[daily.date.dt.dayofweek==5, "revenue"].sum()):,.2f} £ in revenue.")'
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — weekly seasonality\n\n"
            "The calculated profile covers **6 active days**: Monday to Friday and Sunday. "
            "Saturday is structurally closed in the series and should be modeled with "
            "the `is_trading_day` feature, not as a random absence."
        ),
        nbf.v4.new_markdown_cell("## d. Monthly seasonality"),
        nbf.v4.new_code_cell(
            "monthly = summary.query(\"dimension == 'month'\")\n"
            "fig, ax = plt.subplots(figsize=(10, 4)); sns.barplot(monthly, x='key', y='revenue', ax=ax); ax.tick_params(axis='x', rotation=60)\n"
            "ax.set(title='Monthly revenue', ylabel='£'); fig.tight_layout(); fig.savefig(figures/'monthly_revenue.png'); plt.show()\n"
            "nov = monthly.loc[monthly.key.eq('2011-11'), 'revenue'].iloc[0]\n"
            "print(f\"NOVEMBER_TOTAL: £{nov:,.2f}; rank={nov_total_rank}/{len(monthly_stats)}, median_month_ratio={nov_total_ratio:.2f}x.\")\n"
            "print(f\"NOVEMBER_ACTIVE_DAILY_MEAN: £{nov_stats['mean']:,.2f}; rank={nov_mean_rank}/{len(monthly_stats)}, median_monthly_mean_ratio={nov_mean_ratio:.2f}x; active days={int(nov_stats['n_days'])}.\")\n"
            "observed_monthly = active_eda.groupby('month').agg(n_days=('date', 'size'))\n"
            "print(f\"DECEMBER_2010: December 2010 is affected by the closure from the 24th to the 31st, with {int(observed_monthly.loc['2010-12', 'n_days'])} observed active days; excluded from ranking.\")\n"
            "print(f\"DECEMBER_PARTIAL: December 2011 is partial through {daily.date.max().date()}, with {int(observed_monthly.loc['2011-12','n_days'])} observed active days; excluded from ranking.\")\n"
            "print(f\"CLOSURE: {int(daily.loc[daily.date.between('2010-12-24','2011-01-03'),'n_transactions'].sum())} transactions and £{daily.loc[daily.date.between('2010-12-24','2011-01-03'),'revenue'].sum():,.2f} from December 24 to January 3.\")"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — monthly seasonality\n\n"
            "Across the **11 complete months from January to November 2011**, November "
            "ranks first in total revenue (**£1,438,851.68**, **2.09x** the median) "
            "and first in active-day average (**£55,340.45**, **2.09x**). "
            "December 2010 is affected by the closure from the 24th to the 31st, and "
            "December 2011 is partial through the 8th; both months are excluded from ranking."
        ),
        nbf.v4.new_markdown_cell("## e. Weekly decomposition"),
        nbf.v4.new_code_cell(
            "decomp = decompose_weekly(daily)\n"
            "fig = decomp.plot(); fig.set_size_inches(12, 8); fig.tight_layout(); fig.savefig(figures/'decomposition.png'); plt.show()\n"
            "print(f\"DECOMPOSITION_DAYS: {len(decomp.observed)} days after excluding closed days and the December 24-January 3 window.\")\n"
            "print('Method: closed days and the December 24-January 3 window removed; "
            "seasonality = average by weekday and residual = deviation after a rolling trend of 7 active observations.')"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — decomposition\n\n"
            "The decomposition uses a **7-day** weekly period and covers exactly "
            "**304 active days** after excluding closed days and the December 24 to "
            "January 3 window. Only the weekly component is robustly interpretable; "
            "annual seasonality is limited by the single observed Christmas."
        ),
        nbf.v4.new_markdown_cell("## f. Anomalous days"),
        nbf.v4.new_code_cell(
            "display(details)\n"
            "fig, ax = plt.subplots(figsize=(10, 4)); ax.scatter(anomalous.date, anomalous.revenue); ax.tick_params(axis='x', rotation=60)\n"
            "ax.set(title='Detected anomalous days', ylabel='£'); fig.tight_layout(); fig.savefig(figures/'anomalies.png'); plt.show()\n"
            "for date, group in details.groupby('date'):\n"
            '    print(f"{date.date()}: top 3 = £{group.Revenue.sum():,.2f}, share = {group.share_of_day.sum():.1%}; conclusion: inspect the three largest lines before classifying the day as a true peak.")'
            "\n"
            "day_0920 = clean.loc[clean.InvoiceDate.dt.normalize().eq(pd.Timestamp('2011-09-20'))].copy()\n"
            "customer_0920 = day_0920.groupby('CustomerID', dropna=False)['Revenue'].sum().sort_values(ascending=False).head(5).rename('revenue').reset_index()\n"
            "customer_0920['customer_key'] = customer_0920.CustomerID.astype('string').str.replace(r'\\.0$', '', regex=True)\n"
            "customer_0920['share_of_day'] = customer_0920.revenue / day_0920.Revenue.sum()\n"
            "display(customer_0920)\n"
            "share_17450 = customer_0920.loc[customer_0920.customer_key.eq('17450'), 'share_of_day'].iloc[0]\n"
            "print(f\"CustomerID 17450 represents {share_17450:.2%} of 2011-09-20 revenue.\")"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — anomalies\n\n"
            "The **7 anomalous days** represent **£475,740.25**, or **4.86%** "
            "of total daily revenue. 2011-09-20 is included and should be flagged "
            "as an unpredictable event: customer **17450 represents 64.69%** "
            "of that day's revenue."
        ),
        nbf.v4.new_code_cell(
            "anomaly_dates = set(anomalous.date)\n"
            "with_anomalies = daily.revenue\n"
            "without_anomalies = daily.loc[~daily.date.isin(anomaly_dates), 'revenue']\n"
            "anomaly_effect = pd.DataFrame([\n"
            "    {'scope': 'avec_7_anomalies', 'mean': with_anomalies.mean(), 'median': with_anomalies.median(), 'std': with_anomalies.std(), 'share_total': with_anomalies.sum() / daily.revenue.sum()},\n"
            "    {'scope': 'sans_7_anomalies', 'mean': without_anomalies.mean(), 'median': without_anomalies.median(), 'std': without_anomalies.std(), 'share_total': without_anomalies.sum() / daily.revenue.sum()},\n"
            "])\n"
            "display(anomaly_effect)\n"
            "print('ANOMALY_DATES:', ', '.join(anomalous.date.dt.strftime('%Y-%m-%d')))\n"
            "fig, ax = plt.subplots(figsize=(10, 4)); ax.hist(with_anomalies, bins=30, alpha=.55, label='with anomalies'); ax.hist(without_anomalies, bins=30, alpha=.55, label='without anomalies'); ax.set(title='Effect of anomalies on daily distribution', xlabel='Revenue (£)', ylabel='Number of days'); ax.legend(); fig.tight_layout(); fig.savefig(figures/'anomaly_effect.png'); plt.show()\n"
            "print(f\"With anomalies: mean £{with_anomalies.mean():,.2f}, median £{with_anomalies.median():,.2f}, standard deviation £{with_anomalies.std():,.2f}, share {with_anomalies.sum()/daily.revenue.sum():.2%}.\")\n"
            "print(f\"Without anomalies: mean £{without_anomalies.mean():,.2f}, median £{without_anomalies.median():,.2f}, standard deviation £{without_anomalies.std():,.2f}, share {without_anomalies.sum()/daily.revenue.sum():.2%}.\")\n"
            "mean_drop = 1 - without_anomalies.mean()/with_anomalies.mean()\n"
            "std_drop = 1 - without_anomalies.std()/with_anomalies.std()\n"
            "print(f\"ANOMALY_RELATIVE_DROP: mean={mean_drop:.2%}, standard deviation={std_drop:.2%}; conclusion: anomalies explain little variance if the standard deviation decrease remains small.\")"
        ),
        nbf.v4.new_markdown_cell("## g. Pays et produits"),
        nbf.v4.new_code_cell(
            "country = summary.query(\"dimension == 'country'\").sort_values('revenue', ascending=False).head(15)\n"
            "products = summary.query(\"dimension == 'product'\").head(20)\n"
            "fig, ax = plt.subplots(figsize=(10, 5)); sns.barplot(country, y='key', x='revenue', ax=ax); fig.tight_layout(); fig.savefig(figures/'country.png'); plt.show()\n"
            "fig, ax = plt.subplots(figsize=(10, 7)); sns.barplot(products, y='key', x='revenue', ax=ax); fig.tight_layout(); fig.savefig(figures/'products.png'); plt.show()\n"
            "uk = country.loc[country.key.eq('United Kingdom'), 'revenue'].iloc[0]\n"
            'print(f"The United Kingdom represents {uk/clean.Revenue.sum():.1%} of cleaned revenue among summarized countries.")'
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — geographic concentration\n\n"
            "The United Kingdom represents **84.46%** of calculated cleaned revenue. "
            "Geographic concentration is therefore strong; other countries should "
            "be monitored as minority segments rather than mixed without control."
        ),
        nbf.v4.new_markdown_cell("## h. Summary and implications"),
        nbf.v4.new_code_cell(
            'print(f"1. {len(clean):,} cleaned transactions produce £{clean.Revenue.sum():,.2f}.")\n'
            'print(f"2. The series contains {len(daily)} days and {int((~daily.is_trading_day).sum())} inactive days.")\n'
            'print(f"3. The daily peak reaches £{daily.revenue.max():,.2f} on {daily.loc[daily.revenue.idxmax(), "date"].date()}.")\n'
            'print(f"4. November totals £{nov:,.2f}; only one Christmas is observed, so year-end seasonality is fragile.")\n'
            'print(f"5. The {len(anomalous)} anomalous days should be retained as candidate events, not automatically removed.")\n'
            'print(f"6. The series contains {int((daily.n_transactions==0).sum())} days without transactions; closed days must be handled explicitly in features.")\n'
            "print('Implications: the model must incorporate closures, the November peak, and uncertainty from a single Christmas cycle; the target remains total revenue.')"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — summary\n\n"
            "The calculated results combine **304 decomposed days**, **69 inactive days**, "
            "a daily maximum of **£110,815.68 on 2011-11-14**, a **75.05%** increase "
            "between January and October in active-day averages, **7 anomalies**, and "
            "**84.46%** UK concentration. With one annual cycle, the underlying trend "
            "remains uncertain; features should represent closures and events without "
            "removing observations."
        ),
    ]
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "name": "retail-forecast",
            "display_name": "Python 3.12 (retail-forecast)",
            "language": "python",
        }
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(nbf.writes(nb), encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
