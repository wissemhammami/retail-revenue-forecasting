"""Crée le notebook EDA reproductible."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_eda.ipynb"


def main() -> None:
    """Écrit le notebook avec ses sections, calculs et figures."""
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Étape 4 — Analyse exploratoire\n"
            "Toutes les conclusions ci-dessous sont calculées à partir des fichiers "
            "du projet. Le 2011-12-09, jour partiel, est exclu."
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
        nbf.v4.new_markdown_cell("## a. Contexte et chaîne de nettoyage"),
        nbf.v4.new_code_cell(
            "cleaning_summary = pd.read_csv(ROOT / 'data/interim/cleaning_summary.csv')\n"
            "display(cleaning_summary)\n"
            "display(impact)\n"
            "display(cleaning_steps)\n"
            'print(f"Le nettoyage conserve {len(clean):,} lignes sur {len(raw):,}, soit {len(clean)/len(raw):.1%}, et le revenu nettoyé vaut £{clean.Revenue.sum():,.2f}.")'
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — nettoyage\n\n"
            "Le nettoyage conserve **519 726 lignes sur 541 909**, soit **95,9 %**, "
            "pour un revenu nettoyé calculé de **£9 821 479,21**. Les annulations et "
            "codes non-produits sont donc traités avant toute modélisation."
        ),
        nbf.v4.new_markdown_cell("## b. Revenu journalier et hebdomadaire"),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(daily.date, daily.revenue)\n"
            "ax.fill_between(daily.date, 0, daily.revenue, where=~daily.is_trading_day, alpha=.25, label='fermé')\n"
            "ax.set(title='Revenu journalier', ylabel='£'); ax.legend(); fig.tight_layout(); fig.savefig(figures/'daily_revenue.png'); plt.show()\n"
            "fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(weekly.date, weekly.revenue)\n"
            "partial = weekly.loc[weekly.is_partial_week]; ax.plot(partial.date, partial.revenue, linestyle='None', marker='o', label='semaine partielle')\n"
            "ax.set(title='Revenu hebdomadaire', ylabel='£'); ax.legend(); fig.tight_layout(); fig.savefig(figures/'weekly_revenue.png'); plt.show()\n"
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
            "print(f\"Série: {daily.date.min().date()} à {daily.date.max().date()}, {len(daily)} jours.\")\n"
            "print(f\"IDXMAX: date={max_day.date.date()}, revenue=£{max_day.revenue:,.2f}\")\n"
            "print('TOP5_DAYS:'); print(top5_days.to_string(index=False))\n"
            "print('MONTHLY_ACTIVE_MEANS_COMPLETE='); print(monthly_stats[['mean', 'n_days']].to_string())\n"
            "print(f\"TENDANCE_JAN_OCT: janvier=£{jan_mean:,.2f}, octobre=£{oct_mean:,.2f}, variation=£{oct_mean-jan_mean:,.2f} ({oct_mean/jan_mean-1:.2%}); pente linéaire janvier-octobre=£{trend_slope:,.2f}/jour. Avec un seul cycle annuel, la tendance de fond n'est pas identifiable de façon fiable.\")"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — tendance\n\n"
            "Le maximum calculé par `daily.revenue.idxmax()` est **£110 815,68 le "
            "2011-11-14**. Sur les mois complets de janvier à novembre 2011, la "
            "moyenne par jour actif passe de **£23 530,54 en janvier** à "
            "**£41 190,79 en octobre**, soit **+75,05 %** hors novembre ; la pente "
            "janvier-octobre est de **£73,63/jour**. Avec un seul cycle annuel, "
            "la tendance de fond n'est pas identifiable de façon fiable."
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — fermeture observée\n\n"
            "La fenêtre du **24 décembre 2010 au 3 janvier 2011** contient **0 transaction** "
            "et **£0,00** de revenu dans les données. Pour Noël 2011, cette fermeture est "
            "une hypothèse de calendrier affichée, car un seul Noël est observé."
        ),
        nbf.v4.new_code_cell(
            "closure_window = daily.loc[daily.date.between('2010-12-15', '2011-01-10')]\n"
            "fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(closure_window.date, closure_window.revenue, marker='o')\n"
            "ax.axvspan(pd.Timestamp('2010-12-24'), pd.Timestamp('2011-01-03'), alpha=.2, color='red', label='fermeture 24 déc.–3 janv.')\n"
            "ax.set(title='Fenêtre de fermeture de Noël', ylabel='£'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures/'christmas_closure_window.png'); plt.show()\n"
            "print(f\"Fenêtre 2010-12-15 → 2011-01-10 : {len(closure_window)} jours, {int(closure_window.n_transactions.sum())} transactions dans la fermeture, £{closure_window.loc[closure_window.date.between('2010-12-24','2011-01-03'),'revenue'].sum():,.2f} sur la fermeture.\")"
        ),
        nbf.v4.new_markdown_cell("## c. Profil par jour de semaine"),
        nbf.v4.new_code_cell(
            "display(profile)\n"
            "fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(profile, x='weekday', y='mean', ax=ax)\n"
            "ax.set(title='Revenu moyen des jours actifs', ylabel='£'); fig.tight_layout(); fig.savefig(figures/'weekday_profile.png'); plt.show()\n"
            'print(f"Le samedi compte {int((daily.date.dt.dayofweek==5).sum())} jours dans la série et {int(daily.loc[daily.date.dt.dayofweek==5, "revenue"].sum()):,.2f} £ de revenu.")'
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — saisonnalité hebdomadaire\n\n"
            "Le profil calculé couvre **6 jours actifs** : lundi à vendredi et dimanche. "
            "Le samedi est structurellement fermé dans la série et doit être modélisé par "
            "la variable `is_trading_day`, non comme une absence aléatoire."
        ),
        nbf.v4.new_markdown_cell("## d. Saisonnalité mensuelle"),
        nbf.v4.new_code_cell(
            "monthly = summary.query(\"dimension == 'month'\")\n"
            "fig, ax = plt.subplots(figsize=(10, 4)); sns.barplot(monthly, x='key', y='revenue', ax=ax); ax.tick_params(axis='x', rotation=60)\n"
            "ax.set(title='Revenu mensuel', ylabel='£'); fig.tight_layout(); fig.savefig(figures/'monthly_revenue.png'); plt.show()\n"
            "nov = monthly.loc[monthly.key.eq('2011-11'), 'revenue'].iloc[0]\n"
            "print(f\"NOVEMBER_TOTAL: £{nov:,.2f}; rang={nov_total_rank}/{len(monthly_stats)}, ratio_mois_median={nov_total_ratio:.2f}x.\")\n"
            "print(f\"NOVEMBER_ACTIVE_DAILY_MEAN: £{nov_stats['mean']:,.2f}; rang={nov_mean_rank}/{len(monthly_stats)}, ratio_moyenne_mensuelle_mediane={nov_mean_ratio:.2f}x; jours actifs={int(nov_stats['n_days'])}.\")\n"
            "observed_monthly = active_eda.groupby('month').agg(n_days=('date', 'size'))\n"
            "print(f\"DECEMBER_2010: décembre 2010 est affecté par la fermeture du 24 au 31, soit {int(observed_monthly.loc['2010-12', 'n_days'])} jours actifs observés; exclu du rang.\")\n"
            "print(f\"DECEMBER_PARTIAL: décembre 2011 est partiel, jusqu'au {daily.date.max().date()}, soit {int(observed_monthly.loc['2011-12','n_days'])} jours actifs observés; exclu du rang.\")\n"
            "print(f\"CLOSURE: {int(daily.loc[daily.date.between('2010-12-24','2011-01-03'),'n_transactions'].sum())} transactions et £{daily.loc[daily.date.between('2010-12-24','2011-01-03'),'revenue'].sum():,.2f} du 24 décembre au 3 janvier.\")"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — saisonnalité mensuelle\n\n"
            "Sur les **11 mois complets de janvier à novembre 2011**, novembre est "
            "premier en revenu total (**£1 438 851,68**, ratio **2,09×** la médiane) "
            "et premier en moyenne par jour actif (**£55 340,45**, ratio **2,09×**). "
            "Décembre 2010 est affecté par la fermeture du 24 au 31 et décembre 2011 "
            "est partiel jusqu'au 8 : ces deux mois sont exclus du rang."
        ),
        nbf.v4.new_markdown_cell("## e. Décomposition hebdomadaire"),
        nbf.v4.new_code_cell(
            "decomp = decompose_weekly(daily)\n"
            "fig = decomp.plot(); fig.set_size_inches(12, 8); fig.tight_layout(); fig.savefig(figures/'decomposition.png'); plt.show()\n"
            "print(f\"DECOMPOSITION_DAYS: {len(decomp.observed)} jours après exclusion des jours fermés et de la fenêtre 24 décembre–3 janvier.\")\n"
            "print('Méthode: jours fermés et la fenêtre 24 décembre–3 janvier retirés; "
            "saisonnalité = moyenne par jour de semaine et résidu = écart après tendance roulante de 7 observations actives.')"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — décomposition\n\n"
            "La décomposition utilise une période hebdomadaire de **7 jours** et "
            "porte sur exactement **304 jours actifs** après exclusion des jours fermés "
            "et de la fenêtre du 24 décembre au 3 janvier. Seule la composante "
            "hebdomadaire est interprétable de façon robuste ; l’annualité est limitée "
            "par le seul Noël observé."
        ),
        nbf.v4.new_markdown_cell("## f. Jours anormaux"),
        nbf.v4.new_code_cell(
            "display(details)\n"
            "fig, ax = plt.subplots(figsize=(10, 4)); ax.scatter(anomalous.date, anomalous.revenue); ax.tick_params(axis='x', rotation=60)\n"
            "ax.set(title='Jours anormaux détectés', ylabel='£'); fig.tight_layout(); fig.savefig(figures/'anomalies.png'); plt.show()\n"
            "for date, group in details.groupby('date'):\n"
            '    print(f"{date.date()}: top 3 = £{group.Revenue.sum():,.2f}, part = {group.share_of_day.sum():.1%}; conclusion: contribution des trois plus grosses lignes à examiner avant de qualifier le jour de pic réel.")'
            "\n"
            "day_0920 = clean.loc[clean.InvoiceDate.dt.normalize().eq(pd.Timestamp('2011-09-20'))].copy()\n"
            "customer_0920 = day_0920.groupby('CustomerID', dropna=False)['Revenue'].sum().sort_values(ascending=False).head(5).rename('revenue').reset_index()\n"
            "customer_0920['customer_key'] = customer_0920.CustomerID.astype('string').str.replace(r'\\.0$', '', regex=True)\n"
            "customer_0920['share_of_day'] = customer_0920.revenue / day_0920.Revenue.sum()\n"
            "display(customer_0920)\n"
            "share_17450 = customer_0920.loc[customer_0920.customer_key.eq('17450'), 'share_of_day'].iloc[0]\n"
            "print(f\"CustomerID 17450 représente {share_17450:.2%} du revenu du 2011-09-20.\")"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — anomalies\n\n"
            "Les **7 jours anormaux** représentent **£475 740,25**, soit **4,86 %** "
            "du revenu journalier total. Le 2011-09-20 est inclus et doit être signalé "
            "comme événement non prévisible : le client **17450 représente 64,69 %** "
            "du revenu de cette journée."
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
            "fig, ax = plt.subplots(figsize=(10, 4)); ax.hist(with_anomalies, bins=30, alpha=.55, label='avec anomalies'); ax.hist(without_anomalies, bins=30, alpha=.55, label='sans anomalies'); ax.set(title='Effet des anomalies sur la distribution journalière', xlabel='Revenu (£)', ylabel='Nombre de jours'); ax.legend(); fig.tight_layout(); fig.savefig(figures/'anomaly_effect.png'); plt.show()\n"
            "print(f\"Avec anomalies : moyenne £{with_anomalies.mean():,.2f}, médiane £{with_anomalies.median():,.2f}, écart-type £{with_anomalies.std():,.2f}, part {with_anomalies.sum()/daily.revenue.sum():.2%}.\")\n"
            "print(f\"Sans anomalies : moyenne £{without_anomalies.mean():,.2f}, médiane £{without_anomalies.median():,.2f}, écart-type £{without_anomalies.std():,.2f}, part {without_anomalies.sum()/daily.revenue.sum():.2%}.\")\n"
            "mean_drop = 1 - without_anomalies.mean()/with_anomalies.mean()\n"
            "std_drop = 1 - without_anomalies.std()/with_anomalies.std()\n"
            "print(f\"ANOMALY_RELATIVE_DROP: moyenne={mean_drop:.2%}, écart-type={std_drop:.2%}; conclusion: les anomalies expliquent peu de la variance si la baisse de l'écart-type reste faible.\")"
        ),
        nbf.v4.new_markdown_cell("## g. Pays et produits"),
        nbf.v4.new_code_cell(
            "country = summary.query(\"dimension == 'country'\").sort_values('revenue', ascending=False).head(15)\n"
            "products = summary.query(\"dimension == 'product'\").head(20)\n"
            "fig, ax = plt.subplots(figsize=(10, 5)); sns.barplot(country, y='key', x='revenue', ax=ax); fig.tight_layout(); fig.savefig(figures/'country.png'); plt.show()\n"
            "fig, ax = plt.subplots(figsize=(10, 7)); sns.barplot(products, y='key', x='revenue', ax=ax); fig.tight_layout(); fig.savefig(figures/'products.png'); plt.show()\n"
            "uk = country.loc[country.key.eq('United Kingdom'), 'revenue'].iloc[0]\n"
            'print(f"Le Royaume-Uni représente {uk/clean.Revenue.sum():.1%} du revenu nettoyé parmi les pays résumés.")'
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — concentration géographique\n\n"
            "Le Royaume-Uni représente **84,46 %** du revenu nettoyé calculé. "
            "La concentration géographique est donc forte ; les autres pays doivent "
            "être suivis comme segments minoritaires plutôt que mélangés sans contrôle."
        ),
        nbf.v4.new_markdown_cell("## h. Synthèse et implications"),
        nbf.v4.new_code_cell(
            'print(f"1. {len(clean):,} transactions nettoyées produisent £{clean.Revenue.sum():,.2f}.")\n'
            'print(f"2. La série comporte {len(daily)} jours et {int((~daily.is_trading_day).sum())} jours non actifs.")\n'
            'print(f"3. Le pic journalier atteint £{daily.revenue.max():,.2f} le {daily.loc[daily.revenue.idxmax(), "date"].date()}.")\n'
            'print(f"4. Novembre totalise £{nov:,.2f}; un seul Noël est observé, donc la saisonnalité de fin d’année est fragile.")\n'
            'print(f"5. Les {len(anomalous)} jours anormaux doivent être conservés comme événements candidats, pas supprimés automatiquement.")\n'
            'print(f"6. La série contient {int((daily.n_transactions==0).sum())} jours sans transaction; les jours fermés doivent être traités explicitement dans les features.")\n'
            "print('Implications: le modèle devra intégrer les fermetures, le pic de novembre et l’incertitude liée à un seul cycle de Noël; la cible reste le revenu total.')"
        ),
        nbf.v4.new_markdown_cell(
            "### Conclusion — synthèse\n\n"
            "Les résultats calculés combinent **304 jours décomposés**, **69 jours non actifs**, "
            "un maximum journalier de **£110 815,68 le 2011-11-14**, une hausse de "
            "**75,05 % entre janvier et octobre** sur les moyennes journalières actives, "
            "**7 anomalies** et une concentration UK de **84,46 %**. Avec un seul cycle "
            "annuel, la tendance de fond reste incertaine ; les features doivent représenter "
            "les fermetures et les événements sans supprimer les observations."
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
