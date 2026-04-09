"""Unconditional analysis of 0DTE option strategy returns.

Loads data_structures.parquet, data_opt.parquet, vix.parquet,
ex_post_moments.h5, and ALL_eod.csv.  Produces LaTeX tables
(``0dte_stratret*.tex``) and PDF figures (time-series plots, bar charts,
VRP plots).

Adapted from the private-repo script for the public replication package.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator, interp1d
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler

from _paths import get_project_root, get_data_dir, get_tables_dir, get_figures_dir

idx = pd.IndexSlice
pd.options.mode.chained_assignment = None

# ---------------------------------------------------------------------------
# Helper stubs for functions that lived in private-repo utility modules
# ---------------------------------------------------------------------------

MNES_LIMITS = [0.5, 0.97, 0.98, 0.995, 1.005, 1.02, 1.03, 1.5]
MNES_BUCKETS = [str((MNES_LIMITS[i], MNES_LIMITS[i + 1])) for i in range(len(MNES_LIMITS) - 1)]
MNES_LABELS = ['VDITM', 'DITM', 'ITM', 'ATM', 'OTM', 'DOTM', 'VDOTM']
CALL_LABELS = {k: v for k, v in zip(MNES_BUCKETS, MNES_LABELS)}
PUT_LABELS = {k: v for k, v in zip(MNES_BUCKETS, MNES_LABELS[-1::-1])}

IONAMES = {
    'quote_time': 'Bar time',
    'mean': 'Mean',
    'std': 'Volatility',
    'min': 'Min',
    'max': 'Max',
    'skew': 'Skew',
    '25%': r'25\%',
    '50%': r'50\%',
    '75%': r'75\%',
    '1%': r'1\%',
    '99%': r'99\%',
    'count': 'Count',
    'sr': 'SR, p.a.',
    'const': 'Constant',
}


def newey_west_sem(group, lags=1):
    yy = group.dropna().values
    if len(yy.shape) > 1:
        nr, nc = yy.shape
        if nr * nc == 1:
            yy = yy.reshape((-1, 1))
            nc = 1
    else:
        yy = yy.reshape((-1, 1))
        nc = 1
    res = []
    for c in range(nc):
        y = yy[:, c]
        X = np.ones_like(y)
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        res.append(model.HC0_se[0])
    return pd.Series(res)


def robust_mean_se(group):
    y = group.values
    X = np.ones_like(y)
    model = sm.RLM(y, X).fit()
    robust_mean = model.params[0]
    robust_se = model.bse[0]
    return pd.Series({'mean': robust_mean, 'se': robust_se})


# ---------------------------------------------------------------------------
# Environment-variable helpers (kept from original)
# ---------------------------------------------------------------------------

def _env_int(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return int(val)


def _normalize_data_version(value):
    return value if value.endswith('/') else f"{value}/"


# ---------------------------------------------------------------------------
# Configuration flags (overridable via environment variables)
# ---------------------------------------------------------------------------

Buckets_for_returns_Prep1_Save2_Load3_LoadJustOpt4 = _env_int('ODTE_BUCKETS_MODE', 4)

TSFigures_0DTE_Strats = _env_int('ODTE_TSFIGURES', 1)
MakeGraphsOptions = _env_int('ODTE_MAKE_GRAPHS_OPTIONS', 1)
MakeGraphsStrats = _env_int('ODTE_MAKE_GRAPHS_STRATS', 1)
Continue_Processing_0DTE_Strats = _env_int('ODTE_CONTINUE_PROCESSING', 1)
AnalyzeVRP = _env_int('ODTE_ANALYZE_VRP', 1)

mnes_limits = MNES_LIMITS
mnes_buckets = MNES_BUCKETS
call_labels = CALL_LABELS
put_labels = PUT_LABELS

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

strategies = {
    "strangle": [(1, 1), (0.995, 1.005), (0.99, 1.01), (0.985, 1.015), (0.98, 1.02)],
    "iron_condor": [(0.995, 1, 1.005), (0.99, 1, 1.01), (0.98, 1, 1.02),
                    (0.995, 0.997, 1.003, 1.005), (0.99, 0.995, 1.005, 1.01), (0.98, 0.99, 1.01, 1.02)],
    "risk_reversal": [(0.995, 1.005), (0.99, 1.01), (0.985, 1.015), (0.98, 1.02)],
    "bull_call_spread": [(1, 1.005), (1, 1.01), (1, 1.015), (1, 1.02),
                         (0.995, 1.005), (0.99, 1.01), (0.985, 1.015), (0.98, 1.02)],
    "call_ratio_spread": [(1, 1.005), (1, 1.01), (1, 1.015), (1, 1.02),
                          (0.995, 1.005), (0.99, 1.01), (0.985, 1.015), (0.98, 1.02)],
    "bear_put_spread": [(1, 0.995), (1, 0.99), (1, 0.985), (1, 0.98),
                        (1.005, 0.995), (1.01, 0.99), (1.015, 0.985), (1.02, 0.98)],
    "put_ratio_spread": [(1, 0.995), (1, 0.99), (1, 0.985), (1, 0.98),
                         (1.005, 0.995), (1.01, 0.99), (1.015, 0.985), (1.02, 0.98)]
}
strategies = {key: [tuple(sorted(z)) for z in values] for key, values in strategies.items()}
strats_sort = {key: ['/'.join([str(zz) for zz in sorted(z)]) for z in values] for key, values in strategies.items()}

strat_names = {
    'strangle': 'Strangle/Straddle',
    'iron_condor': 'Iron Butterfly/Condor',
    'bull_call_spread': 'Bull Call Spread',
    'itm_bull_call_spread': 'ITM Bull Call Spread',
    'bear_put_spread': 'Bear Put Spread',
    'itm_bear_put_spread': 'ITM Bear Put Spread',
    'call_ratio_spread': 'Call Ratio Spread',
    'itm_call_ratio_spread': 'ITM Call Ratio Spread',
    'put_ratio_spread': 'Put Ratio Spread',
    'itm_put_ratio_spread': 'ITM Put Ratio Spread',
    'risk_reversal': 'Risk Reversal'
}


# ---------------------------------------------------------------------------
# Analysis helper
# ---------------------------------------------------------------------------

def fn_prepare_stats_for_plot_tabs(interpolated_data, option_type,
                                   selected_times, vars=['tv_pct', 'mid_pct'],
                                   robust_est=False,
                                   exclude_mnes_with_low_obs=True,
                                   winsorize_level=0.0,
                                   period=(2000, 2100)):
    res_avr = {}
    res_ci = {}
    extras = {}

    vars_to_convert = [z for z in ['tv', 'mid', 'bas'] if z in interpolated_data.columns]
    interpolated_data[vars_to_convert] = interpolated_data[vars_to_convert] * 100
    for time in selected_times:
        vars_initial = vars.copy()
        start_date = pd.to_datetime(f'{period[0]}-01-01')
        end_date = pd.to_datetime(f'{period[1]}-12-31')
        mask = (interpolated_data['quote_time'] == time) \
               & (interpolated_data['quote_date'] >= start_date) & (interpolated_data['quote_date'] <= end_date)
        time_data = interpolated_data[mask].copy()

        if exclude_mnes_with_low_obs:
            cc = time_data.groupby('mnes', sort=False, observed=True)[vars[0]].count()
            cc = cc / cc.max()
            sel = (cc[cc > 0.2]).index.max()
            time_data = time_data[time_data.mnes <= sel]
            sel = (cc[cc > 0.2]).index.min()
            time_data = time_data[time_data.mnes >= sel]

        if winsorize_level > 0:
            wl = (winsorize_level, winsorize_level)
            vars_extra = [f'{z}_wins' for z in vars_initial]
            time_data[vars_extra] = \
                (time_data.groupby('mnes', observed=True, group_keys=False, sort=False)[vars_initial].
                 apply(lambda x: x.apply(lambda col: winsorize(col, limits=wl, nan_policy='omit'))))
            vars_initial += vars_extra

        if robust_est:
            grouped_results = time_data.groupby('mnes', observed=True, sort=False)[vars_initial].apply(
                lambda x: x.apply(robust_mean_se)).unstack()
            res_avr[time] = grouped_results.xs('mean', level=1, axis=1)
            res_ci[time] = grouped_results.xs('se', level=1, axis=1) * 1.96
        else:
            grouped_mean = time_data.groupby('mnes', sort=False, observed=True)[vars_initial].mean()
            se_values = time_data.groupby('mnes', sort=False, observed=True)[vars_initial].std() / np.sqrt(
                time_data.groupby('mnes', sort=False, observed=True)[vars_initial].count())
            ci_values = 1.96 * se_values
            res_avr[time] = grouped_mean
            res_ci[time] = ci_values

        for cv in vars_initial:
            df = time_data.groupby('mnes', observed=True, sort=False)[cv].describe(percentiles=[.01, .25, .5, .75, .99])
            skewness = time_data.groupby('mnes', observed=True, sort=False)[cv].skew()
            df['skew'] = skewness
            df['sr'] = df['mean'] / df['std'] * np.sqrt(252)
            extras = extras | {(time, cv): df.copy()}

        grouped_median = time_data.groupby('mnes', sort=False, observed=True)[vars_initial].median()
        grouped_percentile_25 = time_data.groupby('mnes', sort=False, observed=True)[vars_initial].quantile(0.25)
        grouped_percentile_75 = time_data.groupby('mnes', sort=False, observed=True)[vars_initial].quantile(0.75)
        extras = extras | {(time, 'grouped_median'): grouped_median,
                           (time, 'grouped_percentile_25'): grouped_percentile_25,
                           (time, 'grouped_percentile_75'): grouped_percentile_75}

    return (res_avr, res_ci, extras), vars_initial


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(project_root: Path | None = None):
    root = get_project_root(project_root)
    data_dir = get_data_dir(root)
    figs_dir = get_figures_dir(root)
    tables_dir = get_tables_dir(root)

    # ###################################
    # ######### LOAD THE DATA ###########
    # ###################################
    df_opt = pd.read_parquet(data_dir / 'data_opt.parquet')
    strats_all = pd.read_parquet(data_dir / 'data_structures.parquet')
    df_vix = pd.read_parquet(data_dir / 'vix.parquet')
    df_vix['quote_time'] = pd.to_datetime(df_vix['quote_time'], format='%H:%M:%S').dt.time
    df_rv = pd.read_hdf(data_dir / 'ex_post_moments.h5', key='future_moments_SPX')
    df_opt = df_opt.sort_values(by=['quote_date', 'quote_time', 'option_type', 'mnes_rel'])

    df_eod = pd.read_csv(data_dir / 'ALL_eod.csv')
    df_eod['Date'] = pd.to_datetime(df_eod['Date'])
    mask = (df_eod.Date >= df_opt.quote_date.min()) & (df_eod.Date <= df_opt.quote_date.max())
    df_eod = df_eod[mask]

    df_spx_eod = df_eod[df_eod.root == 'SPX']
    df_spx_eod = df_spx_eod.set_index('Date')
    df_spx_eod['Close01'] = MinMaxScaler((0, 1)).fit_transform(df_spx_eod['Close'].values.reshape(-1, 1))
    df_vix_eod = df_eod[df_eod.root == 'VIX']
    df_vix_eod = df_vix_eod.set_index('Date')
    df_vix_eod['Close01'] = MinMaxScaler((0, 1)).fit_transform(df_vix_eod['Close'].values.reshape(-1, 1))

    df_vix_eod = df_vix_eod.rename(columns={'Close': 'VIX'})
    df_spx_eod = df_spx_eod.rename(columns={'Close': 'SPX'})

    # ###################################
    # TS Figures for 0DTE Strategies
    # ###################################
    if TSFigures_0DTE_Strats:
        all_periods = ((2000, 2100),)
        for period in all_periods:
            selected_times_strat = ['10:00:00']
            selected_times_strat = [pd.to_datetime(y).time() for y in selected_times_strat]
            vars_all = ['reth_und']
            res_ts_strats = {}
            for opt_type in strategies.keys():
                mask = (strats_all.option_type == opt_type) & (strats_all.quote_time == selected_times_strat[0])
                opt_temp = strats_all[mask]

                sel = opt_temp.mnes.unique()
                rets = opt_temp.groupby(['mnes'])[['quote_date', 'reth_und']]. \
                    apply(lambda x: x.sort_values('quote_date').set_index('quote_date').
                          rolling(window=63, min_periods=1).apply(lambda z: np.nanmean(z)))
                rets = rets.unstack(level=0)
                rets.columns = rets.columns.droplevel(0)
                res_ts_strats[opt_type] = rets.copy()

        rest = {}
        for k, i in res_ts_strats.items():
            rest[k] = i.mean(axis=1)
        pd.DataFrame(rest).corr()

        opt_type = 'iron_condor'
        mask = (strats_all.option_type == opt_type) & (strats_all.quote_time == selected_times_strat[0])
        opt_temp = strats_all[mask].set_index(['mnes', 'quote_date'])['reth_und'].sort_index().unstack(level=0)

        opt_temp = pd.concat([opt_temp, df_spx_eod['SPX'], df_vix_eod['VIX']], axis=1)

        for opt_type in strategies.keys():
            with PdfPages(figs_dir / f'ts_{opt_type}_reth_und.pdf') as pdf:
                fig, ax = plt.subplots(figsize=(10, 6.5))

                res_ts_strats[opt_type].ffill(limit=10).plot(ax=ax)

                ax.set_title(f'Moving average PNL for {opt_type}')
                ax.set_xlabel('Date')
                ax.set_ylabel('% of Underlying')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
                ax.legend(loc='best')

                ax2 = ax.twinx()
                df_spx_eod['Close01'].plot(ax=ax2, color='green', linewidth=0.3, alpha=0.25)
                df_vix_eod['Close01'].plot(ax=ax2, color='blue', linewidth=0.3, alpha=0.25)
                ax2.set_ylabel('SPX/VIX Scaled Level', color='black')
                ax2.tick_params(axis='y', labelcolor='black')

                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12], bymonthday=-1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

                start_date = res_ts_strats[opt_type].index.min()
                end_date = res_ts_strats[opt_type].index.max()
                quarter_ends = pd.date_range(start=start_date, end=end_date, freq='Y')

                for quarter_end in quarter_ends:
                    ax.axvline(x=quarter_end, color='red', linestyle=':', linewidth=0.1)

                ax.set_xlim(start_date, end_date)

                plt.setp(ax.get_xticklabels(), rotation=0)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

    # ###################################
    # make graphs CALLS and PUTS
    # ###################################
    if MakeGraphsOptions:
        vars_all = ['tv', 'mid', 'bas', 'reth', 'reth_und']
        selected_times = ['10:00:00', '13:00:00', '15:00:00', '16:00:00']
        selected_times = [pd.to_datetime(y).time() for y in selected_times]
        res_calls, vars_adj = fn_prepare_stats_for_plot_tabs(df_opt[df_opt.option_type == 'C'], option_type='c',
                                                             selected_times=selected_times,
                                                             vars=vars_all,
                                                             winsorize_level=0)
        res_puts, vars_adj = fn_prepare_stats_for_plot_tabs(df_opt[df_opt.option_type == 'P'], option_type='p',
                                                            selected_times=selected_times,
                                                            vars=vars_all,
                                                            winsorize_level=0)
        conf_int = False
        for var_curr in vars_adj:
            for opt_type in ['calls', 'puts']:
                res_avr, res_ci, extras = res_calls if opt_type == 'calls' else res_puts
                with PdfPages(figs_dir / f'{opt_type}_{var_curr}_bymnes_bars.pdf') as pdf:
                    for z in selected_times:
                        grouped_mean = res_avr[z][var_curr].sort_index()
                        confidence_interval = res_ci[z][var_curr].reindex(index=grouped_mean.index)
                        grouped_median = extras[(z, 'grouped_median')][var_curr].reindex(index=grouped_mean.index)
                        grouped_percentile_25 = extras[(z, 'grouped_percentile_25')][var_curr].reindex(index=grouped_mean.index)
                        grouped_percentile_75 = extras[(z, 'grouped_percentile_75')][var_curr].reindex(index=grouped_mean.index)

                        fig, ax = plt.subplots(figsize=(10, 6.5))

                        x_positions = range(len(grouped_mean))
                        bars_color = plt.cm.RdBu(200)
                        if conf_int:
                            ax.bar(x_positions, grouped_mean, width=0.8, color=bars_color,
                                   yerr=confidence_interval.values, capsize=5, alpha=0.7, label='Mean')
                        else:
                            ax.bar(x_positions, grouped_mean, width=0.8, color=bars_color, label='Mean')

                        for xpos, p25, p75, pmed in zip(x_positions, grouped_percentile_25, grouped_percentile_75,
                                                        grouped_median):
                            ax.hlines(pmed, xpos - 0.2, xpos + 0.2, colors='green', lw=4,
                                      label='Median' if xpos == 0 else "")
                            ax.hlines(p25, xpos - 0.4, xpos - 0.2, colors='blue', lw=3,
                                      label='25th Percentile' if xpos == 0 else "")
                            ax.hlines(p75, xpos + 0.2, xpos + 0.4, colors='red', lw=3,
                                      label='75th Percentile' if xpos == 0 else "")

                        ax.set_title(f'Average {var_curr} for {opt_type} time {z}')
                        ax.set_xlabel('Moneyness in %')
                        ylab = '% of Option Mid-price' if var_curr in ['bas', 'reth', 'bas_wins',
                                                                       'reth_wins'] else '% of Underlying'
                        ax.set_ylabel(ylab)
                        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

                        ax.legend(loc='best')
                        ax.set_xticks(x_positions)
                        ax.set_xticklabels([f'{t / 1e3:.3f}' for t in grouped_mean.index], rotation=45)

                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close()

    # ###################################
    # PREPARE STATS / make graphs STRATS
    # ###################################
    if MakeGraphsStrats:
        selected_times_strat = ['10:00:00', '13:00:00', '15:00:00', '16:00:00']
        selected_times_strat = [pd.to_datetime(y).time() for y in selected_times_strat]
        vars_all = ['mid', 'reth_und']
        res_strats = {}
        for opt_type in strats_all.option_type.unique():
            opt_temp = strats_all[strats_all.option_type == opt_type]
            opt_temp['mnes'] = opt_temp['mnes'].astype(str)
            temp, vars_adj = fn_prepare_stats_for_plot_tabs(opt_temp, option_type=opt_temp,
                                                            selected_times=selected_times_strat,
                                                            vars=vars_all,
                                                            winsorize_level=0)

            res_strats = res_strats | {opt_type: temp}

            conf_int = False
            for var_curr in vars_adj:
                res_avr, res_ci, extras = temp
                with PdfPages(figs_dir / f'{opt_type}_{var_curr}_bymnes_bars.pdf') as pdf:
                    for z in selected_times_strat:
                        grouped_mean = res_avr[z][var_curr].reindex(strats_sort[opt_type])
                        confidence_interval = res_ci[z][var_curr].reindex(strats_sort[opt_type])
                        grouped_median = extras[(z, 'grouped_median')][var_curr].reindex(strats_sort[opt_type])
                        grouped_percentile_25 = extras[(z, 'grouped_percentile_25')][var_curr].reindex(
                            strats_sort[opt_type])
                        grouped_percentile_75 = extras[(z, 'grouped_percentile_75')][var_curr].reindex(
                            strats_sort[opt_type])

                        fig, ax = plt.subplots(figsize=(10, 6))

                        x_positions = range(len(grouped_mean))
                        bars_color = plt.cm.RdBu(200)
                        if conf_int:
                            ax.bar(x_positions, grouped_mean, width=0.8, color=bars_color,
                                   yerr=confidence_interval.values, capsize=5, alpha=0.7)
                        else:
                            ax.bar(x_positions, grouped_mean, width=0.8, color=bars_color,
                                   label='Mean')

                        for xpos, p25, p75, pmed in zip(x_positions, grouped_percentile_25, grouped_percentile_75,
                                                        grouped_median):
                            ax.hlines(pmed, xpos - 0.2, xpos + 0.2, colors='green', lw=4,
                                      label='Median' if xpos == 0 else "")
                            ax.hlines(p25, xpos - 0.4, xpos - 0.2, colors='blue', lw=3,
                                      label='25th Percentile' if xpos == 0 else "")
                            ax.hlines(p75, xpos + 0.2, xpos + 0.4, colors='red', lw=3,
                                      label='75th Percentile' if xpos == 0 else "")

                        ax.set_title(f'Average {var_curr} for {opt_type} time {z}')
                        ax.legend(loc='best')
                        ax.set_xlabel('Moneyness')
                        ylab = '% of Option Mid-price' if var_curr in ['bas', 'reth', 'bas_wins',
                                                                       'reth_wins'] else '% of Underlying'
                        ax.set_ylabel(ylab)
                        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

                        ax.set_xticks(x_positions)
                        ax.set_xticklabels([f'{t}' for t in grouped_mean.index], rotation=10)

                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close()

    # ###################################
    # make graphs VRP
    # ###################################
    if AnalyzeVRP:
        df_vix_vrp = df_vix[df_vix.dts == 0].copy()
        df_vix_vrp = df_vix_vrp.merge(df_rv[['date', 'time', 'SPX_lrv']],
                                       left_on=['quote_date', 'quote_time'],
                                       right_on=['date', 'time'])

        df_vix_vrp['vrp_pct'] = (df_vix_vrp['vix'] - df_vix_vrp['SPX_lrv'] / 1e5) * 100
        df_vix_vrp['vix_pct'] = df_vix_vrp['vix'] * 100
        df_vix_vrp['vrp_volapct'] = (df_vix_vrp['vix'] ** 0.5 - (df_vix_vrp['SPX_lrv'] / 1e5) ** 0.5) * 100
        df_vix_vrp['vix_volapct'] = df_vix_vrp['vix'] ** 0.5 * 100

        vars_to_print = ['vrp_pct', 'vix_pct', 'vrp_volapct', 'vix_volapct']
        df_vrp_hours = df_vix_vrp.query('(root=="SPXW") & (dts==0)').set_index('quote_datetime')
        df_vrp_hours = df_vrp_hours[vars_to_print]

        df_vrp_hours.dropna(inplace=True)

        grouped_mean = df_vrp_hours.groupby(df_vrp_hours.index.time, observed=True).mean()
        grouped_median = df_vrp_hours.groupby(df_vrp_hours.index.time, observed=True).median()
        grouped_percentile_25 = df_vrp_hours.groupby(df_vrp_hours.index.time, observed=True).quantile(0.25)
        grouped_percentile_75 = df_vrp_hours.groupby(df_vrp_hours.index.time, observed=True).quantile(0.75)
        grouped_sem = df_vrp_hours.groupby(df_vrp_hours.index.time, observed=True).apply(newey_west_sem, lags=3)
        grouped_sem.columns = vars_to_print
        z_score = 1.96
        confidence_interval = z_score * grouped_sem

        with PdfPages(figs_dir / 'vrp_bytime_bars.pdf') as pdf:
            for z in vars_to_print:
                fig, ax = plt.subplots(figsize=(10, 6))

                x_positions = range(len(grouped_mean))
                bars_color = plt.cm.RdBu(200)

                ax.bar(x_positions, grouped_mean[z], width=0.8, color=bars_color,
                       yerr=confidence_interval[z].values.T, capsize=5, alpha=0.7, label='Mean')

                for xpos, p25, p75, pmed in zip(x_positions, grouped_percentile_25[z], grouped_percentile_75[z],
                                                grouped_median[z]):
                    ax.hlines(pmed, xpos - 0.2, xpos + 0.2, colors='green', lw=4,
                              label='Median' if xpos == 0 else "")
                    ax.hlines(p25, xpos - 0.4, xpos - 0.2, colors='blue', lw=3,
                              label='25th Percentile' if xpos == 0 else "")
                    ax.hlines(p75, xpos + 0.2, xpos + 0.4, colors='red', lw=3,
                              label='75th Percentile' if xpos == 0 else "")
                custom_lines = [Line2D([0], [0], color=bars_color, lw=2, label='Average'),
                                Line2D([0], [0], color='black', lw=0, marker='|', markersize=15, markeredgewidth=2,
                                       label='Confidence Bounds')]
                ax.legend(loc='best')

                ax.set_title(f'Average {z} by Time of Day')
                ax.set_xlabel('Time of Day')
                ax.set_ylabel('% to expiry')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(t)[:-3] for t in grouped_mean.index], rotation=45)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

    # ###################################
    # make tables STRATS
    # ###################################
    SaveStratsTable = True
    if Continue_Processing_0DTE_Strats:
        all_periods = ((2000, 2100), (2022, 2023), (2024, 2026))
        for period in all_periods:
            selected_times_strat = ['10:00:00']
            selected_times_strat = [pd.to_datetime(y).time() for y in selected_times_strat]
            vars_all = ['reth_und']
            res_strats = {}
            for opt_type in strategies.keys():
                opt_temp = strats_all[strats_all.option_type == opt_type]
                opt_temp['mnes'] = opt_temp['mnes'].astype(str)
                temp, vars_adj = fn_prepare_stats_for_plot_tabs(opt_temp, option_type=opt_temp,
                                                                selected_times=selected_times_strat,
                                                                vars=vars_all,
                                                                winsorize_level=0,
                                                                period=period)

                res_strats = res_strats | {opt_type: temp}

            vars_summary = ['reth_und']
            var_now = vars_summary[0]
            tables_summary = {}
            summary_temp = []
            for opt_type in strategies.keys():
                res = res_strats[opt_type][-1]
                temp = res[(datetime.time(10, 0, 0), var_now)].reindex(strats_sort[opt_type]).reset_index()
                temp['opt_type'] = opt_type
                temp = temp.set_index(['opt_type', 'mnes'])
                summary_temp.append(temp.copy())
            summary_temp = pd.concat(summary_temp, axis=0)

            time_map = {datetime.time(10, 0, 0): 6,
                        datetime.time(10, 30, 0): 5.5,
                        datetime.time(11, 0, 0): 5,
                        datetime.time(11, 30, 0): 4.5,
                        datetime.time(12, 0, 0): 4,
                        datetime.time(12, 30, 0): 3.5,
                        datetime.time(13, 0, 0): 3,
                        datetime.time(13, 30, 0): 2.5,
                        datetime.time(14, 0, 0): 2,
                        datetime.time(14, 30, 0): 1.5,
                        datetime.time(15, 0, 0): 1,
                        datetime.time(15, 30, 0): 0.5}

            formatters = {
                'count': '{:.0f}'.format,
                'mean': '{:.4f}'.format,
                'std': '{:.2f}'.format,
                'min': '{:.2f}'.format,
                '25%': '{:.2f}'.format,
                '50%': '{:.2f}'.format,
                '75%': '{:.2f}'.format,
                '1%': '{:.2f}'.format,
                '99%': '{:.2f}'.format,
                'max': '{:.2f}'.format,
                'skew': '{:.2f}'.format,
                'sr': '{:.2f}'.format
            }

            df_save = summary_temp.copy()
            df_save.index.names = ['Strategy', 'Moneyness']

            df_save = df_save.apply(lambda col: col.map(formatters[col.name]) if col.name in formatters else col)
            df_save.rename(columns=IONAMES, index=strat_names, inplace=True)
            df_save = df_save.reset_index()

            replacement_dict = {'(': '', ')': '', ', ': '/'}
            for old, new in replacement_dict.items():
                df_save['Moneyness'] = df_save['Moneyness'].str.replace(old, new, regex=False)

            for i in range(df_save.shape[0] - 1, 0, -1):
                if df_save['Strategy'].iloc[i] == df_save['Strategy'].iloc[i - 1]:
                    df_save.loc[df_save.index[i], 'Strategy'] = ''

            if SaveStratsTable:
                colform = 'll' + 'c' * (df_save.shape[1] - 2)
                df_save.to_latex(tables_dir / f'0dte_stratret{period[0]}_{period[1]}.tex',
                                 column_format=colform, index=False)

    print("Unconditional analysis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unconditional 0DTE strategy analysis')
    parser.add_argument('--project-root', type=Path, default=None,
                        help='Root of the public replication repo (default: auto-detect)')
    args = parser.parse_args()
    main(project_root=args.project_root)
