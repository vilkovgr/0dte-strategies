#!/usr/bin/env python3
"""Plot payoff profiles for 0DTE option strategies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from _paths import get_project_root, get_figures_dir

stock_prices = np.arange(0, 100, 1)


def call_payoff(stock_price, strike_price):
    """Calculate the payoff for a call option."""
    return np.maximum(stock_price - strike_price, 0)


def put_payoff(stock_price, strike_price):
    """Calculate the payoff for a put option."""
    return np.maximum(strike_price - stock_price, 0)


def plot_payoff(profiles, strategy, pdf=None):
    """Plot the payoff profiles."""
    fig, ax = plt.subplots(figsize=(10, 6))
    final_payoff = np.zeros_like(stock_prices)
    for profile in profiles:
        ax.plot(stock_prices, profile['payoff'], linewidth=2, label=profile['label'])
        final_payoff += profile['payoff']
    ax.plot(stock_prices, final_payoff, 'r:',  marker='o', linewidth=2, alpha=0.3, label=f'{strategy}')
    ax.set_title(f"{strategy} Strategy Payoff Profile")
    ax.set_xlabel('Stock Price at Expiry')
    ax.set_ylabel('Payoff')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig)
        plt.close()
    else:
        plt.show()


strike_prices = {
    'low': 30,
    'medium': 50,
    'high': 70
}

option_strategies = {'Straddle': [
    {'label': 'Call', 'payoff': call_payoff(stock_prices, strike_prices['medium'])},
    {'label': 'Put', 'payoff': put_payoff(stock_prices, strike_prices['medium'])}
], 'Strangle': [
    {'label': 'Call', 'payoff': call_payoff(stock_prices, strike_prices['high'])},
    {'label': 'Put', 'payoff': put_payoff(stock_prices, strike_prices['low'])}
], 'Call Bull Spread': [
    {'label': 'Long Call', 'payoff': call_payoff(stock_prices, strike_prices['low'])},
    {'label': 'Short Call', 'payoff': -call_payoff(stock_prices, strike_prices['high'])}
], 'Put Bear Spread': [
    {'label': 'Long Put', 'payoff': put_payoff(stock_prices, strike_prices['high'])},
    {'label': 'Short Put', 'payoff': -put_payoff(stock_prices, strike_prices['low'])}
], 'Call Ratio Spread': [
    {'label': 'Long Call', 'payoff': call_payoff(stock_prices, strike_prices['medium'])},
    {'label': 'Short Call x2', 'payoff': -2 * call_payoff(stock_prices, strike_prices['high'])}
], 'Put Ratio Spread': [
    {'label': 'Long Put', 'payoff': put_payoff(stock_prices, strike_prices['medium'])},
    {'label': 'Short Put x2', 'payoff': -2 * put_payoff(stock_prices, strike_prices['low'])}
], 'Risk Reversal': [
    {'label': 'Long Call', 'payoff': call_payoff(stock_prices, strike_prices['high'])},
    {'label': 'Short Put', 'payoff': -put_payoff(stock_prices, strike_prices['low'])}
], 'Iron Butterfly': [
    {'label': 'Short ATM Call', 'payoff': -call_payoff(stock_prices, strike_prices['medium'])},
    {'label': 'Short ATM Put', 'payoff': -put_payoff(stock_prices, strike_prices['medium'])},
    {'label': 'Long OTM Call', 'payoff': call_payoff(stock_prices, strike_prices['high'])},
    {'label': 'Long OTM Put', 'payoff': put_payoff(stock_prices, strike_prices['low'])}
], 'Iron Condor': [
    {'label': 'Short OTM Call', 'payoff': -call_payoff(stock_prices, strike_prices['high'])},
    {'label': 'Short OTM Put', 'payoff': -put_payoff(stock_prices, strike_prices['low'])},
    {'label': 'Long Further OTM Call', 'payoff': call_payoff(stock_prices, strike_prices['high'] + 15)},
    {'label': 'Long Further OTM Put', 'payoff': put_payoff(stock_prices, strike_prices['low'] - 15)}
]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot strategy payoff profiles.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to ODTE_REPO_ROOT or repo root inferred from script location.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to <root>/output/figures/strat_payoffs.pdf",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root(args.project_root)
    figures_dir = get_figures_dir(root)
    output_file = args.output or (figures_dir / "strat_payoffs.pdf")

    with PdfPages(output_file) as pdf:
        for strategy, profiles in option_strategies.items():
            plot_payoff(profiles, strategy, pdf=pdf)

    print(f"Output: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
