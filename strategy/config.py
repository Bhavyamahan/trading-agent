"""All tunable numbers from Rulebook v1.1, in one place.

Change a value here only through a rulebook version bump. The plateau test
(later) varies these by +/-20%.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Params:
    # Layer 1: market regime
    breadth_min: float = 0.50
    regime_confirm_days: int = 2
    # Layer 2: universe and liquidity
    universe_size: int = 500
    universe_lookback: int = 126
    min_price: float = 50.0
    min_adv_crore: float = 15.0
    min_listing_sessions: int = 273
    # Layer 3: trend template
    above_52w_low: float = 1.30
    within_52w_high: float = 0.75
    rs_gate: float = 70.0
    # Layer 4: relative strength
    rs_bonus: float = 85.0
    rs_line_near_high: float = 0.95
    industry_min_members: int = 3
    # Layer 7: setup and trigger
    base_max_len: int = 65
    base_min_len: int = 15
    base_depth_min: float = 0.08
    base_depth_max: float = 0.35
    swing_side: int = 3
    contraction_ratio: float = 0.70
    final_contraction_max: float = 0.10
    dryup_ratio: float = 0.70
    upper_base: float = 0.90
    setup_valid_sessions: int = 10
    breakout_volume: float = 1.5
    chase_limit: float = 1.05
    upper_half: float = 0.5
    stop_buffer: float = 0.005
    stop_max: float = 0.08
    stop_min: float = 0.03
    # Sizing and risk
    risk_full: float = 0.010
    risk_half: float = 0.005
    max_position_pct: float = 0.20
    max_positions: int = 8
    max_per_industry: int = 3
    max_new_per_day: int = 2
    max_heat: float = 0.06
    dd_halve: float = 0.10
    dd_halve_recover: float = 0.05
    dd_pause: float = 0.15
    dd_pause_sessions: int = 20
    # Exits
    failed_breakout_sessions: int = 5
    failed_breakout_level: float = 0.97
    partial_r: float = 2.5
    partial_fraction: float = 1 / 3
    climax_gain_15: float = 1.25
    climax_above_dma50: float = 1.30
    fast_gain: float = 0.20
    fast_window: int = 15
    dma50_break_volume: float = 1.5
    dma50_hard_break: float = 0.97
    time_stop_sessions: int = 30
    time_stop_min_r: float = 1.0
    # Costs (verify rates at build time) and tax
    brokerage_per_order: float = 20.0
    stt: float = 0.001
    exchange_fee: float = 0.0000297
    sebi_fee: float = 0.000001
    stamp_duty_buy: float = 0.00015
    gst: float = 0.18
    dp_charge_sell: float = 15.93
    slippage: float = 0.0025
    stcg_tax: float = 0.20
    # Backtest
    starting_capital: float = 1_000_000.0


@dataclass(frozen=True)
class Run:
    """One ablation run: which layers are switched on."""
    name: str
    use_regime: bool
    use_rs_score: bool
    description: str


RUNS = [
    Run("A0", False, False, "Layers 2, 3, 7 + all exits, fixed 1% risk"),
    Run("A1", True, False, "A0 + Layer 1 market regime"),
    Run("A2", True, True, "A1 + Layer 4 relative-strength score sizing"),
]

PERIODS = {
    "in_sample": ("2010-01-01", "2019-12-31"),
    "out_of_sample": ("2020-01-01", "2099-12-31"),
}
