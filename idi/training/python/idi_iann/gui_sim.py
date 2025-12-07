"""Interactive GUI for visualizing the production CryptoMarket simulator.

NOTE: This module is an optional visualization tool. Requires matplotlib.
It is excluded from coverage but uses the production CryptoMarket.
"""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .crypto_env import CryptoMarket, MarketParams


def run_gui_sim(steps: int = 400) -> None:
    """Interactive GUI with sliders for market parameters.
    
    Uses the production CryptoMarket simulator, showing:
    - Price evolution with regime switching
    - Position and PnL tracking
    - Real-time parameter adjustment
    
    NOTE: This is an optional GUI component. Requires matplotlib.
    Raises ImportError if matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for GUI simulation. "
            "Install with: pip install matplotlib"
        )

    # Initialize market with default params
    params = MarketParams(
        drift_bull=0.002,
        drift_bear=-0.002,
        drift_chop=0.0,
        vol_base=0.01,
        vol_panic=0.04,
        shock_prob=0.02,
        shock_scale=0.05,
        fee_bps=5.0,
        seed=7,
    )
    
    market = CryptoMarket(params)
    market.reset()

    # Create figure with subplots for price, position, and PnL
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.35, hspace=0.3)
    
    ax_price = plt.subplot(3, 1, 1)
    ax_position = plt.subplot(3, 1, 2)
    ax_pnl = plt.subplot(3, 1, 3)
    
    ax_price.set_title("IDI Crypto Market Simulator (Production)")
    ax_price.set_ylabel("Price")
    ax_position.set_ylabel("Position")
    ax_position.set_ylim(-1.5, 1.5)
    ax_pnl.set_ylabel("PnL")
    ax_pnl.set_xlabel("Step")
    
    price_line, = ax_price.plot([], [], lw=2, label="Price")
    position_line, = ax_position.plot([], [], lw=2, label="Position", drawstyle="steps-post")
    pnl_line, = ax_pnl.plot([], [], lw=2, label="PnL")
    
    ax_price.legend()
    ax_position.legend()
    ax_pnl.legend()
    
    prices: list[float] = []
    positions: list[int] = []
    pnls: list[float] = []
    regimes: list[str] = []

    def reset_market() -> None:
        """Reset market with current parameters."""
        nonlocal market
        params.seed = int(params.seed)  # Ensure integer seed
        market = CryptoMarket(params)
        market.reset()
        prices.clear()
        positions.clear()
        pnls.clear()
        regimes.clear()
        prices.append(market.state.price)
        positions.append(market.state.position)
        pnls.append(market.state.pnl)
        regimes.append(market.state.regime)

    reset_market()

    def step_market() -> None:
        """Advance market by one step."""
        # Simple strategy: buy on low price, sell on high price
        current_price = market.state.price
        price_ratio = current_price / prices[0] if prices else 1.0
        
        if price_ratio < 0.95 and market.state.position <= 0:
            action = "buy"
        elif price_ratio > 1.05 and market.state.position >= 0:
            action = "sell"
        else:
            action = "hold"
        
        state, pnl, info = market.step(action)
        prices.append(state.price)
        positions.append(state.position)
        pnls.append(state.pnl)
        regimes.append(state.regime)

    def update(frame: int):
        """Animation update function."""
        step_market()
        
        x_data = list(range(len(prices)))
        
        price_line.set_data(x_data, prices)
        ax_price.set_xlim(0, max(100, len(prices)))
        if prices:
            ax_price.set_ylim(min(prices) * 0.98, max(prices) * 1.02)
        
        position_line.set_data(x_data, positions)
        ax_position.set_xlim(0, max(100, len(positions)))
        
        pnl_line.set_data(x_data, pnls)
        ax_pnl.set_xlim(0, max(100, len(pnls)))
        if pnls:
            pnl_min, pnl_max = min(pnls), max(pnls)
            pnl_range = max(abs(pnl_min), abs(pnl_max), 1.0)
            ax_pnl.set_ylim(pnl_min - pnl_range * 0.1, pnl_max + pnl_range * 0.1)
        
        # Update title with current regime
        if regimes:
            current_regime = regimes[-1]
            ax_price.set_title(f"IDI Crypto Market Simulator - Regime: {current_regime.upper()}")
        
        return price_line, position_line, pnl_line

    # Create sliders
    ax_drift_bull = plt.axes([0.15, 0.25, 0.3, 0.03])
    ax_drift_bear = plt.axes([0.5, 0.25, 0.3, 0.03])
    ax_vol_base = plt.axes([0.15, 0.21, 0.3, 0.03])
    ax_shock_prob = plt.axes([0.5, 0.21, 0.3, 0.03])
    ax_fee_bps = plt.axes([0.15, 0.17, 0.3, 0.03])
    ax_seed = plt.axes([0.5, 0.17, 0.3, 0.03])
    
    drift_bull_slider = Slider(ax_drift_bull, "Drift Bull", -0.01, 0.01, valinit=params.drift_bull)
    drift_bear_slider = Slider(ax_drift_bear, "Drift Bear", -0.01, 0.0, valinit=params.drift_bear)
    vol_base_slider = Slider(ax_vol_base, "Vol Base", 0.001, 0.05, valinit=params.vol_base)
    shock_prob_slider = Slider(ax_shock_prob, "Shock Prob", 0.0, 0.2, valinit=params.shock_prob)
    fee_bps_slider = Slider(ax_fee_bps, "Fee (bps)", 0.0, 50.0, valinit=params.fee_bps)
    seed_slider = Slider(ax_seed, "Seed", 1, 1000, valinit=params.seed, valstep=1)

    def on_change(_val):
        """Handle slider changes."""
        params.drift_bull = drift_bull_slider.val
        params.drift_bear = drift_bear_slider.val
        params.vol_base = vol_base_slider.val
        params.shock_prob = shock_prob_slider.val
        params.fee_bps = fee_bps_slider.val
        params.seed = int(seed_slider.val)
        reset_market()

    for s in (drift_bull_slider, drift_bear_slider, vol_base_slider, 
              shock_prob_slider, fee_bps_slider, seed_slider):
        s.on_changed(on_change)

    ax_reset = plt.axes([0.82, 0.17, 0.1, 0.04])
    btn_reset = Button(ax_reset, "Reset")

    def on_reset(_event):
        """Handle reset button."""
        reset_market()

    btn_reset.on_clicked(on_reset)

    plt.connect("close_event", lambda _: None)
    import matplotlib.animation as animation
    animation.FuncAnimation(fig, update, frames=range(steps), interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    run_gui_sim()
