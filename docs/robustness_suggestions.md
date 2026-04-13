# Robustness suggestions

These are the highest-leverage improvements once the baseline engine is working end-to-end.

## Data & microstructure realism
- **Dynamic spread**: Use bid/ask from tick data to derive a time-varying spread (or at least session-dependent spread).\n
- **Slippage distribution**: Replace fixed slippage with a distribution conditioned on volatility/volume.\n
- **Commission/fees**: Add explicit commission per lot/contract if your broker charges it.

## Signal correctness
- **Intra-bar ordering model**: Current SL/TP resolution is conservative (worst-case when both hit). For more realism, use tick replay for exits.\n
- **FVG “fill” semantics**: Test multiple definitions: partial fill thresholds (e.g. 50% fill), wick vs close inside the gap, timeouts.\n
- **Structure definitions**: Compare swing-window structure vs ATR-based pivots vs fractals to see sensitivity.

## Overfitting control
- **Walk-forward**: Replace single split with rolling optimize/test windows.\n
- **Parameter stability**: Prefer parameter sets whose performance is stable across regimes (not just top score in-sample).\n
- **Purged CV**: Use embargo around the split to reduce leakage when features depend on rolling windows.

## Risk management
- **Multiple exits**: Scale out partial position at 1R and trail remainder.\n
- **Daily loss limit**: Stop trading after hitting a daily drawdown threshold.\n
- **Volatility scaling**: Size positions with ATR-based stop distances and volatility caps.

## Metrics and diagnostics
- **Distributional metrics**: Return skew/kurtosis, underwater duration, consecutive losses.\n
- **Trade attribution**: Break down PnL by session (London/NY), direction, structure regime.\n
- **Monte Carlo**: Shuffle trade outcomes (or bootstrapped resampling) to estimate performance uncertainty.

