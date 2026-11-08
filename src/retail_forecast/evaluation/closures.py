"""Calendar rules known before forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def is_known_closed(
    dates: pd.Series | Sequence[object],
    config: Mapping[str, Any],
) -> pd.Series:
    """Identify Saturdays and explicitly configured closures."""
    values = pd.to_datetime(pd.Series(dates)).dt.normalize()
    closure_dates = pd.to_datetime(
        config["aggregation"].get("closure_dates", [])
    ).normalize()
    return pd.Series(
        values.dt.dayofweek.eq(5) | values.isin(closure_dates),
        index=values.index,
        dtype=bool,
    )


def is_reliable_after_closure(
    dates: pd.Series | Sequence[object],
    config: Mapping[str, Any],
) -> pd.Series:
    """Marks dates whose post-closure level is not validated as unreliable."""
    values = pd.to_datetime(pd.Series(dates)).dt.normalize()
    configured = sorted(
        pd.to_datetime(config["aggregation"].get("closure_dates", [])).normalize().unique()
    )
    reliable = pd.Series(True, index=values.index, dtype=bool)
    if not configured or values.empty:
        return reliable

    windows: list[list[pd.Timestamp]] = []
    current: list[pd.Timestamp] = [configured[0]]
    for date in configured[1:]:
        if date == current[-1] + pd.Timedelta(days=1):
            current.append(date)
        else:
            windows.append(current)
            current = [date]
    windows.append(current)

    for window in windows:
        last_closed = window[-1]
        if values.isin(window).any():
            reliable &= values.le(last_closed)
    return reliable
