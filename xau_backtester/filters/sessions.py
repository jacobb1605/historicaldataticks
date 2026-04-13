from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SessionFilter:
    """
    Simple UTC session filter.

    - `allowed_weekdays`: Monday=0 ... Sunday=6
    - `start_hour_utc` inclusive, `end_hour_utc` exclusive
    """

    allowed_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    start_hour_utc: int = 7
    end_hour_utc: int = 20

    def allows(self, t: pd.Timestamp) -> bool:
        tt = t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC")
        if int(tt.weekday()) not in self.allowed_weekdays:
            return False
        h = int(tt.hour)
        return self.start_hour_utc <= h < self.end_hour_utc

