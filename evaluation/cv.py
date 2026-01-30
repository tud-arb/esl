from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class TimeSeriesCV:
    n_splits: int = 5
    test_size: int = 10_000
    gap: int = 0
    train_size: int | None = None  # if None => expanding window

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if self.n_splits <= 0:
            raise ValueError("n_splits must be > 0")
        if self.test_size <= 0:
            raise ValueError("test_size must be > 0")
        if self.gap < 0:
            raise ValueError("gap must be >= 0")
        if self.train_size is not None and self.train_size <= 0:
            raise ValueError("train_size must be > 0 or None")

        # We place the last split's test window at the end, and step backwards.
        # This yields splits that are all valid and comparable.
        total_test = self.n_splits * self.test_size
        min_train_needed = 1 if self.train_size is None else self.train_size

        if n_samples < (min_train_needed + self.gap + total_test):
            raise ValueError(
                f"Not enough samples ({n_samples}) for "
                f"n_splits={self.n_splits}, test_size={self.test_size}, gap={self.gap}, "
                f"train_size={self.train_size}. Try reducing test_size / n_splits / gap."
            )

        # Define the end of the final test window
        end = n_samples

        for k in range(self.n_splits):
            test_end = end - k * self.test_size
            test_start = test_end - self.test_size

            train_end = test_start - self.gap
            if self.train_size is None:
                train_start = 0
            else:
                train_start = train_end - self.train_size

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx


def time_series_cv(
    n_samples: int,
    n_splits: int = 5,
    test_size: int = 10_000,
    gap: int = 0,
    train_size: int | None = None,
):
    """
    Convenience generator so you can do:
      for train_idx, test_idx in time_series_cv(len(df), ...):
          ...
    """
    splitter = TimeSeriesCV(
        n_splits=n_splits, test_size=test_size, gap=gap, train_size=train_size
    )
    yield from splitter.split(n_samples)
