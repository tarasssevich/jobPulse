"""Salary normalization domain service.

Converts diverse CIS vacancy salary representations into a unified, net
monthly salary in Belarusian Rubles (BYN).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

# Baseline CIS currency exchange rates relative to BYN (Belarusian Ruble)
DEFAULT_EXCHANGE_RATES_TO_BYN: Final[dict[str, float]] = {
    "BYN": 1.0,
    "BYR": 1.0,  # HeadHunter legacy code for Belarusian Ruble
    "RUR": 0.034,  # HeadHunter legacy code for Russian Ruble
    "RUB": 0.034,  # Standard ISO Russian Ruble
    "USD": 3.30,  # US Dollar
    "EUR": 3.55,  # Euro
    "KZT": 0.0068,  # Kazakhstani Tenge
}

# Standard 13% income tax shield factor for gross-to-net normalization
TAX_SHIELD_FACTOR: Final[float] = 0.87


def parse_salary_bound(val: Any) -> float | None:
    """Parse and validate numeric salary bound.

    Returns a positive finite float, or None if invalid/non-positive/non-finite.
    """
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        return fval if math.isfinite(fval) and fval > 0.0 else None
    if isinstance(val, str):
        cleaned = val.replace(" ", "").replace("\xa0", "").strip()
        if "," in cleaned and "." in cleaned:
            # Comma used as thousands separator, e.g. "2,500.50"
            cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) == 3 and parts[1].isdigit():
                # Thousands separator e.g. "2,500"
                cleaned = cleaned.replace(",", "")
            else:
                # Decimal comma e.g. "2500,50"
                cleaned = cleaned.replace(",", ".")
        try:
            fval = float(cleaned)
            return fval if math.isfinite(fval) and fval > 0.0 else None
        except (ValueError, TypeError):
            return None
    return None


def _parse_gross_flag(gross: Any) -> bool:
    """Parse gross pre-tax indicator safely."""
    if gross is True:
        return True
    if isinstance(gross, str):
        return gross.strip().lower() in ("true", "1", "yes")
    if isinstance(gross, int) and not isinstance(gross, bool):
        return gross == 1
    return False


class SalaryNormalizer:
    """Immutable salary normalization service.

    Calculates midpoint net salary converted to baseline BYN currency.
    """

    def __init__(
        self,
        exchange_rates: Mapping[str, float] | None = None,
        tax_shield_factor: float = TAX_SHIELD_FACTOR,
    ) -> None:
        """Initialize normalizer with currency rates and tax shield factor.

        Args:
            exchange_rates: Mapping of currency codes to BYN rate.
                If None, DEFAULT_EXCHANGE_RATES_TO_BYN is used.
            tax_shield_factor: Factor applied when salary is gross (default 0.87).

        Raises:
            ValueError: If tax_shield_factor is outside (0.0, 1.0] or non-finite.
        """
        rates = exchange_rates if exchange_rates is not None else DEFAULT_EXCHANGE_RATES_TO_BYN
        self._exchange_rates: Final[dict[str, float]] = {
            str(k).strip().upper(): float(v)
            for k, v in rates.items()
            if isinstance(k, str) and math.isfinite(float(v)) and float(v) > 0.0
        }
        factor = float(tax_shield_factor)
        if not (0.0 < factor <= 1.0) or not math.isfinite(factor):
            raise ValueError(
                f"tax_shield_factor must be in range (0.0, 1.0], got {tax_shield_factor}"
            )
        self._tax_shield_factor: Final[float] = factor

    @property
    def exchange_rates(self) -> dict[str, float]:
        """Return a copy of configured exchange rates."""
        return self._exchange_rates.copy()

    @property
    def tax_shield_factor(self) -> float:
        """Return configured gross-to-net tax factor."""
        return self._tax_shield_factor

    def normalize(
        self,
        salary_from: float | str | None = None,
        salary_to: float | str | None = None,
        currency: str | None = None,
        gross: bool | str | int | None = None,
    ) -> float | None:
        """Calculate net salary midpoint normalized to BYN.

        Rules:
            1. If both bounds are None, or currency is unknown/missing -> None.
            2. Midpoint:
               - both `from` and `to`: (from + to) / 2
               - only `from`: from
               - only `to`: to
               - if from > to: bounds are swapped before midpoint calculation
            3. Gross/Net:
               - if gross is True: net = amount * tax_shield_factor
               - if gross is False or None: net = amount
            4. Conversion to BYN:
               - byn = net * rate (rounded to 2 decimal places)

        Args:
            salary_from: Lower salary bound.
            salary_to: Upper salary bound.
            currency: Currency code (e.g. 'BYN', 'RUR', 'USD').
            gross: Whether salary is gross (pre-tax).

        Returns:
            Normalized net monthly midpoint in BYN, or None if invalid/missing.
        """
        # Validate currency
        if not currency or not isinstance(currency, str):
            return None

        curr_clean = currency.strip().upper()
        rate = self._exchange_rates.get(curr_clean)
        if rate is None or not math.isfinite(rate) or rate <= 0.0:
            return None

        # Parse bounds safely
        s_from = parse_salary_bound(salary_from)
        s_to = parse_salary_bound(salary_to)

        # Invert if bounds were entered backwards (e.g. from 5000 to 3000)
        if s_from is not None and s_to is not None and s_from > s_to:
            s_from, s_to = s_to, s_from

        # Compute midpoint
        if s_from is not None and s_to is not None:
            midpoint = (s_from + s_to) / 2.0
        elif s_from is not None:
            midpoint = s_from
        elif s_to is not None:
            midpoint = s_to
        else:
            return None

        if not math.isfinite(midpoint) or midpoint <= 0.0:
            return None

        # Apply gross-to-net tax shield if gross is True
        if _parse_gross_flag(gross):
            net_amount = midpoint * self._tax_shield_factor
        else:
            net_amount = midpoint

        # Convert to BYN
        salary_mid_byn = net_amount * rate
        if not math.isfinite(salary_mid_byn) or salary_mid_byn <= 0.0:
            return None

        return round(salary_mid_byn, 2)

    def normalize_dict(self, salary_data: Mapping[str, Any] | None) -> float | None:
        """Convenience method to normalize raw API salary dictionary.

        Args:
            salary_data: Dictionary typically containing 'from', 'to',
                'currency', and 'gross' keys.

        Returns:
            Normalized net monthly midpoint in BYN, or None.
        """
        if not salary_data or not isinstance(salary_data, Mapping):
            return None

        return self.normalize(
            salary_from=salary_data.get("from"),
            salary_to=salary_data.get("to"),
            currency=salary_data.get("currency"),
            gross=salary_data.get("gross"),
        )


def normalize_salary(
    salary_from: float | str | None = None,
    salary_to: float | str | None = None,
    currency: str | None = None,
    gross: bool | str | int | None = None,
    exchange_rates: Mapping[str, float] | None = None,
) -> float | None:
    """Pure functional interface for salary normalization.

    Args:
        salary_from: Lower salary bound.
        salary_to: Upper salary bound.
        currency: Currency code.
        gross: Gross flag.
        exchange_rates: Optional custom exchange rates to BYN.

    Returns:
        Normalized net monthly midpoint in BYN, or None.
    """
    normalizer = SalaryNormalizer(exchange_rates=exchange_rates)
    return normalizer.normalize(
        salary_from=salary_from,
        salary_to=salary_to,
        currency=currency,
        gross=gross,
    )
