from collections.abc import Mapping

from mtf_calc.models import BarSection, FitResult, MtfResult


def compute(results: Mapping[BarSection, FitResult]) -> MtfResult:
    del results
    raise NotImplementedError
