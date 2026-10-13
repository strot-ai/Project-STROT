import math

import pytest

from strot.calibrate import (
    apply_temperature,
    calibration_report,
    conformal_quantile,
    fit_temperature,
    prediction_set,
)


class TestTemperature:
    def test_identity_at_t1(self):
        probs = [0.7, 0.2, 0.1]
        assert apply_temperature(probs, 1.0) == pytest.approx(probs)

    def test_high_t_flattens_low_t_sharpens(self):
        probs = [0.7, 0.2, 0.1]
        flat = apply_temperature(probs, 4.0)
        sharp = apply_temperature(probs, 0.25)
        assert flat[0] < probs[0] < sharp[0]
        assert sum(flat) == pytest.approx(1.0)
        assert sum(sharp) == pytest.approx(1.0)

    def test_fit_softens_overconfident_model(self):
        # Model says 0.9 but is right only ~half the time -> T should be > 1.
        distributions = [[0.9, 0.1]] * 10
        truths = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        assert fit_temperature(distributions, truths) > 1.0

    def test_fit_keeps_wellcalibrated_model_near_t1(self):
        distributions = [[0.9, 0.1]] * 10
        truths = [0] * 9 + [1]  # right 90% of the time at 0.9 confidence
        assert 0.6 <= fit_temperature(distributions, truths) <= 1.7


class TestConformal:
    def test_quantile_covers_easy_set(self):
        # Perfectly separable: true label always at 0.95.
        distributions = [[0.95, 0.05]] * 20
        truths = [0] * 20
        q = conformal_quantile(distributions, truths, 1.0, alpha=0.1)
        assert q <= 0.06  # scores all 0.05: tight bar

    def test_too_few_points_degrades_to_full_coverage(self):
        q = conformal_quantile([[0.9, 0.1]], [0], 1.0, alpha=0.05)
        assert q == 1.0  # n=1 cannot support alpha=0.05: everything conformal

    def test_singleton_dispatch_rule(self):
        included, calibrated = prediction_set([0.9, 0.08, 0.02], 1.0, 0.2)
        assert included == [0]
        ambiguous, _ = prediction_set([0.5, 0.45, 0.05], 1.0, 0.6)
        assert len(ambiguous) > 1  # ambiguous record -> fallback at runtime

    def test_report_shape(self):
        distributions = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]
        truths = [0, 1, 0]
        t = fit_temperature(distributions, truths)
        q = conformal_quantile(distributions, truths, t, alpha=0.25)
        report = calibration_report(distributions, truths, t, q, 0.25)
        assert report["n_calibration"] == 3
        assert 0.0 <= report["dispatch_rate_on_calibration"] <= 1.0
        assert report["quantile_granularity"] == 0.25


class TestGuards:
    def test_bad_temperature(self):
        with pytest.raises(ValueError):
            apply_temperature([0.5, 0.5], 0.0)

    def test_bad_alpha(self):
        with pytest.raises(ValueError):
            conformal_quantile([[0.5, 0.5]], [0], 1.0, alpha=1.5)

    def test_mismatched_inputs(self):
        with pytest.raises(ValueError):
            fit_temperature([[0.5, 0.5]], [0, 1])
