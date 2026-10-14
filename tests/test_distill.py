"""Unit tests for the pure-Python distillation trainer and compile integration."""

import random

import pytest

from strot.distill import (
    head_accuracy,
    head_predict,
    three_way_split,
    train_head,
)


class TestTrainer:
    def _separable(self, n=80, dim=8, seed=3):
        rng = random.Random(seed)
        embeddings, labels = [], []
        for _ in range(n):
            label = rng.randrange(2)
            center = 1.0 if label else -1.0
            embeddings.append([center + rng.gauss(0, 0.3) for _ in range(dim)])
            labels.append(label)
        return embeddings, labels

    def test_learns_separable_data(self):
        embeddings, labels = self._separable()
        weights, bias = train_head(embeddings, labels, n_classes=2)
        assert head_accuracy(embeddings, labels, weights, bias) >= 0.95

    def test_generalizes_to_fresh_points(self):
        train_e, train_y = self._separable(seed=3)
        test_e, test_y = self._separable(seed=99)
        weights, bias = train_head(train_e, train_y, n_classes=2)
        assert head_accuracy(test_e, test_y, weights, bias) >= 0.9

    def test_deterministic(self):
        embeddings, labels = self._separable()
        assert train_head(embeddings, labels, 2) == train_head(embeddings, labels, 2)

    def test_predict_is_distribution(self):
        embeddings, labels = self._separable(n=20)
        weights, bias = train_head(embeddings, labels, 2)
        probs = head_predict(embeddings[0], weights, bias)
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_three_way_split_disjoint_and_complete(self):
        train, measure, calibrate = three_way_split(range(100))
        assert not (set(train) & set(measure))
        assert not (set(train) & set(calibrate))
        assert not (set(measure) & set(calibrate))
        assert sorted(train + measure + calibrate) == list(range(100))
        assert len(train) == 50 and len(measure) == 25 and len(calibrate) == 25

    def test_guards(self):
        with pytest.raises(ValueError):
            train_head([], [], 2)
        with pytest.raises(ValueError):
            train_head([[1.0]], [0, 1], 2)
