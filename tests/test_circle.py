# from https://www.nayuki.io/page/smallest-enclosing-circle

import random
from typing import Tuple, List

import pytest

from stcal.jump.circle import Circle

RELATIVE_TOLERANCE = 1e-12


@pytest.mark.parametrize('trial', range(10))
def test_circle_matching_naive_algorithm(trial):
    points = _random_points(random.randint(1, 30))

    reference_circle = _smallest_enclosing_circle_naive(points)
    test_circle = Circle.from_points(points)

    assert test_circle.almost_equals(reference_circle, delta=RELATIVE_TOLERANCE)


@pytest.mark.parametrize('trial', range(10))
def test_circle_translation(trial):
    points = _random_points(random.randint(1, 300))

    test_circle = Circle.from_points(points)

    dx = random.gauss(0, 1)
    dy = random.gauss(0, 1)
    translated_points = [(x + dx, y + dy) for (x, y) in points]

    translated_circle = Circle.from_points(translated_points)
    reference_circle = test_circle + (dx, dy)

    assert translated_circle.almost_equals(reference_circle, delta=RELATIVE_TOLERANCE)


@pytest.mark.parametrize('trial', range(10))
def test_circle_scaling(trial):
    points = _random_points(random.randint(1, 300))

    test_circle = Circle.from_points(points)

    scale = random.gauss(0, 1)
    scaled_points = [(x * scale, y * scale) for (x, y) in points]

    scaled_circle = Circle.from_points(scaled_points)
    reference_circle = Circle((test_circle.center[0] * scale, test_circle.center[1] * scale),
                              test_circle.radius * abs(scale))

    assert scaled_circle.almost_equals(reference_circle, delta=RELATIVE_TOLERANCE)


def _random_points(n: int) -> List[Tuple[float, float]]:
    if random.random() < 0.2:  # Discrete lattice (to have a chance of duplicated points)
        return [(random.randrange(10), random.randrange(10)) for _ in range(n)]
    else:  # Gaussian distribution
        return [(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(n)]


def _smallest_enclosing_circle_naive(points: List[Tuple[float, float]]) -> Circle:
    """
    Returns the smallest enclosing circle in O(n^4) time using the naive algorithm.
    """

    # Degenerate cases
    if len(points) == 0:
        return None
    elif len(points) == 1:
        return Circle(points[0], 0.0)

    # Try all unique pairs
    result = None
    for i in range(len(points)):
        p = points[i]
        for j in range(i + 1, len(points)):
            q = points[j]
            c = Circle.from_points([p, q])
            if (result is None or c.radius < result.radius) and all(r in c for r in points):
                result = c
    if result is not None:
        return result  # This optimization is not mathematically proven

    # Try all unique triples
    for i in range(len(points)):
        p = points[i]
        for j in range(i + 1, len(points)):
            q = points[j]
            for k in range(j + 1, len(points)):
                r = points[k]
                c = Circle.from_points([p, q, r])
                if c is not None and (result is None or c.radius < result.radius) and all(s in c for s in points):
                    result = c

    if result is None:
        raise AssertionError()
    return result
