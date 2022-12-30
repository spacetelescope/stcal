# from https://www.nayuki.io/page/smallest-enclosing-circle

import math
import random
from typing import Union, Tuple, List

import numpy


class Circle:
    RELATIVE_TOLERANCE = 1 + 1e-14

    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = center
        self.radius = radius

    @classmethod
    def from_points(cls, points: List[Tuple[float, float]]) -> 'Circle':
        """
        Returns the smallest circle that encloses all the given points.

        If 0 points are given, `None` is returned.
        If 1 point is given, a circle of radius 0 is returned.
        If 2 points are given, a circle with diameter between them is returned.
        If 3 or more points are given, uses the algorithm described in this PDF:
        https://www.cise.ufl.edu/~sitharam/COURSES/CG/kreveldnbhd.pdf

        :param points: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
        :return: the smallest circle that encloses all points, to within the relative tolerance defined by the class
        """

        if len(points) == 0:
            return None
        elif len(points) == 1:
            return Circle(points[0], 0.0)
        elif len(points) == 2:
            a, b = points

            cx = (a[0] + b[0]) / 2
            cy = (a[1] + b[1]) / 2
            r0 = math.hypot(cx - a[0], cy - a[1])
            r1 = math.hypot(cx - b[0], cy - b[1])
            return cls((cx, cy), max(r0, r1))
        else:
            # Convert to float and randomize order
            shuffled_points = [(float(point[0]), float(point[1])) for point in points]
            random.shuffle(shuffled_points)

            # Progressively add points to circle or recompute circle
            circle = None
            for (index, point) in enumerate(shuffled_points):
                if circle is None or point not in circle:
                    circle = _expand_circle_from_one_point(point, shuffled_points[:index + 1])

            return circle

    def __getitem__(self, index: int) -> Union[tuple, float]:
        if index == 0:
            return self.center
        elif index == 1:
            return self.radius
        else:
            raise IndexError(f'{self.__class__.__name__} index out of range')

    def __add__(self, delta: Tuple[float, float]) -> 'Circle':
        if isinstance(delta, float):
            delta = [delta, delta]
        return self.__class__((self.center[0] + delta[0], self.center[1] + delta[1]), self.radius)

    def __mul__(self, factor: float) -> 'Circle':
        return self.__class__(self.center, self.radius + factor)

    def __contains__(self, point: Tuple[float, float]):
        return math.hypot(point[0] - self.center[0], point[1] - self.center[1]) <= self.radius * self.RELATIVE_TOLERANCE

    def __eq__(self, other: 'Circle') -> bool:
        return self.center[0] == other.center[0] and self.center[1] == other.center[1] and self.radius == other.radius

    def almost_equals(self, other: 'Circle', delta: float = None) -> bool:
        if delta is None:
            delta = self.RELATIVE_TOLERANCE
        return numpy.allclose([self.center[0], self.center[1], self.radius],
                              [other.center[0], other.center[1], other.radius], atol=delta)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.center}, {self.radius})'


def _expand_circle_from_one_point(
        a: Tuple[float, float],
        points: List[Tuple[float, float]],
) -> Circle:
    """
    One boundary point known
    """

    circle = Circle(a, 0.0)
    for (b_index, b) in enumerate(points):
        if b not in circle:
            if circle.radius == 0.0:
                circle = Circle.from_points([a, b])
            else:
                circle = _expand_circle_from_two_points(a, b, points[: b_index + 1])
    return circle


def _expand_circle_from_two_points(
        a: Tuple[float, float],
        b: Tuple[float, float],
        points: List[Tuple[float, float]],
) -> Circle:
    """
    Two boundary points known
    """

    circ = Circle.from_points([a, b])
    left = None
    right = None
    px, py = a
    qx, qy = b

    # For each point not in the two-point circle
    for r in points:
        if r in circ:
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(((px, py), (qx, qy), (r[0], r[1])))
        c = circumcircle(a, b, r)
        cross_2 = _cross_product(((px, py), (qx, qy), (c.center[0], c.center[1])))
        if c is None:
            continue
        elif cross > 0.0 and (left is None or cross_2 > _cross_product(((px, py), (qx, qy), left.center))):
            left = c
        elif cross < 0.0 and (right is None or cross_2 < _cross_product(((px, py), (qx, qy), right.center))):
            right = c
        # cross = _cross_product(((px, py), (qx, qy), (r[0], r[1])))
        # c = Circle.from_points([p, q, r])
        # if c is None:
        #     continue
        # elif cross > 0.0 and (
        #         left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0],
        #                                                                                     left[1])):
        #     left = c
        # elif cross < 0.0 and (
        #         right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0],
        #                                                                                      right[1])):
        #     right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left.radius <= right.radius) else right


def circumcircle(
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
) -> Circle:
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
    oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2
    ax = a[0] - ox
    ay = a[1] - oy
    bx = b[0] - ox
    by = b[1] - oy
    cx = c[0] - ox
    cy = c[1] - oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    ra = math.hypot(x - a[0], y - a[1])
    rb = math.hypot(x - b[0], y - b[1])
    rc = math.hypot(x - c[0], y - c[1])
    return Circle((x, y), max(ra, rb, rc))


def _cross_product(triangle: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> float:
    """
    :param triangle: three points defining a triangle
    :return: twice the signed area of triangle
    """

    return (triangle[1][0] - triangle[0][0]) \
        * (triangle[2][1] - triangle[0][1]) \
        - (triangle[1][1] - triangle[0][1]) \
        * (triangle[2][0] - triangle[0][0])
