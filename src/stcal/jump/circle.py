import random
from typing import Optional, Union

import numpy

RELATIVE_TOLERANCE = 1 + 1e-14


class Circle:
    def __init__(self, center: tuple[float, float], radius: float):
        self.center = numpy.array(center)
        self.radius = radius

    @classmethod
    def from_points(cls, points: list[tuple[float, float]]) -> 'Circle':
        """
        Returns the smallest circle that encloses all the given points.
        from https://www.nayuki.io/page/smallest-enclosing-circle

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
            return cls(center=points[0], radius=0.0)
        elif len(points) == 2:
            points = numpy.array(points)
            center = numpy.mean(points, axis=0)
            radius = numpy.max(numpy.hypot(*(center - points).T))

            return cls(center=center, radius=radius)
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
            return tuple(self.center)
        elif index == 1:
            return self.radius
        else:
            raise IndexError(f'{self.__class__.__name__} index out of range')

    def __add__(self, delta: tuple[float, float]) -> 'Circle':
        if isinstance(delta, float):
            delta = (delta, delta)
        return self.__class__(center=self.center + numpy.array(delta), radius=self.radius)

    def __mul__(self, factor: float) -> 'Circle':
        return self.__class__(center=self.center, radius=self.radius + factor)

    def __contains__(self, point: tuple[float, float]) -> bool:
        return numpy.hypot(*(numpy.array(point) - self.center)) <= self.radius * RELATIVE_TOLERANCE

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circle):
            return False
        return numpy.all(self.center == other.center) and self.radius == other.radius

    def almost_equals(self, other: 'Circle', delta: Optional[float] = None) -> bool:
        if delta is None:
            delta = RELATIVE_TOLERANCE
        return numpy.allclose(self.center, other.center, atol=delta) and \
            numpy.allclose(self.radius, other.radius, atol=delta)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.center}, {self.radius})'


def _expand_circle_from_one_point(
        known_boundary_point: tuple[float, float],
        points: list[tuple[float, float]],
) -> Circle:
    """
    iteratively expand a circle from one known boundary point to enclose the given set of points
    from https://www.nayuki.io/page/smallest-enclosing-circle
    """

    circle = Circle(known_boundary_point, 0.0)
    for point_index, point in enumerate(points):
        if point not in circle:
            if circle.radius == 0.0:
                circle = Circle.from_points([known_boundary_point, point])
            else:
                circle = _expand_circle_from_two_points(known_boundary_point, point, points[: point_index + 1])
    return circle


def _expand_circle_from_two_points(
        known_boundary_point_a: tuple[float, float],
        known_boundary_point_b: tuple[float, float],
        points: list[tuple[float, float]],
) -> Circle:
    """
    iteratively expand a circle from two known boundary points to enclose the given set of points
    from https://www.nayuki.io/page/smallest-enclosing-circle
    """

    known_boundary_points = numpy.array([known_boundary_point_a, known_boundary_point_b])

    circle = Circle.from_points(known_boundary_points)
    left = None
    right = None

    # For each point not in the two-point circle
    for point in points:
        if circle is None or point in circle:
            continue

        # Form a circumcircle and classify it on left or right side
        circumcircle_cross_product = _triangle_cross_product((*known_boundary_points, point))
        circumcircle = circumcircle_from_points(known_boundary_point_a, known_boundary_point_b, point)

        if circumcircle is None:
            continue

        circumcenter_cross_product = _triangle_cross_product((*known_boundary_points, circumcircle.center))

        if circumcircle_cross_product > 0.0 and \
                (
                        left is None or
                        circumcenter_cross_product > _triangle_cross_product((*known_boundary_points, left.center))
                ):
            left = circumcircle
        elif circumcircle_cross_product < 0.0 and \
                (
                        right is None or
                        circumcenter_cross_product < _triangle_cross_product((*known_boundary_points, right.center))
                ):
            right = circumcircle

    # Select which circle to return
    if left is None and right is None:
        return circle
    elif left is None:
        return right if right is not None else Circle(center=(0.0, 0.0), radius=0.0)
    elif right is None:
        return left
    else:
        return left if (left.radius <= right.radius) else right


def circumcircle_from_points(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
) -> Circle | None:
    """
    build a circumcircle from three points, using the algorithm from https://en.wikipedia.org/wiki/Circumscribed_circle
    from https://www.nayuki.io/page/smallest-enclosing-circle
    """

    points = numpy.array([a, b, c])
    incenter = ((numpy.min(points, axis=0) + numpy.max(points, axis=0)) / 2)

    relative_points = points - incenter
    a, b, c = relative_points

    intermediate = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if intermediate == 0:
        return None

    relative_circumcenter = numpy.array([
        (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1]) +
        (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1]) +
        (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1]),
        (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0]) +
        (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0]) +
        (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0]),
    ]) / intermediate

    return Circle(
        center=relative_circumcenter + incenter,
        radius=numpy.max(numpy.hypot(*(relative_circumcenter - relative_points).T)),
    )


def _triangle_cross_product(triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> float:
    """
    calculates twice the signed area of the provided triangle
    from https://www.nayuki.io/page/smallest-enclosing-circle

    :param triangle: three points defining a triangle
    :return: twice the signed area of triangle
    """

    return (triangle[1][0] - triangle[0][0]) \
        * (triangle[2][1] - triangle[0][1]) \
        - (triangle[1][1] - triangle[0][1]) \
        * (triangle[2][0] - triangle[0][0])
