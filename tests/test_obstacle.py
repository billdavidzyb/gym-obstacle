from unittest import TestCase

import numpy as np
from obstacle.obstacle import Segment

class TestObstacle(TestCase):
    def test_segment(self):
        seg_h = Segment([1, 1], [2, 2])
        self.assertEqual(seg_h.diff_x(), 1)
        self.assertEqual(seg_h.diff_y(), 1)
        # cross
        seg1 = Segment([0, 0], [3, 3])
        seg2 = Segment([0, 2], [2, 0])
        print(seg1.get_intersection(seg2))
        print(seg2.get_intersection(seg1))
        assert np.array_equal(seg1.get_intersection(seg2), seg2.get_intersection(seg1))
        seg1 = Segment([0, 3], [0, -1])
        seg2 = Segment([1, 0], [-1, 0])
        assert np.array_equal(seg1.get_intersection(seg2), seg2.get_intersection(seg1))
        # T
        seg1 = Segment([0, 0], [2, 0])
        seg2 = Segment([1, 1], [1, 2])
        assert seg1.get_intersection(seg2) == seg2.get_intersection(seg1) == None
        seg1 = Segment([0, 0], [0, 2])
        seg2 = Segment([1, 1], [2, 1])
        assert seg1.get_intersection(seg2) == seg2.get_intersection(seg1) == None
        # parallel
        seg1 = Segment([0, 0], [1, 0])
        seg2 = Segment([0, 1], [1, 1])
        assert seg1.get_intersection(seg2) == seg2.get_intersection(seg1) == None



