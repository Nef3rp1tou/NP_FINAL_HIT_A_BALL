

from FINAL import shooting_method_2d, detect_targets, Animator, main

import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import math


class TestProjectileMotion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple test image with a circle
        self.test_image = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(self.test_image, (200, 200), 30, 255, -1)

        # Common test parameters
        self.origin = (0.0, 0.0)
        self.g = 9.81

    def test_detect_targets(self):
        """Test circle detection functionality"""
        # Test with empty image
        empty_image = np.zeros((400, 400), dtype=np.uint8)
        self.assertEqual(detect_targets(empty_image), [])

        # Test with single circle
        targets = detect_targets(self.test_image)
        self.assertEqual(len(targets), 1)
        x, y = targets[0]
        self.assertAlmostEqual(x, 200, delta=5)
        self.assertAlmostEqual(y, 200, delta=5)

        # Test with multiple circles
        multi_circle_image = self.test_image.copy()
        cv2.circle(multi_circle_image, (100, 100), 30, 255, -1)
        targets = detect_targets(multi_circle_image)
        self.assertEqual(len(targets), 2)

    def test_shooting_method_2d(self):
        """Test trajectory calculation"""
        # Test case 1: Target at reasonable distance and height
        v0, theta = shooting_method_2d(
            target=(10.0, 5.0),
            origin=(0.0, 0.0),
            g=9.81,
            v0_range=(1.0, 100.0),
            theta_range_right=(5, 80),
            theta_range_left=(100, 175)
        )
        self.assertIsNotNone(v0)
        self.assertIsNotNone(theta)
        self.assertTrue(0 < theta < 90)

        # Test case 2: Target to the left at reasonable distance
        v0, theta = shooting_method_2d(
            target=(-10.0, 5.0),
            origin=(0.0, 0.0),
            g=9.81,
            v0_range=(1.0, 100.0),
            theta_range_right=(5, 80),
            theta_range_left=(100, 175)
        )
        self.assertIsNotNone(v0)
        self.assertIsNotNone(theta)
        self.assertTrue(90 < theta < 180)

        # Test case 3: Target extremely far (should be unreachable)
        v0, theta = shooting_method_2d(
            target=(10000.0, 10000.0),
            origin=(0.0, 0.0),
            g=9.81,
            v0_range=(1.0, 100.0),  # Limited velocity range
            theta_range_right=(5, 80),
            theta_range_left=(100, 175)
        )
        self.assertIsNone(v0)
        self.assertIsNone(theta)

    def test_animator(self):
        """Test animation functionality"""
        fig, ax = plt.subplots()

        # Create a simple trajectory
        t = np.linspace(0, 1, 100)
        x_vals = t * 10
        y_vals = 5 * t - 4.905 * t ** 2
        trajectories = [(x_vals, y_vals)]

        animator = Animator(fig, ax, trajectories, (0, 0))

        # Test initialization
        self.assertIsNotNone(animator.anim)
        self.assertEqual(len(animator.trajectories), 1)

        # Test frame generator
        gen = animator.frame_generator()
        first_frame = next(gen)
        self.assertEqual(len(first_frame), 2)
        self.assertIsInstance(first_frame[0], (int, float))
        self.assertIsInstance(first_frame[1], (int, float))

    @patch('builtins.input')
    def test_main_manual_input(self, mock_input):
        """Test main function with manual target input"""
        # Mock user inputs
        mock_inputs = [
            '10',  # grid width
            '10',  # grid height
            'n',  # don't use image
            '2',  # number of targets
            '5',  # target 1 x
            '5',  # target 1 y
            '7',  # target 2 x
            '3'  # target 2 y
        ]
        mock_input.side_effect = mock_inputs

        # Mock matplotlib to prevent display
        with patch('matplotlib.pyplot.show'):
            main()

    @patch('builtins.input')
    def test_main_image_input(self, mock_input):
        """Test main function with image input"""
        # Mock user inputs
        mock_inputs = [
            '10',  # grid width
            '10',  # grid height
            'y',  # use image
            'test.png'  # image path
        ]
        mock_input.side_effect = mock_inputs

        # Create test image file
        cv2.imwrite('test.png', self.test_image)

        # Mock matplotlib to prevent display
        with patch('matplotlib.pyplot.show'):
            main()

        # Clean up
        import os
        os.remove('test.png')

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with invalid image
        invalid_image = np.zeros((1, 1), dtype=np.uint8)
        self.assertEqual(detect_targets(invalid_image), [])

        # Test with very high target (should be unreachable with given velocity range)
        v0, theta = shooting_method_2d(
            target=(5.0, 1000.0),
            origin=(0.0, 0.0),
            g=9.81,
            v0_range=(1.0, 100.0),  # Limited velocity range
            theta_range_right=(5, 80),
            theta_range_left=(100, 175)
        )
        self.assertIsNone(v0)
        self.assertIsNone(theta)

        # Test with zero gravity (special case)
        v0, theta = shooting_method_2d(
            target=(10.0, 10.0),
            origin=(0.0, 0.0),
            g=0,
            v0_range=(1.0, 100.0),
            theta_range_right=(5, 80),
            theta_range_left=(100, 175)
        )
        self.assertIsNotNone(v0)
        self.assertIsNotNone(theta)


if __name__ == '__main__':
    unittest.main(verbosity=2)