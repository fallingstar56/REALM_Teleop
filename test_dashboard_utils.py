import unittest
import pandas as pd
from dashboard_utils import get_completed_experiments, filter_videos, SUPPORTED_TASKS, SUPPORTED_PERTURBATIONS

class TestDashboardUtils(unittest.TestCase):
    def test_get_completed_experiments(self):
        # Create a dummy dataframe
        data = {
            'task': ['put_green_block_in_bowl', 'put_green_block_in_bowl', 'rotate_marker'],
            'perturbation': ["['Default']", "['Default']", "['V-AUG']"]
        }
        df = pd.DataFrame(data)

        # Test with repeats=2
        completed = get_completed_experiments(df, required_repeats=2)
        self.assertIn(('put_green_block_in_bowl', 'Default'), completed)
        self.assertNotIn(('rotate_marker', 'V-AUG'), completed)

        # Test with repeats=1
        completed = get_completed_experiments(df, required_repeats=1)
        self.assertIn(('put_green_block_in_bowl', 'Default'), completed)
        self.assertIn(('rotate_marker', 'V-AUG'), completed)

    def test_filter_videos(self):
        videos = [
            "/path/to/2024_rollout_put_green_block_in_bowl_Default_0.mp4",
            "/path/to/2024_rollout_put_green_block_in_bowl_V-AUG_0.mp4",
            "/path/to/2024_rollout_rotate_marker_Default_0.mp4"
        ]

        # Test empty filter (should return all)
        self.assertEqual(filter_videos(videos, []), videos)

        # Test filtering
        filters = [('put_green_block_in_bowl', 'Default')]
        filtered = filter_videos(videos, filters)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], videos[0])

        # Test multiple filters
        filters = [
            ('put_green_block_in_bowl', 'V-AUG'),
            ('rotate_marker', 'Default')
        ]
        filtered = filter_videos(videos, filters)
        self.assertEqual(len(filtered), 2)
        self.assertIn(videos[1], filtered)
        self.assertIn(videos[2], filtered)

if __name__ == '__main__':
    unittest.main()
