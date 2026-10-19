"""Run with python inference/Utils/test_generate_attention.py; no model checkpoints needed."""
import unittest
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from generate_attention import ensemble_attention_frames, write_plot_data


class EnsembleAttentionTests(unittest.TestCase):
    def frame(self, fold, values=(1.0, 3.0)):
        return pd.DataFrame({
            'SAMPLE_ID': ['slide', 'slide'],
            'pred_map_location': ['(0, 1, 0, 1)', '(1, 2, 0, 1)'],
            'tumor_fraction': [0.2, 0.8],
            'att': values,
            'FOLD': fold,
        })

    def test_coordinates_align_even_when_fold_rows_are_shuffled(self):
        result = ensemble_attention_frames([
            self.frame(0), self.frame(1, (5.0, 7.0)).iloc[::-1]
        ], [0, 1])
        self.assertEqual(result['mean_att'].tolist(), [3.0, 5.0])
        self.assertEqual(result['n_folds'].tolist(), [2, 2])
        self.assertEqual(result['n_att_values'].tolist(), [2, 2])

    def test_excluded_tiles_stay_missing_and_do_not_count_as_scored(self):
        result = ensemble_attention_frames([
            self.frame(0, (np.nan, 2.0)), self.frame(1, (np.nan, 6.0))
        ], [0, 1])
        self.assertTrue(pd.isna(result['mean_att'].iloc[0]))
        self.assertEqual(result['n_folds'].tolist(), [0, 2])
        self.assertEqual(result['n_att_values'].tolist(), [0, 2])

    def test_missing_fold_for_slide_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fold outputs'):
            ensemble_attention_frames([self.frame(0)], [0, 1])

    def test_mismatched_tile_coordinates_are_rejected(self):
        second = self.frame(1)
        second.loc[0, 'pred_map_location'] = '(8, 9, 0, 1)'
        with self.assertRaisesRegex(ValueError, 'partial ensemble'):
            ensemble_attention_frames([self.frame(0), second], [0, 1])

    def test_infinite_attention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            ensemble_attention_frames([self.frame(0, (np.inf, 2.0))], [0])

    def test_duplicate_fold_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'distinct folds'):
            ensemble_attention_frames([self.frame(0)], [0, 0])

    def test_duplicate_tile_keys_follow_original_grouping(self):
        first = self.frame(0)
        first = pd.concat([first, first.iloc[[0]]], ignore_index=True)
        result = ensemble_attention_frames([first, self.frame(1, (7.0, 9.0))], [0, 1])
        self.assertEqual(result['mean_att'].iloc[0], 3.0)
        self.assertEqual(result['n_att_values'].iloc[0], 3)
        self.assertEqual(result['n_folds'].iloc[0], 2)


class FinalOutputTests(unittest.TestCase):
    def test_only_plot_data_csv_is_saved_before_rendering(self):
        frame = pd.DataFrame({
            'SAMPLE_ID': ['slide', 'slide', 'slide'],
            'pred_map_location': ['(0, 1, 0, 1)', '(1, 2, 0, 1)', '(2, 3, 0, 1)'],
            'tumor_fraction': [0.2, 0.8, 0.0],
            'mean_att': [-2.0, 2.0, np.nan],
            'n_folds': [5, 5, 0], 'n_att_values': [5, 5, 0],
        })
        with tempfile.TemporaryDirectory() as d:
            paths = write_plot_data(frame, Path(d))
            self.assertEqual([p.name for p in paths], ['slide_ensemble_attention_plot_data.csv'])
            self.assertEqual(len(list(Path(d).rglob('*.*'))), 1)
            self.assertEqual(paths[0].parent, Path(d) / 'ensemble_attention')
            data = pd.read_csv(paths[0])
            self.assertEqual(data.attention_heatmap_value.tolist(), [0.0, 1.0])
            self.assertEqual(data.mean_att.tolist(), [-2.0, 2.0])
            self.assertEqual(data.x1.tolist(), [0.0, 1.0])
            self.assertEqual(data.n_folds.tolist(), [5, 5])

    def test_constant_attention_uses_manuscript_midpoint(self):
        frame = pd.DataFrame({'SAMPLE_ID': ['s'], 'pred_map_location': ['(0, 1, 0, 1)'],
                              'mean_att': [2.0]})
        with tempfile.TemporaryDirectory() as d:
            path = write_plot_data(frame, Path(d))[0]
            self.assertEqual(pd.read_csv(path).attention_heatmap_value.iloc[0], 0.5)

    def test_invalid_tile_bounds_are_rejected(self):
        frame = pd.DataFrame({'SAMPLE_ID': ['s'], 'pred_map_location': ['(2, 1, 0, 1)'],
                              'mean_att': [2.0]})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'bounds'):
                write_plot_data(frame, Path(d))


if __name__ == '__main__':
    unittest.main()
