"""Behavior checks: leakage, uncertain labels, and real image batch loading."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceguard.data import BASE_FIELDS, FaceImageDataset, validate_manifest, write_manifest
from PIL import Image
from torch.utils.data import DataLoader


class ManifestTests(unittest.TestCase):
    def row(self, **overrides):
        row = dict.fromkeys(BASE_FIELDS, '')
        row.update(sample_id='sample1', path='one.png', label='0', dataset='fixture',
                   group_id='fixture:person1', split='train', media_type='image',
                   provenance_status='verified')
        row.update(overrides)
        return row

    def test_group_cannot_cross_splits_even_between_datasets(self):
        rows = [self.row(), self.row(sample_id='sample2', path='two.png',
                                    dataset='derived', split='test')]
        with self.assertRaisesRegex(ValueError, 'group_id leakage'):
            validate_manifest(rows)

    def test_shared_source_cannot_cross_different_groups(self):
        rows = [self.row(target_id='001'), self.row(sample_id='sample2',
            path='two.png', group_id='other', source_id='001', split='val')]
        with self.assertRaisesRegex(ValueError, 'source videos'):
            validate_manifest(rows)

    def test_related_frames_in_same_split_are_allowed(self):
        validate_manifest([self.row(), self.row(sample_id='frame2', path='two.png')])

    def test_repeated_image_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_manifest([self.row(), self.row(sample_id='frame2')])

    def test_real_rgb_batch_and_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (32, 16), (255, 0, 0)).save(root / 'one.png')
            manifest = root / 'manifest.csv'
            write_manifest(manifest, [self.row(label='1')], BASE_FIELDS)
            data = FaceImageDataset(manifest, 'train', root=root)
            batch = next(iter(DataLoader(data, batch_size=1)))
            self.assertEqual(tuple(batch['image'].shape), (1, 3, 256, 256))
            self.assertEqual(batch['image'][0, 0, 0, 0].item(), 1)
            self.assertEqual(batch['image'][0, 2, 0, 0].item(), 0)
            self.assertEqual(batch['label'].item(), 1)

    def test_unverified_data_and_videos_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.csv'
            write_manifest(manifest, [self.row(provenance_status='unverified')], BASE_FIELDS)
            with self.assertRaisesRegex(ValueError, 'Unverified'):
                FaceImageDataset(manifest, 'train')
            write_manifest(manifest, [self.row(media_type='video')], BASE_FIELDS)
            with self.assertRaisesRegex(ValueError, 'requires images'):
                FaceImageDataset(manifest, 'train')


if __name__ == '__main__':
    unittest.main()
