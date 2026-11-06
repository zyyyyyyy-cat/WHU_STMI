import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

path = Path(__file__).resolve().parents[1] / 'tools/generate_sam2_masks.py'
spec = importlib.util.spec_from_file_location('offline_sam2', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OfflineMaskTests(unittest.TestCase):
    def test_frozen_selection_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            selection = root / 'selection.json'
            name = 'train/IR/0001_c6.jpg'
            image = root / name
            image.parent.mkdir(parents=True)
            Image.new('RGB', (24, 36)).save(image)
            selection.write_text(json.dumps([name]), encoding='utf-8')
            self.assertEqual(module.load_selection(root, selection), [image])
            for names in ([name, name], ['../outside.jpg'], ['train/IR/./0001_c6.jpg'],
                          ['query/IR/0001_c6.jpg'], [1], []):
                selection.write_text(json.dumps(names), encoding='utf-8')
                with self.assertRaises(ValueError):
                    module.load_selection(root, selection)

    def test_prompts_and_selection(self):
        point, label, box = module.prompts(55, 176)
        np.testing.assert_array_equal(point, [[27, 87.5]])
        np.testing.assert_array_equal(label, [1])
        np.testing.assert_array_equal(box, [0, 0, 54, 175])
        masks = np.zeros((3, 176, 55), dtype=bool)
        masks[1, :80, :20] = True
        mask, score = module.choose_mask(masks, [.1, .9, .5], 176, 55)
        self.assertEqual(mask.shape, (176, 55))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertEqual(score, .9)
        with self.assertRaises(ValueError):
            module.choose_mask(masks, [.1, np.nan, .5], 176, 55)

    def test_stratified_train_only_and_native_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'images'
            for split in ('train', 'query', 'test'):
                for modality in ('RGB', 'IR', 'Thermal'):
                    for cam in (1, 6):
                        for index in range(2):
                            path = root / split / modality / ('0001_c{}_f{}.jpg'.format(cam, index))
                            path.parent.mkdir(parents=True, exist_ok=True)
                            Image.new('RGB', (55, 176)).save(path)
            selection = module.select_images(root, 1, 1234)
            self.assertEqual(len(selection), 6)
            self.assertTrue(all(p.relative_to(root).parts[0] == 'train' for p in selection))
            self.assertEqual(selection, module.select_images(root, 1, 1234))
            self.assertEqual(len(module.select_images(root, 0, 1234)), 36)
            mask = np.zeros((176, 55), dtype=np.uint8)
            mask[:80] = 255
            path = Path(directory) / 'mask.png'
            Image.fromarray(mask).save(path)
            module.validate_existing(path, (55, 176))
            with self.assertRaises(ValueError):
                module.validate_existing(path, (176, 55))
            module.save_preview(Image.new('RGB', (55, 176)), mask,
                                Path(directory) / 'preview.png', 'synthetic')
            with Image.open(path) as image:
                self.assertEqual(image.size, (55, 176))


if __name__ == '__main__':
    unittest.main(verbosity=2)
