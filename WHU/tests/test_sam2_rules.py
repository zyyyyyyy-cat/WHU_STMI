import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

tools = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(tools))
spec=importlib.util.spec_from_file_location('rule_comparison',tools/'compare_sam2_rules.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakePredictor:
    def __init__(self): self.embeddings=0
    def set_image(self,image): self.image=image; self.embeddings+=1
    def predict(self,**kwargs):
        h,w=self.image.shape[:2]; n=3 if kwargs['multimask_output'] else 1
        masks=np.zeros((n,h,w),dtype=bool)
        masks[:,h//4:3*h//4,w//4:3*w//4]=True
        if n>1: masks[1]=True
        return masks,np.linspace(.9,.7,n),None


class RuleTests(unittest.TestCase):
    def test_equivalence_check_rejects_pixel_and_condition_changes(self):
        from verify_mask_equivalence import compare
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated, reference = root/'generated', root/'reference'
            name = 'train/IR/0001_c6.jpg'
            rule = 'R08_box_center'
            conditions = dict(checkpoint_sha256='checkpoint', model_config='config', config_sha256='config-hash',
                              precision='bf16', sam2_commit='commit', torch_version='torch', cuda_version='cuda')
            generated.mkdir(); reference.mkdir()
            (generated/'provenance.json').write_text(json.dumps(dict(conditions, prompt_rule=rule)))
            (reference/'provenance.json').write_text(json.dumps(dict(conditions, selection=[name])))
            left = generated/Path(name).with_suffix('.png')
            right = reference/'masks'/(rule+'__iou')/Path(name).with_suffix('.png')
            for path in (left, right):
                path.parent.mkdir(parents=True)
                Image.new('L', (24, 36)).save(path)
            self.assertEqual(compare(generated, reference, rule)['identical'], 1)
            image = Image.open(left).copy(); image.putpixel((0, 0), 255); image.save(left)
            self.assertEqual(compare(generated, reference, rule)['mismatches'], [name])
            (generated/'provenance.json').write_text(json.dumps(dict(conditions, prompt_rule='R00_v1')))
            with self.assertRaises(ValueError):
                compare(generated, reference, rule)

    def test_production_prompts_and_mask_match_comparison(self):
        import generate_sam2_masks as production
        for size in ((55, 176), (24, 36), (391, 738)):
            image = Image.new('RGB', size)
            for rule in (r for r in module.rules() if r['name'] in production.PROMPT_RULES):
                expected = module.prepare_prompt(rule, size, (0, 0))
                point, label, box = production.prompts(*size, rule=rule['name'])
                for key, actual in [('point_coords', point), ('point_labels', label), ('box', box)]:
                    if expected[key] is None:
                        self.assertIsNone(actual)
                    else:
                        np.testing.assert_array_equal(actual, expected[key])
                predictor = FakePredictor()
                mask, score, index = production.predict_mask(predictor, image, 'fp32', rule['name'])
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    rows = module.run_image(FakePredictor(), image, Path('train/IR/0001_c6.jpg'),
                                            output, 'fp32', [rule])
                    with Image.open(output/'masks'/(rule['name']+'__iou')/'train/IR/0001_c6.png') as reference:
                        np.testing.assert_array_equal(mask, np.array(reference))
                    self.assertEqual(score, rows[0]['predicted_iou'])
                    self.assertEqual(index, rows[0]['selected_candidate'])

    def test_coordinates_and_rank(self):
        rules=module.rules(); self.assertEqual(len(rules),15)
        self.assertEqual(len({r['name'] for r in rules}),15)
        prompt=module.prepare_prompt(rules[0],(55,176),(21,21))
        np.testing.assert_allclose(prompt['point_coords'],[[48,108.5]])
        np.testing.assert_allclose(prompt['box'],[21,21,75,196])
        person=np.zeros((100,50),dtype=bool); person[10:90,15:35]=True
        full=np.ones_like(person)
        a=module.candidate_metrics(person,.9,rules[1])
        b=module.candidate_metrics(full,.98,rules[1])
        self.assertGreater(a['geometry_rank'],b['geometry_rank'])
        # Area/score metrics never edit a candidate or silently invert it.
        self.assertEqual(person.sum(),1600)

    def test_pipeline_embeddings_outputs_and_original_size(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory); image=Image.new('RGB',(24,36),color=(80,80,80))
            predictor=FakePredictor()
            rows=module.run_image(predictor,image,Path('train/IR/0001_c6.jpg'),output,'fp32',module.rules())
            self.assertEqual(predictor.embeddings,3)
            self.assertEqual(len(rows),30)
            self.assertEqual(len(list((output/'candidates').rglob('*.png'))),41)
            for path in (output/'masks').rglob('*.png'):
                with Image.open(path) as mask:
                    self.assertEqual(mask.size,image.size)
                    self.assertTrue(set(np.unique(np.array(mask))) <= {0,255})
            summary=module.summarize(rows)
            self.assertEqual(len(summary),30)
            self.assertTrue(all(key.endswith('/IR_A') for key in summary))
            self.assertTrue((output/'review/selected/train/IR/0001_c6.jpg').is_file())
            predictor=FakePredictor()
            rgb=module.run_image(predictor,image,Path('train/RGB/0001_c1.jpg'),output,'fp32',module.rules())
            self.assertEqual(predictor.embeddings,2)
            self.assertEqual(len(rgb),28)
            self.assertFalse(any('R14' in r['variant'] for r in rgb))


if __name__=='__main__': unittest.main(verbosity=2)
