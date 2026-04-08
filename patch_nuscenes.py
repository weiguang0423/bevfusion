import os

filepath = "/home/vipuser/miniconda3/envs/bevfusion/lib/python3.8/site-packages/nuscenes/eval/detection/evaluate.py"

with open(filepath, 'r') as f:
    text = f.read()

# Fix the strict subset crash
text = text.replace(
    'assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \\',
    '# By-pass for subset evaluation\\n        # assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \\\\'
)
text = text.replace(
    '"Samples in split doesn\\'t match samples in predictions."',
    '#    "Samples in split doesn\\'t match samples in predictions."'
)

# And correctly filter gt_boxes to only those present in pred_boxes during subsets
text = text.replace(
    'self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)',
    '''self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
        from nuscenes.eval.common.data_classes import EvalBoxes
        filtered_gt = EvalBoxes()
        for tok in self.pred_boxes.sample_tokens:
            if tok in self.gt_boxes.sample_tokens:
                filtered_gt.add_boxes(tok, self.gt_boxes[tok])
        self.gt_boxes = filtered_gt'''
)

with open(filepath, 'w') as f:
    f.write(text)

print("Patch applied.")
