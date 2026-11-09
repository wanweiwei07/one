"""SAM 3 (``transformers``): text-prompted instance segmentation, and the
object under a clicked point.

    seg = Segmenter()                    # ~20-45 s, once
    masks = seg.masks(bgr_image, 'leaf') # bool (n, h, w), best first

    pseg = PointSegmenter()              # ~5 s, once
    masks, scores = pseg.masks(bgr_image, (u, v))   # 3 candidates

The model (``facebook/sam3``) is gated on Hugging Face: it needs a login
with access. A prompt takes ~0.15 s on the GPU.
"""

import cv2
import numpy as np


SAM3_MODEL = 'facebook/sam3'
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


class Segmenter:
    """SAM 3, loaded once."""

    def __init__(self, model_id=SAM3_MODEL):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    def masks(self, image, prompt, threshold=SCORE_THRESHOLD,
              mask_threshold=MASK_THRESHOLD):
        """Bool masks (n, h, w) of every instance of ``prompt`` in the BGR
        ``image``, best first."""
        from PIL import Image
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=rgb, text=prompt,
                                return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=mask_threshold,
            target_sizes=inputs.get('original_sizes').tolist())[0]
        masks = result['masks'].cpu().numpy().astype(bool)
        order = np.argsort(-result['scores'].cpu().numpy())
        return masks[order]


class PointSegmenter:
    """SAM 3's interactive (SAM 2 style) part: the object under a clicked
    point, not every instance of a word -- it keeps an object's thin or
    odd parts that a text prompt can leave out. Loaded once (~5 s), a click
    ~0.12 s on the GPU."""

    def __init__(self, model_id=SAM3_MODEL):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Sam3TrackerModel.from_pretrained(model_id).to(
            self.device)
        self.model.eval()
        self.processor = Sam3TrackerProcessor.from_pretrained(model_id)

    def masks(self, image, point):
        """``(masks, scores)``: 3 candidate bool masks (3, h, w) of the
        object under ``point`` (u, v px) in the BGR ``image`` -- a part, the
        whole, a bigger whole; their order is not fixed -- and the model's
        IoU estimate of each."""
        from PIL import Image
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        u, v = float(point[0]), float(point[1])
        inputs = self.processor(images=rgb, input_points=[[[[u, v]]]],
                                input_labels=[[[1]]],
                                return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(
            out.pred_masks.cpu(), inputs['original_sizes'])[0][0].numpy()
        return masks > 0, out.iou_scores[0, 0].cpu().numpy()
