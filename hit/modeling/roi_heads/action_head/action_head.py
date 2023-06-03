import torch

from .roi_action_feature_extractor import make_roi_action_feature_extractor
from .roi_action_predictors import make_roi_action_predictor
from .inference import make_roi_action_post_processor
from .loss import make_roi_action_loss_evaluator
from .metric import make_roi_action_accuracy_evaluator
from hit.modeling.utils import prepare_pooled_feature
from hit.utils.comm import all_reduce

from hit.structures.bounding_box import BoxList


class ROIActionHead(torch.nn.Module):
    """
    Generic Action Head class.
    """

    def __init__(self, cfg, dim_in):
        super(ROIActionHead, self).__init__()
        self.feature_extractor = make_roi_action_feature_extractor(cfg, dim_in)
        self.predictor = make_roi_action_predictor(cfg, self.feature_extractor.dim_out)
        self.post_processor = make_roi_action_post_processor(cfg)
        self.loss_evaluator = make_roi_action_loss_evaluator(cfg)
        self.accuracy_evaluator = make_roi_action_accuracy_evaluator(cfg)
        self.test_ext = cfg.TEST.EXTEND_SCALE
        self.cos_sim = torch.nn.CosineSimilarity()

    def forward(self, slow_features, fast_features, boxes, extras={}, part_forward=-1):
        # In training stage, boxes are from gt.
        # In testing stage, boxes are detected by human detector and proposals should be
        # enlarged boxes.
        assert not (self.training and part_forward >= 0)

        
        if part_forward == 1:
            boxes = extras["current_feat_p"]
            context = extras["current_feat_context"]
        else:
            context = None

        if self.training:
            proposals = self.loss_evaluator.sample_box(boxes)
        else:
            proposals = [box.extend(self.test_ext) for box in boxes]

        x, x_pooled, context_interation = self.feature_extractor(slow_features, fast_features, proposals, context, extras, part_forward)

        if part_forward == 0:
            pooled_feature = prepare_pooled_feature(x_pooled, boxes)
            
            return [pooled_feature, context_interation], {}, {}, {}

        action_logits = self.predictor(x)
        if not self.training:
            result = self.post_processor((action_logits,), boxes)
            return result, {}, {}, {}

        box_num = action_logits.size(0)
        box_num = torch.as_tensor([box_num], dtype=torch.float32, device=action_logits.device)
        all_reduce(box_num, average=True)

        loss_dict, loss_weight = self.loss_evaluator(
            [action_logits], box_num.item(),
        )
        
        metric_dict = self.accuracy_evaluator(
            [action_logits], proposals, box_num.item(),
        )

        pooled_feature = prepare_pooled_feature(x_pooled, proposals)

        return (
            [pooled_feature, context_interation],
            loss_dict,
            loss_weight,
            metric_dict,
        )

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    weight_map[new_key] = val
        return weight_map


def build_roi_action_head(cfg, dim_in):
    return ROIActionHead(cfg, dim_in)
