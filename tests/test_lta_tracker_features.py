from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

from XTA import lta_tracker_features as subject


class _Nested:
    def __init__(self, tensors, mask=None):
        self.tensors = tensors
        self.mask = mask


class _Backbone:
    training = False

    def __init__(self):
        self.calls = []

    def forward_image(self, image, **kwargs):
        self.calls.append((image, kwargs))
        features = {}
        for branch_index, key in enumerate(("interactive", "sam2_backbone_out")):
            pyramid = [
                _Nested(image[:, :1] + (branch_index + level + 1) * 0.0031)
                for level in range(3)
            ]
            features[key] = {
                "vision_mask": None,
                "vision_features": pyramid[-1].tensors,
                "backbone_fpn": pyramid,
                "vision_pos_enc": [
                    torch.full_like(item.tensors, 0.0017 * (level + 1), dtype=torch.float32)
                    for level, item in enumerate(pyramid)
                ],
            }
        return features


@unittest.skipIf(torch is None or not getattr(torch, "__file__", None), "requires real Torch")
class TrackerFeatureTests(unittest.TestCase):
    def _case(self):
        projection_calls = []

        def decoder(name):
            def project(level, scale, tensor):
                projection_calls.append((name, level, tensor.dtype))
                # Keep CPU output in FP32 to expose moving the BF16 cast after
                # the projection, which is not the pinned cache contract.
                return tensor.float() * scale
            return SimpleNamespace(
                conv_s0=lambda tensor: project(0, 1.3, tensor),
                conv_s1=lambda tensor: project(1, 0.7, tensor),
            )

        backbone = _Backbone()
        detector = SimpleNamespace(backbone=backbone, device=torch.device("cpu"))
        original = mock.Mock(side_effect=AssertionError("grounding fallback executed"))
        model = SimpleNamespace(
            detector=detector,
            tracker=SimpleNamespace(
                interactive_sam_mask_decoder=decoder("interactive"),
                sam_mask_decoder=decoder("propagation"),
            ),
            training=False,
            _prepare_backbone_feats=original,
        )
        images = torch.full((5, 3, 2, 4), 0.993, dtype=torch.float16)
        state = {
            "num_frames": 5,
            "input_batch": SimpleNamespace(
                img_batch=_Nested(images),
                find_inputs=[SimpleNamespace(img_ids=torch.tensor([i])) for i in range(5)],
            ),
            "feature_cache": {"text": {"preserved": True}, 0: "zero", 1: "one", 3: "three"},
        }
        return model, state, backbone, projection_calls

    def test_features_match_pinned_preprojection_rounding_positions_and_cache_layout(self):
        model, state, backbone, calls = self._case()
        image = state["input_batch"].img_batch.tensors[2].unsqueeze(0).float()
        # Independent reference to the existing grounding bridge's cache
        # assembly: all visual heads first, BF16 transfer, then projections.
        full = backbone.forward_image(
            image, need_sam3_out=True, need_interactive_out=True, need_propagation_out=True,
        )
        expected = {}
        for key in ("interactive", "sam2_backbone_out"):
            raw = full[key]["backbone_fpn"]
            expected[key] = [
                raw[0].tensors.bfloat16().float() * 1.3,
                raw[1].tensors.bfloat16().float() * 0.7,
                raw[2].tensors.bfloat16(),
            ]
            self.assertFalse(torch.equal(
                expected[key][0], (raw[0].tensors * 1.3).bfloat16().float()
            ))
        with mock.patch.object(subject, "_unsupported_reason", return_value=None):
            receipt = subject.prepare_tracker_frame_features(model, state, 2, False)

        cached_image, cached = state["feature_cache"][2]
        self.assertEqual(cached_image.data_ptr(), state["input_batch"].img_batch.tensors[2].data_ptr())
        self.assertEqual(cached_image.dtype, torch.float16)
        self.assertEqual(backbone.calls[-1][0].dtype, torch.float32)
        self.assertEqual(backbone.calls[-1][1], {
            "need_sam3_out": False, "need_interactive_out": True, "need_propagation_out": True,
        })
        self.assertEqual(calls, [
            ("interactive", 0, torch.bfloat16), ("interactive", 1, torch.bfloat16),
            ("propagation", 0, torch.bfloat16), ("propagation", 1, torch.bfloat16),
        ])
        for key in expected:
            self.assertIs(cached[key]["vision_features"], cached[key]["backbone_fpn"][-1].tensors)
            self.assertIsNone(cached[key]["vision_mask"])
            for index in range(3):
                actual = cached[key]["backbone_fpn"][index]
                self.assertIsNone(actual.mask)
                self.assertTrue(torch.equal(actual.tensors, expected[key][index]))
                position = cached[key]["vision_pos_enc"][index]
                self.assertEqual(position.dtype, torch.float32)
                self.assertTrue(torch.equal(position, full[key]["vision_pos_enc"][index]))
        self.assertNotIn(1, state["feature_cache"])
        self.assertEqual(state["feature_cache"][0], "zero")
        self.assertEqual(state["feature_cache"][3], "three")
        self.assertEqual(state["feature_cache"]["text"], {"preserved": True})
        self.assertEqual(receipt["policy"], subject.TRACKER_FEATURE_POLICY)
        self.assertEqual(state[subject.TRACKER_FEATURE_AUDIT_KEY]["feature_only_preparations"], 1)
        model._prepare_backbone_feats.assert_not_called()

    def test_backward_cache_pruning_and_success_counters_follow_original_rule(self):
        model, state, _backbone, _calls = self._case()
        with mock.patch.object(subject, "_unsupported_reason", return_value=None):
            subject.prepare_tracker_frame_features(model, state, 2, True)
            self.assertNotIn(3, state["feature_cache"])
            self.assertIn(1, state["feature_cache"])
            subject.prepare_tracker_frame_features(model, state, 1, True)
        self.assertNotIn(2, state["feature_cache"])
        self.assertEqual(state[subject.TRACKER_FEATURE_AUDIT_KEY]["feature_only_preparations"], 2)

    def test_fp32_projection_weights_preserve_the_inherited_autocast_boundary(self):
        model, state, backbone, _calls = self._case()
        decoders = {}
        for key, attribute in (
            ("interactive", "interactive_sam_mask_decoder"),
            ("sam2_backbone_out", "sam_mask_decoder"),
        ):
            decoder = SimpleNamespace(
                conv_s0=torch.nn.Conv2d(1, 2, 1, bias=False),
                conv_s1=torch.nn.Conv2d(1, 2, 1, bias=False),
            )
            with torch.no_grad():
                decoder.conv_s0.weight.fill_(1.031)
                decoder.conv_s1.weight.fill_(0.711)
            setattr(model.tracker, attribute, decoder)
            decoders[key] = decoder
        expected = {}
        # H100 stores FP32 weights but the predictor keeps BF16 autocast active.
        # A CPU autocast context verifies that the adapter inherits that exact
        # operation boundary instead of disabling or changing it.
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            full = backbone.forward_image(state["input_batch"].img_batch.tensors[2:3].float())
            for key, decoder in decoders.items():
                features = [item.tensors.bfloat16() for item in full[key]["backbone_fpn"]]
                expected[key] = [decoder.conv_s0(features[0]), decoder.conv_s1(features[1]), features[2]]
            with mock.patch.object(subject, "_unsupported_reason", return_value=None):
                subject.prepare_tracker_frame_features(model, state, 2, False)
        cached = state["feature_cache"][2][1]
        for key, decoder in decoders.items():
            self.assertEqual(decoder.conv_s0.weight.dtype, torch.float32)
            for level, expected_tensor in enumerate(expected[key]):
                actual = cached[key]["backbone_fpn"][level].tensors
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(actual, expected_tensor))

    def test_unsupported_custom_and_distributed_models_use_only_original_bridge(self):
        original = mock.Mock()
        custom = SimpleNamespace(_prepare_backbone_feats=original)
        state = {}
        receipt = subject.prepare_tracker_frame_features(custom, state, 3, True)
        original.assert_called_once_with(state, 3, reverse=True)
        self.assertEqual(receipt, {"policy": "original_feature_bridge", "fallback_reason": "custom_model"})
        self.assertEqual(state[subject.TRACKER_FEATURE_AUDIT_KEY]["fallback_preparations"], 1)
        self.assertEqual(state[subject.TRACKER_FEATURE_AUDIT_KEY]["fallback_reasons"], {"custom_model": 1})

        def typed(name, module, **attributes):
            value = type(name, (), {"__module__": module})()
            value.__dict__.update(attributes)
            return value
        backbone = typed("SAM3VLBackboneTri", "sam3.model.vl_combiner")
        detector = typed(
            "Sam3MultiplexDetector", "sam3.model.sam3_multiplex_detector",
            backbone=backbone, world_size=2, rank=0,
        )
        model = typed(
            "Sam3MultiplexTrackingWithInteractivity", "sam3.model.sam3_multiplex_tracking",
            detector=detector, world_size=2, rank=0, _prepare_backbone_feats=original,
        )
        self.assertEqual(subject._unsupported_reason(model), "distributed_world")

    def test_supported_path_errors_never_fall_back_or_publish_partial_features(self):
        for failure in ("frame_identity", "projection", "pyramid"):
            with self.subTest(failure=failure):
                model, state, backbone, _calls = self._case()
                if failure == "frame_identity":
                    state["input_batch"].find_inputs[2].img_ids = torch.tensor([1])
                elif failure == "projection":
                    model.tracker.sam_mask_decoder.conv_s0 = mock.Mock(side_effect=RuntimeError("projection failed"))
                else:
                    original_forward = backbone.forward_image
                    def malformed(*args, **kwargs):
                        result = original_forward(*args, **kwargs)
                        result["interactive"]["backbone_fpn"].pop()
                        return result
                    backbone.forward_image = malformed
                with mock.patch.object(subject, "_unsupported_reason", return_value=None):
                    with self.assertRaises(RuntimeError):
                        subject.prepare_tracker_frame_features(model, state, 2, False)
                self.assertNotIn(2, state["feature_cache"])
                self.assertEqual(state["feature_cache"][1], "one")
                self.assertNotIn(subject.TRACKER_FEATURE_AUDIT_KEY, state)
                model._prepare_backbone_feats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
