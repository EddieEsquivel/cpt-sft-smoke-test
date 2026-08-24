import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class _Weights:
    def __init__(self, values):
        self.data = values


class _Datum:
    def __init__(self, loss_tokens):
        self.loss_fn_inputs = {"weights": _Weights([1.0] * loss_tokens)}


class _FakeTensor:
    def __init__(self, value):
        self.value = list(value) if isinstance(value, (list, tuple)) else float(value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        return _FakeTensor(self.value[key])

    def float(self):
        return self

    def ne(self, other):
        return _FakeTensor([float(value != other) for value in self.value])

    def sum(self):
        return _FakeTensor(sum(self.value))

    def item(self):
        return float(self.value)

    def __sub__(self, other):
        return _FakeTensor(self.item() - other.item())

    def __truediv__(self, other):
        return _FakeTensor(self.item() / other)


def _load_cpt_module():
    dotenv_stub = types.SimpleNamespace(load_dotenv=lambda: None)
    with mock.patch.dict(sys.modules, {"dotenv": dotenv_stub}):
        script = Path(__file__).parents[1] / "scripts" / "01_cpt.py"
        spec = importlib.util.spec_from_file_location("cpt_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class CptHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpt = _load_cpt_module()

    def test_optimizer_steps_follow_token_accumulation_boundaries(self):
        datums = [_Datum(3), _Datum(5), _Datum(7), _Datum(11)]

        self.assertEqual(self.cpt.datum_loss_token_count(datums[0]), 3)
        self.assertEqual(
            self.cpt.optimizer_steps_per_epoch(
                datums,
                batch_size=2,
                target_tokens_per_step=0,
            ),
            2,
        )
        self.assertEqual(
            self.cpt.optimizer_steps_per_epoch(
                datums,
                batch_size=1,
                target_tokens_per_step=10,
            ),
            2,
        )

    def test_real_cpt_flags_map_to_expected_values(self):
        argv = [
            "01_cpt.py",
            "--lr",
            "2e-6",
            "--lr-schedule",
            "cosine",
            "--warmup-steps",
            "250",
            "--target-tokens-per-step",
            "4000000",
            "--adam-beta2",
            "0.95",
            "--weight-decay",
            "0.1",
            "--grad-clip-norm",
            "1.0",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = self.cpt.parse_args()

        self.assertEqual(args.lr, 2e-6)
        self.assertEqual(args.lr_schedule, "cosine")
        self.assertEqual(args.warmup_steps, 250)
        self.assertEqual(args.target_tokens_per_step, 4_000_000)
        self.assertEqual(args.adam_beta2, 0.95)
        self.assertEqual(args.weight_decay, 0.1)
        self.assertEqual(args.grad_clip_norm, 1.0)

    def test_cpt_loss_returns_raw_sum_and_mean_metric(self):
        fake_torch = types.SimpleNamespace(
            float32="float32",
            tensor=lambda value, **_: _FakeTensor(value),
            dot=lambda left, right: _FakeTensor(
                sum(a * b for a, b in zip(left.value, right.value))
            ),
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            loss_fn = self.cpt.make_cpt_loss()
            raw_loss, metrics = loss_fn(
                [_Datum(3)],
                [_FakeTensor([-1.0, -2.0, -3.0])],
            )

        self.assertEqual(raw_loss.item(), 6.0)
        self.assertEqual(metrics, {"cpt_loss": 2.0, "n_tokens": 3})


if __name__ == "__main__":
    unittest.main()
