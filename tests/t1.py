from otter.test_files import test_case
import torch

OK_FORMAT = False

name = "Exercise 2.1"
points = 2

@test_case(points=2)
def test_1(env):
    f, q1_grad = env['assignment_ex1']()
    assert isinstance(q1_grad, torch.Tensor), "q1_grad must be a torch.Tensor."
    assert isinstance(f, torch.Tensor), "Function output must be a torch.Tensor."
    assert q1_grad.ndim == 0, "q1_grad must be a 0-d tensor."
    assert f.ndim == 0, "Function output must be a 0-d tensor."
    assert f.requires_grad, "Function output must have requires_grad=True."
    x_ref = torch.tensor(2.0)
    assert torch.allclose(q1_grad, 9 * x_ref**2 + 4 * x_ref - 1, rtol=1e-5, atol=1e-7), \
        "gradient is wrong."
