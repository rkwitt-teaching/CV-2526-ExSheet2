from otter.test_files import test_case
import torch

OK_FORMAT = False

name = "Exercise 2.1"
points = 2

@test_case(points=2)
def test_2(env):
    A = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    x = torch.tensor([1.0, 1.0], requires_grad=True)

    f, q2_grad = env['assignment_ex2']()
    assert isinstance(q2_grad, torch.Tensor), "q1_grad must be a torch.Tensor."
    assert isinstance(f, torch.Tensor), "Function output must be a torch.Tensor."
    assert f.ndim == 0, "Function output must be a 0-d tensor."
    assert q2_grad.ndim == 1, "q1_grad must be 1-d tensor."
    assert f.requires_grad, "Function output must have requires_grad=True."
    assert torch.norm(q2_grad - 2*torch.matmul(A.T, A)@x).item() < 1e-6, "grad is wrong."