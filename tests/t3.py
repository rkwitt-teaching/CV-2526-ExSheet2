from otter.test_files import test_case
import torch

OK_FORMAT = False

name = "Exercise 2.3"
points = 3


@test_case(points=3)
def test_3(env):
    Cube = env['assignment_ex3']()
    x = torch.tensor(2.0, requires_grad=True)
    y = Cube.apply(x)
    z = y**2
    z.backward()
    assert x.grad.ndim == 0, "Gradient should be a scalar"
    assert torch.isclose(x.grad, torch.tensor(192.0)), "Gradient incorrect"
