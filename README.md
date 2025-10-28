# Exercise set 1

*All assignments need to be implemented within the function skeletons found in `submission.py`
and you need to hand in this file in the form `submission_<STUDENTID>.py` at the link provided
for this exercise sheet via e-mail.*

## Exercise 1.1

1. Create a tensor that holds $x= 2.0$ with `requires_grad=True`.  
2. Compute
    $$f(x) = 3x^3 + 2x^2 - x + 1$$
    using PyTorch operations.  
3. Use automatic differentiation to compute the derivative $\frac{\partial f}{\partial x}$ at $x = 2.0$.  
4. Return 
    - the value $f(x)$, and  
    - the derivative $\frac{\partial f}{\partial x}\big|_{x=2.0}$

    both as **0-dimensional tensors** (i.e., scalars).

## Exercise 2.2

1. Create two tensors for  

    $$\mathbf{A} = \left(\begin{matrix}
    1 & 2 \\
    3 & 4
    \end{matrix}\right),\ x = (1,1)^\top$$

    and implement

    $$f(\mathbf{x}) = \| \mathbf{A}\mathbf{x} \|^2$$

2. Compute $f(\mathbf{x})$ using PyTorch operations.  
3. Use automatic differentiation to compute the derivative $\frac{\partial f}{\partial \mathbf{x}}$ at the $\mathbf{x}$ given above.  

4. Return
    - the value $f(\mathbf{x})$, and  
    - the derivative $\frac{\partial f}{\partial \mathbf{x}}\big|_{\mathbf{x}=(1,1)^\top}$

    as PyTorch tensors.

## Exercise 2.3

### Exercise 3 — Custom Autograd Function

Implement a custom autograd function `Cube` that computes $f(x) = x^3$ and defines its own backward pass.

1. Create a subclass of `torch.autograd.Function` named `Cube` (template is provided). 
2. In the `forward` method, compute $ x^3$ and save the input tensor for use in automatic differentiation using `ctx.save_for_backward(...)`, i.e., the forward pass looks like 

    ```python
    def forward(ctx, input):
        # YOUR CODE GOES HERE
    ```

3. In the `backward` method, return the gradient of the output with respect to the input,  
   i.e. $\frac{df}{dx} = 3x^2$, multiplied by the incoming `grad_output` (i.e., application of the chain rule). In particular, in your backward pass, i.e., 

    ```python  
    @staticmethod
    def backward(ctx, grad_output):
        # YOUR CODE GOES HERE
    ```

    you can access the saved input via `ctx.saved_tensors`.

The test will call `y = Cube.apply(x)` to compute $z = f(x)$, then `z = y**2)` to square the result and finally call `z.backward()` to compute the gradient of `z` with respect to `x`.
