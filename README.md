#  Journey

---

##  Phase 1: Micrograd — Scalar Autograd Engine

###  Goal
Understand the *core principles* of neural networks and backpropagation using scalar operations.

###  Key Concepts
- Reverse-mode autodiff (backpropagation)
- Computational graph construction
- Gradients via chain rule
- Basic MLP training loop

###  What to Build
- `Value` struct: `.data`, `.grad`, `.backward()`
- Scalar operations: `+`, `-`, `*`, `tanh`, `pow`, etc.
- Graph tracking + topological sort for backward pass
- A tiny MLP (e.g. `2 → 4 → 1`) on toy data (like XOR)

###  What You’ll Have
- A working scalar autograd engine
- A mental model of how frameworks like PyTorch work under the hood

---

## Phase 2: Tinygrad — Tensor-Based Engine

### Goal  
Scale your micrograd knowledge into a full tensor-based deep learning engine capable of training real models (e.g. transformers, CNNs), on CPU or GPU.

---

### Core Concepts to Implement

- Tensors: Multi-dimensional data (vs. scalar `Value` in micrograd)
- Autograd over Tensors: Graph-based reverse-mode differentiation
- Broadcasting & Shape Tracking: NumPy-style broadcasting
- Custom GPU Execution: Generate and run GPU kernels (optional)
- JIT Kernel Generation: Runtime codegen for fused ops (optional but powerful)

---

### What to Build 

#### Tensor Engine
- `Tensor<T>` struct:
  - `.data: Vec<T>` (or GPU buffer)
  - `.grad: Option<Vec<T>>`
  - `.shape`, `.strides`, etc.
  - `.backward: Option<Fn()>`
- Graph tracking using `Rc<RefCell<>>` or similar

#### Core Tensor Ops
- `add`, `mul`, `matmul`
- `reshape`, `transpose`
- `sum`, `mean`, `max`
- `relu`, `softmax`, `gelu`

#### Autograd Engine
- Backpropagation across tensor ops
- Backward functions stored in graph
- Gradient accumulation with correct broadcasting

#### Layers & Models
- `Linear`
- `ReLU`, `GELU`
- `LayerNorm`, `Dropout` (optional)
- Stackable `Module` or `Model` abstraction

#### Optimizers
- `SGD` (momentum optional)
- `Adam`

#### GPU Acceleration
- Integrate with:
  - [`ocl`](https://crates.io/crates/ocl) (OpenCL)
  - [`cust`](https://crates.io/crates/cust) (CUDA)
  - `wgpu` or `vulkan` (if adventurous)
- GPU tensor backend (`TensorGPU`)
- Kernel JIT generator using strings or AST-to-source

### What You’ll Have at the End

- A minimal deep learning framework you fully understand
- A CPU tensor engine with autograd
- A GPU backend with custom kernel execution
- Capability to train real models (MLPs, CNNs, Transformers) on toy datasets
- Strong foundation to build your own PyTorch/tinygrad-style stack

---

##  Phase 3: Transformer — Real Model on Your Engine

###  Goal
Build a basic Transformer architecture and train it using your own deep learning engine.

###  Key Concepts
- Attention mechanism: Q/K/V, dot product, masking
- Transformer block: Attention + MLP + LayerNorm + residual
- Tokenization and sequence modeling
- Training vs inference (e.g., KV caching)

###  What to Build
- Tokenizer: char-level or byte-level
- TransformerBlock: Self-Attention, MLP, LayerNorm
- Stack of transformer blocks
- Loss: Cross-entropy
- Training loop: forward → loss → backward → update
- Inference loop: token-by-token generation

###  What You’ll Have
- A basic language model (nanoGPT-style)
- Complete training and inference capability
- A deep understanding of how transformers work — top to bottom

---

##  Recap: Stack Progression

| Phase           | Focus                   | What You Build                          |
| --------------- | ----------------------- | --------------------------------------- |
| **Micrograd**   | Math & Backprop         | Scalar engine + tiny MLP                |
| **Tinygrad**    | Computation & Gradients | Tensor engine + autograd + layers + GPU |
| **Transformer** | Architecture & Training | Real model + training loop + inference  |

