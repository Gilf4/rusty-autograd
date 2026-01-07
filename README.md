# ru_torch

A lightweight deep learning library in Rust inspired by PyTorch. Built on top of ndarray for numerical computations.

## Features

- **Tensors**: Automatic differentiation with backward mode
- **Neural Networks**: Linear layers, ReLU, Tanh activations, and Sequential containers
- **Optimizers**: SGD optimizer for parameter updates
- **Data Loading**: Dataset and DataLoader utilities for batch processing
- **CSV Support**: Data loading from CSV files
- **ndarray Foundation**: Uses ndarray as the core dependency for matrix operations

## Quick Start

```rust
use ru_torch::{
    data::{DataLoader, TensorDataset},
    nn::{Linear, Module, Sequential, Tanh},
    optim::{Optimizer, SGD},
    tensor::Tensor,
};

// Create a simple neural network
let model = Sequential::new(vec![
    Box::new(Linear::new(2, 4, true)),
    Box::new(Tanh),
    Box::new(Linear::new(4, 1, true)),
]);

// Initialize optimizer
let mut optimizer = SGD::new(&model, 0.05);

// Training loop
for epoch in 0..1000 {
    for (x, target) in dataloader.iter() {
        let output = model.forward(&x);
        let loss = Tensor::mean(&Tensor::square(&Tensor::sub(&output, &target)));
        
        loss.borrow_mut().set_init_grad();
        loss.borrow().backward();
        
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

## Core Components

### Tensors
- Automatic differentiation support
- Basic operations: add, subtract, matrix multiplication
- Activation functions: ReLU, Tanh
- Loss functions: mean squared error

### Neural Network Layers
- `Linear`: Fully connected layers with optional bias
- `ReLU`: Rectified Linear Unit activation
- `Tanh`: Hyperbolic tangent activation
- `Sequential`: Container for stacking layers

### Data Processing
- `TensorDataset`: Simple dataset for tensor data
- `DataLoader`: Batch processing with shuffling support
- CSV loading utilities for quick data import

## License

This project is licensed under the MIT License.
