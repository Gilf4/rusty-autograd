use ndarray::prelude::*;
use ru_torch::nn::{Linear, Module, ReLU, Sequential};
use ru_torch::optim::{Optimizer, SGD};
use ru_torch::tensor::Tensor;

fn main() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU),
        Box::new(Linear::new(3, 4, true)),
    ]);

    let mut optimizer = SGD::new(&model, 0.01);

    let inputs = vec![
        array![[0.0, 0.0]],
        array![[0.0, 1.0]],
        array![[1.0, 0.0]],
        array![[1.0, 1.0]],
    ];

    let targets = vec![
        array![[0.0, 0.0, 0.0, 0.0]],
        array![[1.0, 1.0, 1.0, 1.0]],
        array![[1.0, 1.0, 1.0, 1.0]],
        array![[2.0, 2.0, 2.0, 2.0]],
    ];

    let epochs = 500;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x_data, y_data) in inputs.iter().zip(&targets) {
            let x = Tensor::new(x_data.clone());
            let target = Tensor::new(y_data.clone());

            let output = model.forward(&x);

            let diff = Tensor::sub(&output, &target);
            let squared = Tensor::square(&diff);
            let loss = Tensor::mean(&squared);

            loss.borrow_mut().set_init_grad();
            loss.borrow().backward();

            optimizer.step();
            optimizer.zero_grad();

            total_loss += loss.borrow().data[[0, 0]];
        }

        if epoch % 100 == 0 {
            println!(
                "Epoch {} | Loss: {:.6}",
                epoch,
                total_loss / inputs.len() as f64
            );
        }
    }
}
