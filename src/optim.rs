use crate::nn::Module;
use crate::tensor::TensorRef;
use ndarray::Array2;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD {
    params: Vec<TensorRef>,
    lr: f64,
}

impl SGD {
    pub fn new(model: &impl Module, lr: f64) -> Self {
        SGD {
            params: model.parameters(),
            lr,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for param in &self.params {
            let mut p = param.borrow_mut();
            p.data = &p.data - &(self.lr * &p.grad);
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            let mut p = param.borrow_mut();
            p.grad = Array2::zeros(p.data.dim());
        }
    }
}
