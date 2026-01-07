use crate::tensor::{Tensor, TensorRef};
use ndarray::Array2;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::rc::Rc;

pub trait Module {
    fn forward(&self, input: &TensorRef) -> TensorRef;
    fn parameters(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct Linear {
    weight: TensorRef,
    bias: Option<TensorRef>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let std = (2.0 / (in_features + out_features) as f64).sqrt();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, std).unwrap();

        let weight =
            Array2::from_shape_fn((in_features, out_features), |_| normal.sample(&mut rng));
        let weight = Tensor::new(weight);

        let bias_tensor = if bias {
            Some(Tensor::new(Array2::zeros((1, out_features))))
        } else {
            None
        };

        Linear {
            weight,
            bias: bias_tensor,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &TensorRef) -> TensorRef {
        let mut output = Tensor::mat_mul(input, &self.weight);
        if let Some(b) = &self.bias {
            output = Tensor::add(&output, b);
        }
        output
    }

    fn parameters(&self) -> Vec<TensorRef> {
        let mut params = vec![Rc::clone(&self.weight)];
        if let Some(b) = &self.bias {
            params.push(Rc::clone(b));
        }
        params
    }
}

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &TensorRef) -> TensorRef {
        Tensor::relu(input)
    }
}

pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &TensorRef) -> TensorRef {
        Tensor::tanh(input)
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &TensorRef) -> TensorRef {
        let mut current = Rc::clone(input);
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    fn parameters(&self) -> Vec<TensorRef> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
