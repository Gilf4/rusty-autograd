use crate::tensor::{Tensor, TensorRef};
use ndarray::Array2;
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
        let weight = Tensor::new(Array2::ones((in_features, out_features)));
        let bias_tensor = if bias {
            Some(Tensor::new(Array2::ones((1, out_features))))
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
