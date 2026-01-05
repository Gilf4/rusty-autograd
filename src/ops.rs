use crate::tensor::{Operation, TensorRef};
use ndarray::prelude::*;

#[derive(Clone)]
pub struct AddOp;

impl Operation for AddOp {
    fn backward(&self, grad: &Array2<f64>, _inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        vec![grad.clone(), grad.clone()]
    }
}

#[derive(Clone)]
pub struct SubOp;

impl Operation for SubOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        if inputs.len() != 2 {
            panic!("SubOp expects exactly 2 inputs");
        }
        vec![grad.clone(), grad.mapv(|x| -x)]
    }
}

#[derive(Clone)]
pub struct MatMulOp;

impl Operation for MatMulOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        let a = &inputs[0].borrow().data;
        let b = &inputs[1].borrow().data;
        let lhs = grad.dot(&b.t());
        let rhs = a.t().dot(grad);
        vec![lhs, rhs]
    }
}

#[derive(Clone)]
pub struct ReLUOp;

impl Operation for ReLUOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        let input = &inputs[0].borrow().data;
        let mask = input.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 });
        vec![grad * &mask]
    }
}

#[derive(Clone)]
pub struct MeanOp;

impl Operation for MeanOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        let input = &inputs[0].borrow().data;
        let num_elements = (input.len()) as f64;
        vec![grad.mapv(|x| x / num_elements)]
    }
}

#[derive(Clone)]
pub struct PowOp {
    pub exp: f64,
}

impl Operation for PowOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        let input = &inputs[0].borrow().data;
        let deriv = input.mapv(|val| self.exp * val.powf(self.exp - 1.0));
        vec![grad * &deriv]
    }
}
