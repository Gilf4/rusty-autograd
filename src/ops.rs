use crate::tensor::TensorRef;
use ndarray::prelude::*;

pub trait Operation {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>>;
}

#[derive(Clone)]
pub struct AddOp;

impl Operation for AddOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        if inputs.len() != 2 {
            panic!("AddOp expects exactly 2 inputs");
        }

        let a_ref = inputs[0].borrow();
        let b_ref = inputs[1].borrow();

        let a_shape = a_ref.data.shape();
        let b_shape = b_ref.data.shape();
        let out_shape = grad.shape();

        let mut grad_a = grad.clone();
        if a_shape[0] == 1 && out_shape[0] > 1 {
            grad_a = grad.sum_axis(Axis(0)).insert_axis(Axis(0));
        }

        let mut grad_b = grad.clone();
        if b_shape[0] == 1 && out_shape[0] > 1 {
            grad_b = grad.sum_axis(Axis(0)).insert_axis(Axis(0));
        }

        vec![grad_a, grad_b]
    }
}

#[derive(Clone)]
pub struct SubOp;

impl Operation for SubOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        if inputs.len() != 2 {
            panic!("SubOp expects exactly 2 inputs");
        }

        let a_ref = inputs[0].borrow();
        let b_ref = inputs[1].borrow();

        let a_shape = a_ref.data.shape();
        let b_shape = b_ref.data.shape();
        let out_shape = grad.shape();

        let mut grad_a = grad.clone();
        if a_shape[0] == 1 && out_shape[0] > 1 {
            let summed = grad_a.sum_axis(Axis(0));
            grad_a = summed.insert_axis(Axis(0));
        }

        let mut grad_b = grad.mapv(|x| -x);
        if b_shape[0] == 1 && out_shape[0] > 1 {
            let summed = grad_b.sum_axis(Axis(0));
            grad_b = summed.insert_axis(Axis(0));
        }

        vec![grad_a, grad_b]
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
        let num_elements = input.len() as f64;

        let grad_scalar = if grad.len() == 1 {
            grad[[0, 0]]
        } else {
            grad.mean().unwrap_or(0.0)
        };

        let out = Array2::from_elem(input.dim(), grad_scalar / num_elements);

        vec![out]
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

#[derive(Clone)]
pub struct TanhOp;

impl Operation for TanhOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        let input = &inputs[0].borrow().data;

        let tanh_x = input.mapv(|x| x.tanh());

        // d/dx tanh(x) = 1 - tanh(x)^2
        let deriv = tanh_x.mapv(|y| 1.0 - y * y);

        vec![grad * &deriv]
    }
}
