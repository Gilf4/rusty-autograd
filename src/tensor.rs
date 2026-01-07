use crate::ops::Operation;
use crate::ops::{AddOp, MatMulOp, MeanOp, PowOp, ReLUOp, SubOp, TanhOp};
use ndarray::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

pub type TensorRef = Rc<RefCell<Tensor>>;

pub struct Tensor {
    pub data: Array2<f64>,
    pub grad: Array2<f64>,
    pub op: Option<Box<dyn Operation>>,
    pub parents: Vec<TensorRef>,
}

impl Tensor {
    pub fn new(data: Array2<f64>) -> TensorRef {
        let shape = data.shape();
        let nrows = shape[0];
        let ncols = shape[1];
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: None,
            parents: Vec::new(),
        }))
    }

    pub fn set_init_grad(&mut self) {
        let (nrows, ncols) = self.data.dim();
        self.grad = Array2::ones((nrows, ncols));
    }

    pub fn backward(&self) {
        if let Some(op) = &self.op {
            let grads = op.backward(&self.grad, &self.parents);
            for (i, parent) in self.parents.iter().enumerate() {
                let mut p = parent.borrow_mut();
                p.grad += &grads[i];
            }
            for parent in &self.parents {
                parent.borrow().backward();
            }
        }
    }

    pub fn add(a: &TensorRef, b: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let b_borrow = b.borrow();
        let data = &a_borrow.data + &b_borrow.data;
        let (nrows, ncols) = data.dim();
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(AddOp)),
            parents: vec![Rc::clone(a), Rc::clone(b)],
        }))
    }

    pub fn sub(a: &TensorRef, b: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let b_borrow = b.borrow();
        let data = &a_borrow.data - &b_borrow.data;
        let (nrows, ncols) = data.dim();
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(SubOp)),
            parents: vec![Rc::clone(a), Rc::clone(b)],
        }))
    }

    pub fn mat_mul(a: &TensorRef, b: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let b_borrow = b.borrow();
        let data = a_borrow.data.dot(&b_borrow.data);
        let (nrows, ncols) = data.dim();
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(MatMulOp)),
            parents: vec![Rc::clone(a), Rc::clone(b)],
        }))
    }

    pub fn relu(a: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let data = a_borrow.data.mapv(|val| if val > 0.0 { val } else { 0.0 });
        let (nrows, ncols) = data.dim();
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(ReLUOp)),
            parents: vec![Rc::clone(a)],
        }))
    }

    pub fn mean(a: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let val = a_borrow.data.mean().unwrap();
        let data = Array2::from_elem((1, 1), val);
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((1, 1)),
            op: Some(Box::new(MeanOp)),
            parents: vec![Rc::clone(a)],
        }))
    }

    pub fn square(a: &TensorRef) -> TensorRef {
        let two = Tensor::new(Array2::from_elem(a.borrow().data.dim(), 2.0));
        Tensor::pow(a, &two)
    }

    pub fn pow(a: &TensorRef, b: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let b_borrow = b.borrow();
        let exp = b_borrow.data[[0, 0]];
        let data = a_borrow.data.mapv(|val| val.powf(exp));
        let (nrows, ncols) = data.dim();
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(PowOp { exp })),
            parents: vec![Rc::clone(a)],
        }))
    }

    pub fn tanh(a: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let data = a_borrow.data.mapv(|v| v.tanh());
        let (nrows, ncols) = data.dim();

        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(TanhOp)),
            parents: vec![Rc::clone(a)],
        }))
    }
}
