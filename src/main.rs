use ndarray::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

trait Operation {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>>;
    fn clone_box(&self) -> Box<dyn Operation>;
}

impl Clone for Box<dyn Operation> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone)]
struct AddOp;

impl Operation for AddOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        inputs.iter().map(|_| grad.clone()).collect()
    }
    fn clone_box(&self) -> Box<dyn Operation> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct MatMulOp;

impl Operation for MatMulOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        if inputs.len() != 2 {
            panic!("MatMulOp expects exactly 2 inputs");
        }
        let a = &inputs[0].borrow().data;
        let b = &inputs[1].borrow().data;
        let lhs = grad.dot(&b.t());
        let rhs = a.t().dot(grad);
        vec![lhs, rhs]
    }
    fn clone_box(&self) -> Box<dyn Operation> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct ReLUOp;

impl Operation for ReLUOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        if inputs.len() != 1 {
            panic!("ReLUOp expects exactly 1 input");
        }
        let input = &inputs[0].borrow().data;

        let mask: Array2<f64> = input.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 });
        let output_grad = grad * &mask;

        vec![output_grad]
    }

    fn clone_box(&self) -> Box<dyn Operation> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct Tensor {
    data: Array2<f64>,
    grad: Array2<f64>,
    op: Option<Box<dyn Operation>>,
    parents: Vec<Rc<RefCell<Tensor>>>,
}

type TensorRef = Rc<RefCell<Tensor>>;

impl Tensor {
    fn new(data: Array2<f64>) -> TensorRef {
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

    fn add(a: &TensorRef, b: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let b_borrow = b.borrow();
        let data = &a_borrow.data + &b_borrow.data;
        let shape = data.shape();
        let nrows = shape[0];
        let ncols = shape[1];
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(AddOp)),
            parents: vec![Rc::clone(a), Rc::clone(b)],
        }))
    }

    fn mat_mul(a: &TensorRef, b: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let b_borrow = b.borrow();
        let data = a_borrow.data.dot(&b_borrow.data);
        let shape = data.shape();
        let nrows = shape[0];
        let ncols = shape[1];
        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(MatMulOp)),
            parents: vec![Rc::clone(a), Rc::clone(b)],
        }))
    }

    fn relu(a: &TensorRef) -> TensorRef {
        let a_borrow = a.borrow();
        let input_data = &a_borrow.data;

        let shape = input_data.shape();
        let nrows = shape[0];
        let ncols = shape[1];

        let data = input_data.mapv(|val| if val > 0.0 { val } else { 0.0 });

        Rc::new(RefCell::new(Tensor {
            data,
            grad: Array2::zeros((nrows, ncols)),
            op: Some(Box::new(ReLUOp)),
            parents: vec![Rc::clone(a)],
        }))
    }

    fn set_init_grad(&mut self) {
        let (nrows, ncols) = self.data.dim();
        self.grad = Array2::ones((nrows, ncols));
    }

    fn backward(&self) {
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
}

fn main() {
    // Вход: batch_size=1, in_features=3
    let x = Tensor::new(Array2::from_shape_fn((1, 3), |(_, j)| (j + 1) as f64));
    // x = [[1.0, 2.0, 3.0]]

    // Вес: in_features=3, out_features=2
    let w = Tensor::new(Array2::from_shape_fn((3, 2), |(i, j)| {
        (i * 2 + j + 1) as f64
    }));
    // w = [[1.0, 2.0],
    //      [3.0, 4.0],
    //      [5.0, 6.0]]

    // Bias: out_features=2
    let b = Tensor::new(Array2::from_shape_fn((1, 2), |(_, j)| (j + 1) as f64));
    // b = [[1.0, 2.0]]

    // Прямой проход: y = x @ w + b
    let xw = Tensor::mat_mul(&x, &w); // (1,3) @ (3,2) -> (1,2)
    let y = Tensor::add(&xw, &b); // + bias broadcasting (ndarray поддерживает)
    let z = Tensor::relu(&y); // ReLU

    // Задаём градиент на выходе (представим, что loss дал grad=1 для обоих выходов)
    z.borrow_mut().set_init_grad(); // grad = [[1.0, 1.0]]

    // Обратный проход
    z.borrow().backward();

    // Проверяем градиенты
    println!("Input x:\n{:?}", x.borrow().data);
    println!("Weight w:\n{:?}", w.borrow().data);
    println!("Bias b:\n{:?}", b.borrow().data);
    println!("\nForward output z (after ReLU):\n{:?}", z.borrow().data);

    println!("\n=== Градиенты ===");
    println!("grad wrt x:\n{:?}", x.borrow().grad);
    println!("grad wrt w:\n{:?}", w.borrow().grad);
    println!("grad wrt b:\n{:?}", b.borrow().grad);
}
