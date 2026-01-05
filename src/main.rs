use ndarray::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

trait Operation {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>>;
}

#[derive(Clone)]
struct AddOp;

impl Operation for AddOp {
    fn backward(&self, grad: &Array2<f64>, inputs: &[TensorRef]) -> Vec<Array2<f64>> {
        inputs.iter().map(|_| grad.clone()).collect()
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
}

type TensorRef = Rc<RefCell<Tensor>>;

struct Tensor {
    data: Array2<f64>,
    grad: Array2<f64>,
    op: Option<Box<dyn Operation>>,
    parents: Vec<TensorRef>,
}

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

trait Module {
    fn forward(&self, input: &TensorRef) -> TensorRef;

    // Опционально: метод для сбора параметров (для optimizer в будущем)
    fn parameters(&self) -> Vec<TensorRef> {
        vec![]
    }
}

// Аналог nn.Linear
struct Linear {
    weight: TensorRef,
    bias: Option<TensorRef>,
}

impl Linear {
    fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Инициализация весов (для простоты — единицы, в реальности — random)
        let weight_data = Array2::ones((in_features, out_features));
        let weight = Tensor::new(weight_data);

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

struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &TensorRef) -> TensorRef {
        Tensor::relu(input)
    }
}

impl ReLU {
    fn new() -> Self {
        ReLU
    }
}

struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    fn new(layers: Vec<Box<dyn Module>>) -> Self {
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
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

fn main() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(3, 4, true)),
    ]);

    let x = Tensor::new(Array2::from_shape_fn((1, 2), |(_, j)| (j + 1) as f64));

    let output = model.forward(&x);

    output.borrow_mut().set_init_grad();
    output.borrow().backward();

    println!("Output: {:?}", output.borrow().data);

    for param in model.parameters() {
        println!("Param grad: {:?}", param.borrow().grad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_add_forward() {
        let a = Tensor::new(array![[1.0, 2.0]]);
        let b = Tensor::new(array![[3.0, 4.0]]);
        let c = Tensor::add(&a, &b);
        assert_eq!(c.borrow().data, array![[4.0, 6.0]]);
    }

    #[test]
    fn test_add_backward() {
        let a = Tensor::new(array![[1.0, 2.0]]);
        let b = Tensor::new(array![[3.0, 4.0]]);
        let c = Tensor::add(&a, &b);
        c.borrow_mut().set_init_grad();
        c.borrow().backward();
        assert_eq!(a.borrow().grad, array![[1.0, 1.0]]);
        assert_eq!(b.borrow().grad, array![[1.0, 1.0]]);
    }

    #[test]
    fn test_matmul_forward() {
        let a = Tensor::new(array![[1.0, 2.0, 3.0]]);
        let b = Tensor::new(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let c = Tensor::mat_mul(&a, &b);
        assert_eq!(c.borrow().data, array![[22.0, 28.0]]);
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::new(array![[1.0, 2.0, 3.0]]);
        let b = Tensor::new(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let c = Tensor::mat_mul(&a, &b);
        c.borrow_mut().set_init_grad();
        c.borrow().backward();
        assert_eq!(a.borrow().grad, array![[3.0, 7.0, 11.0]]); // grad * b^T
        assert_eq!(b.borrow().grad, array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]); // a^T * grad
    }

    #[test]
    fn test_relu_forward() {
        let a = Tensor::new(array![[-1.0, 0.0, 2.0]]);
        let b = Tensor::relu(&a);
        assert_eq!(b.borrow().data, array![[0.0, 0.0, 2.0]]);
    }

    #[test]
    fn test_relu_backward() {
        let a = Tensor::new(array![[-1.0, 0.0, 2.0]]);
        let b = Tensor::relu(&a);
        b.borrow_mut().set_init_grad();
        b.borrow().backward();
        assert_eq!(a.borrow().grad, array![[0.0, 0.0, 1.0]]); // grad * mask
    }

    #[test]
    fn test_full_forward_backward() {
        let x = Tensor::new(array![[1.0, 2.0, 3.0]]);
        let w = Tensor::new(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let b = Tensor::new(array![[1.0, 2.0]]);
        let xw = Tensor::mat_mul(&x, &w);
        let y = Tensor::add(&xw, &b);
        let z = Tensor::relu(&y);
        z.borrow_mut().set_init_grad();
        z.borrow().backward();

        assert_eq!(z.borrow().data, array![[23.0, 30.0]]);

        assert_eq!(x.borrow().grad, array![[3.0, 7.0, 11.0]]);
        assert_eq!(w.borrow().grad, array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
        assert_eq!(b.borrow().grad, array![[1.0, 1.0]]);
    }

    #[test]
    #[should_panic(expected = "expects exactly 2 inputs")]
    fn test_matmul_panic_wrong_inputs() {
        let a = Tensor::new(array![[1.0]]);
        let op = MatMulOp;
        op.backward(&array![[1.0]], &vec![a]);
    }
}
