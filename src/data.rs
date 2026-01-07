use crate::tensor::Tensor;
use ndarray::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

pub type TensorRef = Rc<RefCell<Tensor>>;

pub trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> (TensorRef, TensorRef);
}

pub struct TensorDataset {
    inputs: Array2<f64>,
    targets: Array2<f64>,
}

impl TensorDataset {
    pub fn new(inputs: Array2<f64>, targets: Array2<f64>) -> Self {
        assert_eq!(
            inputs.nrows(),
            targets.nrows(),
            "Inputs and targets must have same number of samples"
        );
        Self { inputs, targets }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.nrows()
    }

    fn get(&self, index: usize) -> (TensorRef, TensorRef) {
        let input_row = self
            .inputs
            .slice(s![index, ..])
            .to_owned()
            .into_shape_with_order((1, self.inputs.ncols()))
            .unwrap();
        let target_row = self
            .targets
            .slice(s![index, ..])
            .to_owned()
            .into_shape_with_order((1, self.targets.ncols()))
            .unwrap();

        let input_tensor = Tensor::new(input_row);
        let target_tensor = Tensor::new(target_row);

        (input_tensor, target_tensor)
    }
}

pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();

        Self {
            dataset,
            batch_size: 1,
            shuffle: false,
            indices,
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    fn reset_indices(&mut self) {
        if self.shuffle {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        } else {
            self.indices = (0..self.dataset.len()).collect();
        }
    }

    pub fn iter(&mut self) -> DataLoaderIterator<'_, D> {
        self.reset_indices();
        DataLoaderIterator {
            dataloader: self,
            current_idx: 0,
        }
    }
}

pub struct DataLoaderIterator<'a, D: Dataset> {
    dataloader: &'a mut DataLoader<D>,
    current_idx: usize,
}

impl<'a, D: Dataset> Iterator for DataLoaderIterator<'a, D> {
    type Item = (TensorRef, TensorRef);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataloader.indices.len() {
            return None;
        }

        let batch_end =
            (self.current_idx + self.dataloader.batch_size).min(self.dataloader.indices.len());
        let batch_indices = &self.dataloader.indices[self.current_idx..batch_end];

        if batch_indices.is_empty() {
            return None;
        }

        let first_idx = batch_indices[0];
        let (first_input, first_target) = self.dataloader.dataset.get(first_idx);

        let first_input_data = first_input.borrow().data.clone();
        let first_target_data = first_target.borrow().data.clone();

        let batch_input_shape = (batch_indices.len(), first_input_data.ncols());
        let batch_target_shape = (batch_indices.len(), first_target_data.ncols());

        let mut batch_input_data = Array2::zeros(batch_input_shape);
        let mut batch_target_data = Array2::zeros(batch_target_shape);

        for (i, &idx) in batch_indices.iter().enumerate() {
            let (input_tensor, target_tensor) = self.dataloader.dataset.get(idx);
            batch_input_data
                .row_mut(i)
                .assign(&input_tensor.borrow().data.row(0));
            batch_target_data
                .row_mut(i)
                .assign(&target_tensor.borrow().data.row(0));
        }

        self.current_idx = batch_end;

        let batch_input = Tensor::new(batch_input_data);
        let batch_target = Tensor::new(batch_target_data);

        Some((batch_input, batch_target))
    }
}
