use ndarray::{Array2, Axis, s};
use ru_torch::{
    data::{DataLoader, TensorDataset},
    nn::{Linear, Module, Sequential, Tanh},
    optim::{Optimizer, SGD},
    read_csv::csv_to_array2,
    tensor::Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4, true)),
        Box::new(Tanh),
        Box::new(Linear::new(4, 1, true)),
    ]);

    let mut optimizer = SGD::new(&model, 0.05);

    let array: Array2<f64> = csv_to_array2("xor.csv")?;

    let inputs: Array2<f64> = array.slice(s![.., 0..2]).to_owned();

    let targets: Array2<f64> = array.column(2).to_owned().insert_axis(Axis(1));

    let dataset = TensorDataset::new(inputs, targets);
    let mut dataloader = DataLoader::new(dataset).batch_size(4).shuffle(true);

    for epoch in 0..1000 {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (x, target) in dataloader.iter() {
            let output = model.forward(&x);

            let diff = Tensor::sub(&output, &target);
            let squared = Tensor::square(&diff);
            let loss = Tensor::mean(&squared);

            loss.borrow_mut().set_init_grad();
            loss.borrow().backward();

            optimizer.step();
            optimizer.zero_grad();

            total_loss += loss.borrow().data[[0, 0]];
            num_batches += 1;
        }

        if epoch % 100 == 0 {
            println!(
                "Epoch {} | Loss: {:.6}",
                epoch,
                total_loss / num_batches as f64
            );
        }
    }

    Ok(())
}
