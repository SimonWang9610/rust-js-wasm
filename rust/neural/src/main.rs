extern crate utils;
use utils::activation::Activation;
use utils::dataset::mnist::load_mnist;
use utils::network::{evaluate, Network};

// use ndarray::Axis;
use std::path::Path;
use std::time::Instant;

macro_rules! timeit {
    ($x:expr) => {{
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}

fn main() {
    let paths: Vec<&Path> = vec![
        "./mnist/train-images.idx3-ubyte".as_ref(),
        "./mnist/train-labels.idx1-ubyte".as_ref(),
        "./mnist/t10k-images.idx3-ubyte".as_ref(),
        "./mnist/t10k-labels.idx1-ubyte".as_ref(),
    ];
    let parameters: Vec<usize> = vec![784, 200, 50, 10];
    let network = Network::new(parameters, Activation::Relu, false);
    for layer in network.layers.borrow().iter() {
        println!("layer: {}", layer);
    }

    let ((x_train, y_train), (x_test, y_test)) = load_mnist(paths); // [sample, 784] [sample, 10]
    let epoches = 50;
    let alpha = 0.5;
    let batch_size = 128;
    let num_batches = (x_train.shape()[0] + batch_size - 1) / batch_size;
    println!(
        "x train {:?} y train {:?}",
        x_train.shape(),
        y_train.shape()
    );

    for epoch in 0..epoches {
        println!("Training on Epoch #{}#", epoch);
        timeit!({
            let (loss, train_acc) =
                network.batch(&x_train, &y_train, alpha, batch_size, num_batches);
            println!(
                "Epoch #{}#: Train-Acc {:.4}  Loss: {:.8}",
                epoch, train_acc, loss
            );
        });
        /* println!(
            "bias {}",
            &network.back().unwrap().bias.borrow().mean().unwrap()
        );
        println!(
            "weights {}",
            &network.back().unwrap().weights.borrow().mean().unwrap()
        ); */
    }

    let predictions = network.predict(x_test.reversed_axes());
    let test_correct = evaluate(&predictions, &y_test);
    println!("Test-Acc {}", test_correct / y_test.shape()[0] as f64);
}

// Normalize: x_train = x_train / 255
// Loss function: binary_entropy -> categorical entropy
// one_hot labels
// *** weights initialization
