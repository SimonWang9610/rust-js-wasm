extern crate network;
extern crate utils;

use network::{classification, evaluate, load_model, train_network};
use utils::dataset::mnist::load_mnist;
use utils::utils::transform;

use std::path::Path;
use std::time::Instant;

macro_rules! timing {
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
    let config: Vec<usize> = vec![784, 500, 350, 200, 100, 50, 10];
    let ((x_train, y_train), (x_test, y_test)) = load_mnist(paths); // [sample, 784] [sample, 10]

    // train_network(x_train, y_train, x_test, y_test, config, 50, 128, 1.25);

    let trained_network = load_model("./parameters-32-q.json", true);
    // let quantized_network = trained_network.save("./parameters-32-q.json", true);

    timing!({
        let test_output = trained_network.predict_array(&x_test);

        let acc = evaluate(&test_output, &y_test) / 10000.;
        println!(" Test-Acc: {:?}", acc);
    });

    timing!({
        let predict = trained_network.predict_image("./seven.png");
        println!("predict {:?}", predict);
    });
    

    /* let predictions = trained_network.predict(x_test.reversed_axes());
    let test_correct = evaluate(&predictions, &y_test);
    println!("Test-Acc {}", test_correct / y_test.shape()[0] as f64); */

    /* println!(
        "bias {}",
        &network.back().unwrap().bias.borrow().mean().unwrap()
    );
    println!(
        "weights {}",
        &network.back().unwrap().weights.borrow().mean().unwrap()
    ); */
}

// Normalize: x_train = x_train / 255
// Loss function: binary_entropy -> categorical entropy
// one_hot labels
// *** weights initialization
