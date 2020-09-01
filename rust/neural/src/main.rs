extern crate utils;

use utils::dataset::mnist::load_mnist;
use utils::network::{classification, evaluate, load_model, train_network};
use utils::utils::transform;

use std::path::Path;

fn main() {
    /* let paths: Vec<&Path> = vec![
        "./mnist/train-images.idx3-ubyte".as_ref(),
        "./mnist/train-labels.idx1-ubyte".as_ref(),
        "./mnist/t10k-images.idx3-ubyte".as_ref(),
        "./mnist/t10k-labels.idx1-ubyte".as_ref(),
    ];
    let config: Vec<usize> = vec![784, 200, 50, 10];
    let ((x_train, y_train), (x_test, y_test)) = load_mnist(paths); // [sample, 784] [sample, 10]

    train_network(x_train, y_train, config, 100, 128, 1.); */

    let trained_network = load_model("./parameters.json");

    // let img = transform("./seven.png"); // [1, shape]
    let predict = trained_network.predict_image("./seven.png");
    println!("predict  {}", classification(predict));

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
