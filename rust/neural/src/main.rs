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

    let config: Vec<usize> = vec![784, 500, 350, 200, 100, 50, 10];
    let ((x_train, y_train), (x_test, y_test)) = load_mnist(); // [sample, 784] [sample, 10]

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
    


}

// Normalize: x_train = x_train / 255
// Loss function: binary_entropy -> categorical entropy
// one_hot labels
// *** weights initialization
