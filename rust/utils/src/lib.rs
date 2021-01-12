pub mod activation;
pub mod dataset;
pub mod layer;
pub mod propagation;
pub mod trained;
pub mod utils;
pub mod quantization;

use serde_json::Value;
use std::time::Instant;

#[macro_export]
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


pub trait ConvertT {
    fn to_layer(self) -> layer::Layer;
    fn to_json(&self) -> Value;
}
// use ndarray::{s, Array2, Axis};

// use activation::Activation;
// use layer::Layer;
// use propagation::{backward, forward};
// use std::collections::LinkedList;
// use utils::{compute_loss, permutation};

