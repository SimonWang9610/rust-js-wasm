use ndarray::Array2;
use rand::prelude::*;
use std::f64::consts::E;

pub fn one_hot(labels: Array2<f64>, cols: usize) -> Array2<f64> {
    let rows = labels.shape()[0];
    let mut data = vec![];
    for item in labels.into_iter() {
        let mut row = Vec::with_capacity(cols);
        for index in 0..cols {
            if index as f64 == *item {
                row.push(1.);
            } else {
                row.push(0.);
            }
        }
        data.extend_from_slice(&row);
    }
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

pub fn permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut rand::thread_rng());
    perm
}

pub fn compute_loss(output: &Array2<f64>, labels: &Array2<f64>) -> f64 {
    // output [sample, 10]
    // labels [sample, 10]
    let average = -1. / labels.shape()[0] as f64;
    output
        .into_iter()
        .zip(labels.iter())
        .fold(0., |acc, (o, l)| acc + l * o.log(E))
        * average
}
