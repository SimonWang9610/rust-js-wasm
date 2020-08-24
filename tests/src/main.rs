use ndarray::{array, Array, Array2, Axis};
use ndarray_rand::rand_distr::{Distribution, Normal};

use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let a = Array::random((2, 5), Normal::new(0., 1.).unwrap());
    println!("{:8.4}", a);

    let b = Normal::new(0., 1.).unwrap();
    let cols = 2;
    let rows = 5;
    let mut data = vec![];
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push(b.sample(&mut rand::thread_rng()));
        }
        data.extend_from_slice(&row);
    }
    let w = Array2::from_shape_vec((rows, cols), data).unwrap();
    println!("{:8.4}", &w * 0.01);
    let product = a.dot(&w);
    println!(" product {:8.4}", product);
    let mut product = product.reversed_axes();
    println!("reversed {:8.4}", product);
    product.swap_axes(0, 1);
    println!("swap {:8.4}", product);
    let relu = relu(&product);
    println!("relu {:8.4}", relu);
    println!("relu * product {:8.4}", relu * product);

    /* let exp = product
        .map_axis(Axis(1), |row| {
            row.fold(0., |acc, &ele: &f64| acc + ele.exp())
        })
        .into_shape((product.shape()[0], 1))
        .unwrap();

    println!(" tanh {:8.4}", exp);

    let soft = product / exp;
    println!("{}", soft); */

    /* let test = RefCell::new(product);
    let temp = test.borrow().clone();

    *test.borrow_mut() = temp + 0.5 * Array::ones((2, 2));
    println!("{:?}", test); */
}

fn relu(input: &Array2<f64>) -> Array2<f64> {
    /* for ele in input.iter_mut() {
        *ele = if *ele >= 0. { 1. } else { 0. };
    }
    input */
    input.mapv(|ele| if ele >= 0. { 1. } else { 0. })
}

fn tanh(input: Array2<f64>) -> Array2<f64> {
    input.mapv(|ele| {
        let exp_pos = ele.exp();
        let exp_neg = (-ele).exp();
        (exp_pos - exp_neg) / (exp_pos + exp_neg)
    })
}
