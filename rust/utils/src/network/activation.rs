use ndarray::{Array2, Axis};

// activation functions
#[derive(Debug, Copy, Clone)]
pub enum Activation {
    Relu,
    Tanh,
}

pub fn relu(input: Array2<f64>) -> Array2<f64> {
    /* for ele in input.iter_mut() {
        *ele = if *ele >= 0. { 1. } else { 0. };
    }
    input */
    input.mapv_into(|ele| if ele >= 0. { ele } else { 0. })
}

pub fn tanh(input: Array2<f64>) -> Array2<f64> {
    input.mapv_into(|ele| {
        let exp_pos = ele.exp();
        let exp_neg = (-ele).exp();
        (exp_pos - exp_neg) / (exp_pos + exp_neg)
    })
}

pub fn softmax(input: Array2<f64>) -> Array2<f64> {
    let exp_sum = input
        .map_axis(Axis(1), |row| {
            row.fold(0., |acc, &ele: &f64| acc + ele.exp())
        })
        .into_shape((input.shape()[0], 1))
        .unwrap();
    input / exp_sum
}

pub fn activation_derivate(activation: Activation, input: &Array2<f64>) -> Array2<f64> {
    match activation {
        Activation::Relu => relu_derivate(input),
        Activation::Tanh => tanh_derivate(input),
    }
}

fn relu_derivate(input: &Array2<f64>) -> Array2<f64> {
    // 'input' is reference, not have ownership
    // therefore, use mapv() instead of mapv_into()
    input.mapv(|ele| if ele >= 0. { 1. } else { 0. })
}

fn tanh_derivate(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|ele| {
        let tanh = {
            let exp_pos = ele.exp();
            let exp_neg = (-ele).exp();
            (exp_pos - exp_neg) / (exp_pos + exp_neg)
        };
        1. - tanh.powi(2)
    })
}
