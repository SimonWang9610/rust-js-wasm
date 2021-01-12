extern crate image;
use ndarray::{Array, Array2, Ix2, Axis};
use rand::prelude::*;
use std::f32::consts::E;
use std::path::Path;


pub fn one_hot(labels: Array2<f32>, cols: usize) -> Array2<f32> {
    let rows = labels.shape()[0];
    let mut data = vec![];
    for item in labels.into_iter() {
        let mut row = Vec::with_capacity(cols);
        for index in 0..cols {
            if index as f32 == *item {
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

pub fn compute_loss(output: &Array2<f32>, labels: &Array2<f32>) -> f32 {
    // output [sample, 10]
    // labels [sample, 10]
    let average = -1. / labels.shape()[0] as f32;
    output
        .into_iter()
        .zip(labels.iter())
        .fold(0., |acc, (o, l)| acc + l * o.log(E))
        * average
}

pub fn loss(output: &Array2<f32>, labels: &Array2<f32>) -> f32 {
    let average = -1. / labels.shape()[1] as f32;

    output
        .into_iter()
        .zip(labels.iter())
        .fold(0., |acc, (o, l)| acc + l * o.log(E))
        * average
}

// transform image to specific shape
pub fn transform(path: &str) -> Array2<f32> {
    let img = image::open(Path::new(path)).unwrap().into_luma();
    let pixels: Vec<f32> = img.into_iter().map(|x| *x as f32 / 255.).collect();

    Array::from_shape_vec(Ix2(1, 28 * 28), pixels).unwrap()
}

pub fn evaluate(output: &Array2<f32>, labels: &Array2<f32>, axis: usize) -> f32 {
    // axis = 0: labels [10, sample]
    // axis = 1: labels [sample, 10]
    
    let predictions = output.map_axis(Axis(axis), |row| {
        let mut max = (0, 0.);
        for (i, ele) in row.iter().enumerate() {
            if *ele > max.1 {
                max = (i, *ele);
            }
        }
        max.0 as f32
    });

    let labels = labels.map_axis(Axis(axis), |row| {
        let mut max = (0, 0.);
        for (i, ele) in row.iter().enumerate() {
            if *ele > max.1 {
                max = (i, *ele);
            }
        }
        max.0 as f32
    });

    predictions
        .into_iter()
        .zip(labels.into_iter())
        .fold(
            0.,
            |acc, (prediction, label)| {
                if prediction == label {
                    acc + 1.
                } else {
                    acc
                }
            },
        )
}


pub fn _quantize(arr: &Array2<i32>) -> (Array2<u8>, f32, u8) {
    
    let (min, max) = arr.iter().fold((0, 0), |acc, ele| {

        if &acc.0 > ele {
            (*ele, acc. 1)
        } else if &acc.1 < ele {
            (acc.0, *ele)
        } else {
            acc
        }
    });

    let factor = (max as f32 - min as f32) / 255.;
    let zero = (255. - max as f32 / factor).round() as u8;

    (
        arr.map(|ele| {
            (*ele as f32 / factor).round() as u8 + zero
        }),
        factor,
        zero,
    )
}

pub fn min_max(v: &Vec<f32>) -> (f32, f32) {
    
    v.iter().fold((0., 0.), |acc, ele| {

        if &acc.0 > ele {
            (*ele, acc. 1)
        } else if &acc.1 < ele {
            (acc.0, *ele)
        } else {
            acc
        }
    })
}