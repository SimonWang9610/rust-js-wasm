use ndarray::{Array1, Array2, Axis};

pub fn relu(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|ele| if ele >= 0. { ele } else { 0. })
}

pub fn softmax(mut input: Array2<f32>) -> Array2<f32> {
    input.swap_axes(0, 1);
    let exp_sum = input
        .map_axis(Axis(1), |row| {
            row.fold(0., |acc, &ele: &f32| acc + ele.exp())
        })
        .into_shape((input.shape()[0], 1))
        .unwrap();
    let exp_input = input.mapv_into(|ele| ele.exp());
    exp_input / exp_sum
}

pub fn classification(input: Array2<f32>) -> Array1<f32> {
    // [sample, 10]
    input.map_axis(Axis(1), |row| {
        let mut max = (0, 0.);
        for (index, val) in row.iter().enumerate() {
            if *val > max.1 {
                max = (index, *val);
            }
        }
        max.0 as f32
    })
}

pub fn evaluate(output: &Array2<f32>, labels: &Array2<f32>) -> f32 {
    // labels [samples, 10]
    
    let predictions = output.map_axis(Axis(1), |row| {
        let mut max = (0, 0.);
        for (i, ele) in row.iter().enumerate() {
            if *ele > max.1 {
                max = (i, *ele);
            }
        }
        max.0 as f32
    });

    let labels = labels.map_axis(Axis(1), |row| {
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

pub fn dequantize(weights: Vec<u8>, neurons: usize, prev: usize, zero: f32, factor: f32) -> Array2<f32> {
    let weights_float: Vec<f32> = weights.iter().map(|ele| {
        (*ele as f32 - zero) * factor
    }).collect();

    Array2::from_shape_vec((neurons, prev), weights_float).unwrap()
}