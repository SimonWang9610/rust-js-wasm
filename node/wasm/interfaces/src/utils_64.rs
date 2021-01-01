use ndarray::{Array1, Array2, Axis};

pub fn relu(input: Array2<f64>) -> Array2<f64> {
    input.mapv_into(|ele| if ele >= 0. { ele } else { 0. })
}

pub fn softmax(mut input: Array2<f64>) -> Array2<f64> {
    input.swap_axes(0, 1);
    let exp_sum = input
        .map_axis(Axis(1), |row| {
            row.fold(0., |acc, &ele: &f64| acc + ele.exp())
        })
        .into_shape((input.shape()[0], 1))
        .unwrap();
    let exp_input = input.mapv_into(|ele| ele.exp());
    exp_input / exp_sum
}

pub fn classification(input: Array2<f64>) -> Array1<f64> {
    // [sample, 10]
    input.map_axis(Axis(1), |row| {
        let mut max = (0, 0.);
        for (index, val) in row.iter().enumerate() {
            if *val > max.1 {
                max = (index, *val);
            }
        }
        max.0 as f64
    })
}

pub fn evaluate(output: &Array2<f64>, labels: &Array2<f64>) -> f64 {
    // labels [samples, 10]
    
    let predictions = output.map_axis(Axis(1), |row| {
        let mut max = (0, 0.);
        for (i, ele) in row.iter().enumerate() {
            if *ele > max.1 {
                max = (i, *ele);
            }
        }
        max.0 as f64
    });

    let labels = labels.map_axis(Axis(1), |row| {
        let mut max = (0, 0.);
        for (i, ele) in row.iter().enumerate() {
            if *ele > max.1 {
                max = (i, *ele);
            }
        }
        max.0 as f64
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