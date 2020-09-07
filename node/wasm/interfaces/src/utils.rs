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
