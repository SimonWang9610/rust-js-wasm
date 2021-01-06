

pub mod nn;
pub mod model;

#[cfg(test)]
mod tests {
    use super::nn::array::Matrix;
    use super::nn::variables::VarStore;
    use super::nn::quantize::Dequantization;
    /* use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal; */
    #[test]
    fn it_works() {
        let config = vec![784, 500, 350, 200, 100, 50, 10];
    }
}
