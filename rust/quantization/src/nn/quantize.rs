use ndarray::Array2;


pub trait Quantization<T> {
    fn quantize(&self) -> T;
}

pub trait Dequantization<T, U> {
    
    fn dequantize_matmul(&self, qm: &T) -> U;
    
    fn dequantize(&self) -> U;
}