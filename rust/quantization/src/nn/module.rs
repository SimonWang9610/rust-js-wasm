use super::array::QuantizedMatrix;
use super::variables::VarStore;


pub trait Module: std::fmt::Debug {
    fn forward(&self, input: &QuantizedMatrix, vs: &VarStore) -> QuantizedMatrix;
}

pub trait ModuleT: std::fmt::Debug {
    fn forward_t(&self, input: &QuantizedMatrix, vs: &mut VarStore) -> QuantizedMatrix;
}