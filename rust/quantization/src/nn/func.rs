use super::array::{QuantizedMatrix};
use super::module::{Module, ModuleT};
use super::variables::VarStore;


pub struct Func<'a> {
    f: Box<dyn 'a + Fn(&QuantizedMatrix, &VarStore) -> QuantizedMatrix>,
}


pub fn func_t<'a, F>(f: F) -> Func<'a>
where
    F: 'a + Fn(&QuantizedMatrix, &VarStore) -> QuantizedMatrix,
{
    Func {
        f: Box::new(f)
    }
}

impl<'a> std::fmt::Debug for Func<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Func")
    }
}

impl<'a> Module for Func<'a> {
    fn forward(&self, input: &QuantizedMatrix, vs: &VarStore) -> QuantizedMatrix {
        (*self.f)(input, vs)
    }
}