use super::array::{Uint8Matrix};
use super::variables::VarStore;


pub struct Func<'a> {
    f: Box<dyn 'a + Fn(&Uint8Matrix, &VarStore) -> Uint8Matrix>,
}


pub fn func_t<'a, F>(f: F) -> Func<'a>
where
    F: 'a + Fn(&Uint8Matrix, &VarStore) -> Uint8Matrix,
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

// impl<'a> Module for Func<'a> {
//     fn forward(&self, input: &Uint8Matrix, vs: &VarStore) -> Uint8Matrix {
//         (*self.f)(input, vs)
//     }
// }