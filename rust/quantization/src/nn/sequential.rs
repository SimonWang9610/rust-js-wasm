use super::module::{Module, ModuleT};
use super::array::{QuantizedMatrix, Matrix};
use super::func::func_t;
use super::variables::VarStore;

#[derive(Debug)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}


impl Sequential {

    pub fn seq() -> Self {
        Sequential {
            layers: vec![]
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {

    fn forward(&self, input: &QuantizedMatrix, vs: &VarStore) -> QuantizedMatrix {
        let xs = self.layers[0].forward(input, vs);
        self.layers.iter().fold(xs, |xs, layer| layer.forward(&xs, vs))
    }
}


impl Sequential {
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&QuantizedMatrix, &VarStore) -> QuantizedMatrix,
    {
        self.add(func_t(f))
    }
}


#[derive(Debug)]
pub struct SequentialT {
    layers: Vec<Box<dyn ModuleT>>,
}

// pub fn seq_t() -> SequentialT {
//     SequentialT {
//         layers: vec![]
//     }
// }

impl ModuleT for SequentialT {
    fn forward_t(&self, input: &QuantizedMatrix, vs: &mut VarStore) -> QuantizedMatrix {
        let xs = self.layers[0].forward_t(input, vs);
        self.layers.iter().skip(1).fold(xs, |xs, layer| layer.forward_t(&xs, vs))
    }
}

impl SequentialT {

    pub fn seq() -> Self {
        SequentialT {
            layers: vec![],
        }
    }

    pub fn add<M: ModuleT + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}