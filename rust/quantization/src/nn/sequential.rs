use super::module::{Module, ModuleT};
use super::array::{Uint8Matrix, Matrix};
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

    pub fn forward_q(&self, input: &Uint8Matrix, vs: &VarStore) -> Matrix {
        let n = self.layers.len(); // how many layers
       
        let xs = self.layers[0].forward_q(input, vs);
        let output = self.layers.iter().take(n-1).skip(1).fold(xs, |xs, layer| layer.forward_q(&xs, vs));

        self.layers[n-1].float_forward(&output, vs)
    }

    pub fn forward(&self, input: &Matrix, vs: &VarStore) -> Matrix {
        let xs = self.layers[0].forward(input, vs);
        self.layers.iter().skip(1).fold(xs, |xs, layer| layer.forward(&xs, vs))
    }


}

impl Sequential {
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    // pub fn add_fn<F>(self, f: F) -> Self
    // where
    //     F: 'static + Fn(&Uint8Matrix, &VarStore) -> Uint8Matrix,
    // {
    //     self.add(func_t(f))
    // }
}

#[derive(Debug)]
pub struct SequentialT {
    layers: Vec<Box<dyn ModuleT>>,
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

    pub fn forward_t(&self, input: &Matrix, vs: &mut VarStore, quantized: bool, ema: f32) -> Matrix {

        let n = self.layers.len(); // how many layers
       
        let xs = self.layers[0].forward_t(input, vs, quantized, ema);
        let output = self.layers.iter().take(n-1).skip(1).fold(xs, |xs, layer| layer.forward_t(&xs, vs, quantized, ema));

        self.layers[n-1].forward_t(&output, vs, false, ema)
    }

    pub fn forward_q(&self, input: &Uint8Matrix, vs: &VarStore) -> Matrix {
        let n = self.layers.len(); // how many layers
       
        let xs = self.layers[0].forward_q(input, vs);
        let output = self.layers.iter().take(n-1).skip(1).fold(xs, |xs, layer| layer.forward_q(&xs, vs));

        self.layers[n-1].forward_float(&output, vs)
    }

    pub fn forward(&self, input: &Matrix, vs: &VarStore) -> Matrix {

        let xs = self.layers[0].forward(input, vs);

        self.layers.iter().skip(1).fold(xs, |xs, layer| layer.forward(&xs, vs))

    }
}