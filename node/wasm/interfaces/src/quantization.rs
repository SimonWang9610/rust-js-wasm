use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct IntLayer {
    pub weights: Vec<u8>,
    pub bias: Vec<i32>,
    pub neurons: usize,
    pub prev: usize,
    pub name: usize,
    pub end: bool,
    pub factor: f32,
    pub factor_b: f32,
    pub zero: u8,
    pub max: f32,
    pub min: f32,
}

