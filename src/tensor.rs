// inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
use crate::Device;

#[derive(Debug)]
pub struct Shape(Vec<u16>);

#[derive(Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
    device: Device,
}

impl Tensor<f32> {
    fn new(shape: Shape) -> Self {
        Self {
            data: Vec::new(),
            device: Device::CPU,
            shape,
        }
    }
}
