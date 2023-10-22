use core::fmt;
// inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
use std::{ops::Add, fmt::Display};

#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
}

impl From<Vec<f32>> for Tensor {
    fn from(value: Vec<f32>) -> Self {
        Self { data: value }
    }
}

impl<const N: usize> From<[u32; N]> for Tensor {
    fn from(value: [u32; N]) -> Self {
        value.into_iter().map(|x| x as f32).collect()
    }
}

impl FromIterator<f32> for Tensor {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<f32>>().into()
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.data.len() <= 16 {
            // f.write_str("Tensor<f32> [")?;
            f.write_str("[")?;
            for (i, x) in self.data.iter().enumerate() {
                f.write_fmt(format_args!("{:.2}", x))?;
                if i != self.data.len() - 1 {
                    f.write_str(", ")?;
                }
            }
            f.write_str("]")?;
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Tensor {
        self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&lhs, &rhs)| lhs + rhs)
            .collect()
    }
}
