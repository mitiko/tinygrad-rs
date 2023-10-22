#![allow(dead_code)] // TODO: remove
pub mod tensor;

#[derive(Debug)]
pub enum Device {
    CPU,
    GPU,
}
