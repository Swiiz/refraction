use crate::maths::Float;

pub struct Transform {
    matrix: Matrix4x4,
    inverse: Matrix4x4,
}

pub struct Matrix4x4 {
    elements: [[Float; 4]; 4]
}