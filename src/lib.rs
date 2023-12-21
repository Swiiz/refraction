use std::array;

use camera::Camera;
use integrator::Integrator;
use maths::radiometry::Spectrum;

pub mod camera;
pub mod maths;
pub mod integrator;

pub struct Refraction {
    integrator: Integrator,
    scene: Scene,
}

impl Refraction {
    pub fn new() -> Self {
        Self {
            integrator: Integrator::new(),
            scene: Scene::new()
        }
    }

    pub fn render(&self, width: usize, height: usize) -> Vec<Vec<Spectrum>> {
        vec![vec![Spectrum::new_black(); width]; height]
    }
}

pub struct Scene {
    camera: Camera,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            camera: Camera::new()
        }
    }
}