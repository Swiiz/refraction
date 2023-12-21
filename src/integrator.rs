use crate::{maths::{radiometry::Spectrum, geometry::Ray, Radiance}, Scene};

pub struct Integrator {
    
}

impl Integrator {
    pub fn new() -> Self {
        Self {}
    }

    fn incomming_radiance(ray: &Ray, scene: &Scene) -> Radiance {
        Radiance::new(Spectrum::new_black())
    }
}