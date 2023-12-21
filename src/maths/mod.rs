use std::marker::PhantomData;

use self::radiometry::Spectrum;

pub type Float = f32;

pub mod transforms;
pub mod radiometry;
pub mod geometry;
pub mod num;

pub fn lerp<V: std::ops::Mul<Float, Output=V> + std::ops::Add<Output=V>>(t: Float, a: V, b: V) -> V {
    a * (1.0 - t) + b * t
}

pub struct Proxy<T, M> {
    inner: T,
    _marker: PhantomData<M>
}

pub struct RadianceMarker;
pub type Radiance = Proxy<Spectrum, RadianceMarker>;

impl<T, M> Proxy<T, M> {
    pub fn new(inner: T) -> Self {
        Self {
            inner, _marker: PhantomData
        }
    }
}

impl<T: std::ops::Add<T, Output=T>, M> std::ops::Add for Proxy<T, M> {
    type Output = Proxy<T, M>;
    fn add(self, rhs: Proxy<T, M>) -> Self::Output {
        Self::new(self.inner + rhs.inner)
    }
}