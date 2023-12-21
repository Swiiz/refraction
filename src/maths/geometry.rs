use std::ops::{Deref, DerefMut, Add, AddAssign, Mul, MulAssign, Div, DivAssign, Sub, SubAssign};

use super::{Float, num::{Number, min, max}};

#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub origin: Point3f,
    pub direction: UnitVector3f,
}

impl Ray {
    pub fn new(origin: Point3f, direction: UnitVector3f) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: Float) -> Point3f {
        self.origin + *self.direction * t
    }
}

pub type Vector2f = Vector2<Float>;
pub type Vector2i = Vector2<i32>;
pub type Vector3f = Vector3<Float>;
pub type Vector3i = Vector3<i32>;

pub type Point2f = Point2<Float>;
pub type Point2i = Point2<i32>;
pub type Point3f = Point3<Float>;
pub type Point3i = Point3<i32>;

pub type Bounds2f = Bounds2<Float>;
pub type Bounds2i = Bounds2<i32>;
pub type Bounds3f = Bounds3<Float>;
pub type Bounds3i = Bounds3<i32>;

#[derive(Clone, Debug)]
pub struct CoordinateSystem {
    pub i: UnitVector3f,
    pub j: UnitVector3f,
    pub k: UnitVector3f
}
pub fn coordinate_system(i: UnitVector3f) -> CoordinateSystem {
    let j = if i.x.abs().greater(i.y.abs()) {
        UnitVector3f::new(-i.z, 0.0, i.x)
    } else {
        UnitVector3f::new(0.0, i.z, -i.y)
    };
    let k = UnitVector3f(Vector3f::cross(i.deref(), j.deref())); // We know the cross product is a unit vector
    CoordinateSystem {
        i, j, k
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vector2<T: Number> {
    pub x: T,
    pub y: T,
}

#[derive(Copy, Clone, Debug)]
pub struct Vector3<T: Number> {
    pub x: T,
    pub y: T,
    pub z: T, 
}

impl<T: Number> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y}
    }

    pub fn dot(&self, v: &Vector2<T>) -> T {
        self.x * v.x + self.y * v.y
    }

    pub fn length_squared(&self) -> T {
        self.dot(self)
    }

    pub fn length(&self) -> Float {
        self.length_squared().sqrt()
    }

    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }
}

impl<T: Number> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z}
    }

    pub fn dot(&self, v: &Vector3<T>) -> T {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    pub fn cross(&self, v: &Vector3<T>) -> Vector3<T> {
        Vector3::new((self.y * v.z) - (self.z * v.y),
                      (self.z * v.x) - (self.x * v.z),
                      (self.x * v.y) - (self.y * v.x))
    }

    pub fn length_squared(&self) -> T  {
        self.clone().dot(self)
    }

    pub fn length(&self) -> Float {
        self.length_squared().sqrt()
    }

    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }
}

impl Vector2f {
    pub fn has_nans(&self) -> bool {
        Float::is_nan(self.x) || Float::is_nan(self.y)
    }

    pub fn normalize(self) -> UnitVector2f {
        UnitVector2f(self.clone() / self.length())
    }
}

impl Vector3f {
    pub fn has_nans(&self) -> bool {
        Float::is_nan(self.x) || Float::is_nan(self.y) || Float::is_nan(self.z)
    }

    pub fn abs_dot(self, v: &Vector3f) -> Float {
        self.dot(v).abs()
    }

    pub fn normalize(&self) -> UnitVector3f {
        UnitVector3f(self.clone() / self.length())
    }
}

impl<T: Number> Add for Vector2<T> {
    type Output = Vector2<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: Number> Add for Vector3<T> {
    type Output = Vector3<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Number> AddAssign for Vector2<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: Number> AddAssign for Vector3<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}


impl<T: Number> Sub for Vector2<T> {
    type Output = Vector2<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: Number> Sub for Vector3<T> {
    type Output = Vector3<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Number> SubAssign for Vector2<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: Number> SubAssign for Vector3<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl<T: Number> Mul<T> for Vector2<T> {
    type Output = Vector2<T>;
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: Number> Mul<T> for Vector3<T> {
    type Output = Vector3<T>;
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: Number> MulAssign<T> for Vector2<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: Number> MulAssign<T> for Vector3<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl<T: Number> Div<T> for Vector2<T> {
    type Output = Vector2<T>;
    fn div(self, rhs: T) -> Self::Output {
        let inv = T::one() / rhs;
        Self {
            x: self.x * inv,
            y: self.y * inv,
        }
    }
}

impl<T: Number> Div<T> for Vector3<T>  {
    type Output = Vector3<T>;
    fn div(self, rhs: T) -> Self::Output {
        let inv = T::one() / rhs;
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }
}

impl<T: Number> DivAssign<T> for Vector2<T>  {
    fn div_assign(&mut self, rhs: T) {
        let inv = T::one() / rhs;
        self.x *= inv;
        self.y *= inv;
    }
}

impl<T: Number> DivAssign<T> for Vector3<T>  {
    fn div_assign(&mut self, rhs: T) {
        let inv = T::one() / rhs;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UnitVector2f(Vector2f);
#[derive(Copy, Clone, Debug)]
pub struct UnitVector3f(Vector3f);

impl UnitVector2f {
    pub fn new(x: Float, y: Float) -> Self {
        Vector2f::new(x, y).normalize()
    }
}

impl UnitVector3f {
    pub fn new(x: Float, y: Float, z: Float) -> Self {
        Vector3f::new(x, y, z).normalize()
    }
}

impl Into<Vector2f> for UnitVector2f {
    fn into(self) -> Vector2f {
        *self
    }
}

impl Into<Vector3f> for UnitVector3f {
    fn into(self) -> Vector3f {
        *self
    }
}

impl Deref for UnitVector2f {
    type Target = Vector2f;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UnitVector2f {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for UnitVector3f {
    type Target = Vector3f;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UnitVector3f {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Copy, Clone, Debug)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T, 
}

impl<T: Number> Point2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y}
    }

    pub fn distance_squared(&self, v: &Self) -> T  {
        (*self - *v).length_squared()
    }
}

impl<T: Number> Point3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z}
    }

    pub fn distance_squared(&self, v: &Self) -> T  {
        (*self - *v).length_squared()
    }
}

impl Point2f {
    fn has_nans(&self) -> bool {
        Float::is_nan(self.x) || Float::is_nan(self.y)
    }

    fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    pub fn distance(&self, v: &Self) -> Float {
        self.distance_squared(v).sqrt()
    }
}

impl Point3f {
    pub fn has_nans(&self) -> bool {
        Float::is_nan(self.x) || Float::is_nan(self.y) || Float::is_nan(self.z)
    }

    pub fn distance(&self, v: &Self) -> Float {
        self.distance_squared(v).sqrt()
    }
}

impl<T: Number> Into<Vector2<T>> for Point2<T> {
    fn into(self) -> Vector2<T> {
        Vector2::new(self.x, self.y)
    }
}

impl<T: Number> Into<Vector3<T>> for Point3<T> {
    fn into(self) -> Vector3<T> {
        Vector3::new(self.x, self.y, self.z)
    }
}

impl<T: Number> From<Vector2<T>> for Point2<T> {
    fn from(v: Vector2<T>) -> Self {
        Self::new(v.x, v.y)
    }
}

impl<T: Number> From<Vector3<T>> for Point3<T> {
    fn from(v: Vector3<T>) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

impl<T: Number> Add for Point2<T> {
    type Output = Point2<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: Number> Add for Point3<T> {
    type Output = Point3<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Number> Add<Vector2<T>> for Point2<T> {
    type Output = Point2<T>;
    fn add(self, rhs: Vector2<T>) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: Number> Add<Vector3<T>> for Point3<T> {
    type Output = Point3<T>;
    fn add(self, rhs: Vector3<T>) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Number> AddAssign for Point2<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: Number> AddAssign for Point3<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T: Number> AddAssign<Vector2<T>> for Point2<T> {
    fn add_assign(&mut self, rhs: Vector2<T>) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: Number> AddAssign<Vector3<T>> for Point3<T> {
    fn add_assign(&mut self, rhs: Vector3<T>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T: Number> Sub for Point2<T> {
    type Output = Vector2<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Vector2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: Number> Sub for Point3<T> {
    type Output = Vector3<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Vector3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Number> Sub<Vector2<T>> for Point2<T> {
    type Output = Point2<T>;
    fn sub(self, rhs: Vector2<T>) -> Self::Output {
        Point2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: Number> Sub<Vector3<T>> for Point3<T> {
    type Output = Point3<T>;
    fn sub(self, rhs: Vector3<T>) -> Self::Output {
        Point3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Number> SubAssign<Vector2<T>> for Point2<T> {
    fn sub_assign(&mut self, rhs: Vector2<T>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: Number> SubAssign<Vector3<T>> for Point3<T> {
    fn sub_assign(&mut self, rhs: Vector3<T>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Normal3f {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Normal3f {
    pub fn new(x: Float, y: Float, z: Float) -> Self {
        Self { x, y, z }
    }
}

impl From<Vector3f> for Normal3f {
    fn from(v: Vector3f) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

impl Into<Vector3f> for Normal3f {
    fn into(self) -> Vector3f {
        Vector3f::new(self.x, self.y, self.z)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Bounds2<T> {
    pmin: Point2<T>,
    pmax: Point2<T>,
}

#[derive(Copy, Clone, Debug)]
pub struct Bounds3<T> {
    pmin: Point3<T>,
    pmax: Point3<T>,
}

impl<T: Number> Bounds2<T> {
    pub fn new_point(p: Point2<T>) -> Self  {
        Self { pmin: p, pmax: p }
    }

    pub fn new(p0: Point2<T>, p1: Point2<T>) -> Self  {
        Self {
            pmin: Point2::new(min(p0.x, p1.x), min(p0.y, p1.y)),
            pmax: Point2::new(max(p0.x, p1.x), max(p0.y, p1.y))
        }
    }

    pub fn corner(&self, corner: u8) -> Point2<T>  {
        let x = if corner & 1 == 0 { self.pmin.x } else { self.pmax.x };
        let y = if corner & 2 == 0 { self.pmin.y } else { self.pmax.y };
        Point2::new(x, y)
    }

    pub fn union(&self, b: &Self) -> Self  {
        Bounds2::new(Point2::new(min(b.pmin.x, self.pmin.x),
                                min(b.pmin.y, self.pmin.y)),
                      Point2::new(max(b.pmax.x, self.pmax.x),
                                max(b.pmax.y,self.pmax.y)))
    }

    pub fn intersect(&self, b: &Self) -> Self {
        Bounds2::new(Point2::new(max(self.pmin.x, b.pmin.x),
                                max(self.pmin.y, b.pmin.y)),
                      Point2::new(min(self.pmax.x, b.pmax.x),
                                min(self.pmax.y, b.pmax.y)))
    }

    pub fn overlaps(&self, b: &Self) -> bool {
       let x = (self.pmax.x.greater_equal(b.pmin.x)) && (self.pmin.x.lower_equal(b.pmax.x));
       let y = (self.pmax.y.greater_equal(b.pmin.y)) && (self.pmin.y.lower_equal(b.pmax.y));
        x && y
    }

    pub fn inside(&self, p: Point2<T>) -> bool {
      p.x.greater_equal(self.pmin.x) && p.x.lower_equal(self.pmax.x) &&
           p.y.greater_equal(self.pmin.y) && p.y.lower_equal(self.pmax.y)
    }

    pub fn inside_exclusive(&self, p: Point2<T>) -> bool {
        p.x.greater(self.pmin.x) && p.x.lower(self.pmax.x) &&
            p.y.greater(self.pmin.y) && p.y.lower(self.pmax.y)
    }

    pub fn expand(&self, delta: T) -> Self {
        let delta = Vector2::new(delta, delta);
        Self::new(self.pmin - delta, self.pmax + delta)
    }

    pub fn diagonal(&self) -> Vector2<T> {
        self.pmax - self.pmin
    }

    pub fn surface_area(&self) -> T {
        let d = self.diagonal();
        d.x * d.y
    }

    pub fn lerp(&self, t: Float) -> Point2<T> where Vector2<T>: Mul<Float, Output=Vector2<T>> {
        super::lerp::<Vector2<T>>(t, self.pmin.into(), self.pmax.into()).into()
    }

    pub fn offset(&self, p: Point2<T>) -> Vector2<T> {
        let mut o = p - self.pmin;
        o.x /= self.pmax.x - self.pmin.x;
        o.y /= self.pmax.y - self.pmin.y;
        o
    }
}

impl<T: Number> Bounds3<T> {
    pub fn new_point(p: Point3<T>) -> Self  {
        Self { pmin: p, pmax: p }
    }

    pub fn new(p0: Point3<T>, p1: Point3<T>) -> Self  {
        Self {
            pmin: Point3::new(min(p0.x, p1.x), min(p0.y, p1.y), min(p0.z, p1.z)),
            pmax: Point3::new(max(p0.x, p1.x), max(p0.y, p1.y), max(p0.z, p1.z))
        }
    }

    pub fn corner(&self, corner: u8) -> Point3<T>  {
        let x = if corner & 1 == 0 { self.pmin.x } else { self.pmax.x };
        let y = if corner & 2 == 0 { self.pmin.y } else { self.pmax.y };
        let z = if corner & 4 == 0 { self.pmin.z } else { self.pmax.z };
        Point3::new(x, y, z)
    }

    pub fn union(&self, b: &Self) -> Self  {
        Bounds3::new(Point3::new(min(b.pmin.x, self.pmin.x),
                                min(b.pmin.y, self.pmin.y),
                                min(b.pmin.z, self.pmin.z)),
                      Point3::new(max(b.pmax.x, self.pmax.x),
                                max(b.pmax.y,self.pmax.y),
                                max(b.pmax.z, self.pmax.z)))
    }

    pub fn intersect(&self, b: &Self) -> Self {
        Bounds3::new(Point3::new(max(self.pmin.x, b.pmin.x),
                                max(self.pmin.y, b.pmin.y),
                                max(self.pmin.z, b.pmin.z)),
                      Point3::new(min(self.pmax.x, b.pmax.x),
                                min(self.pmax.y, b.pmax.y),
                                min(self.pmax.z, b.pmax.z)))
    }

    pub fn overlaps(&self, b: &Self) -> bool {
       let x = (self.pmax.x.greater_equal(b.pmin.x)) && (self.pmin.x.lower_equal(b.pmax.x));
       let y = (self.pmax.y.greater_equal(b.pmin.y)) && (self.pmin.y.lower_equal(b.pmax.y));
       let z = (self.pmax.z.greater_equal(b.pmin.z)) && (self.pmin.z.lower_equal(b.pmax.z));
        x && y && z
    }

    pub fn inside(&self, p: Point3<T>) -> bool {
      p.x.greater_equal(self.pmin.x) && p.x.lower_equal(self.pmax.x) &&
          p.y.greater_equal(self.pmin.y) && p.y.lower_equal(self.pmax.y) &&
           p.z.greater_equal(self.pmin.z) && p.z.lower_equal(self.pmax.z)
    }

    pub fn inside_exclusive(&self, p: Point3<T>) -> bool {
        p.x.greater(self.pmin.x) && p.x.lower(self.pmax.x) &&
            p.y.greater(self.pmin.y) && p.y.lower(self.pmax.y) &&
            p.z.greater(self.pmin.z) && p.z.lower(self.pmax.z)
    }

    pub fn expand(&self, delta: T) -> Self {
        let delta = Vector3::new(delta, delta, delta);
        Self::new(self.pmin - delta, self.pmax + delta)
    }

    pub fn diagonal(&self) -> Vector3<T> {
        self.pmax - self.pmin
    }

    pub fn surface_area(&self) -> T {
        let d = self.diagonal();
        (T::one() + T::one()) * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    pub fn volume(&self) -> T {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    pub fn lerp(&self, t: Float) -> Point3<T> where Vector3<T>: Mul<Float, Output=Vector3<T>> {
        super::lerp::<Vector3<T>>(t, self.pmin.into(), self.pmax.into()).into()
    }

    pub fn offset(&self, p: Point3<T>) -> Vector3<T> {
        let mut o = p - self.pmin;
        o.x /= self.pmax.x - self.pmin.x;
        o.y /= self.pmax.y - self.pmin.y;
        o.z /= self.pmax.z - self.pmin.z;
        o
    }
}

impl<T: Number> Into<Bounds2<T>> for Point2<T> {
    fn into(self) -> Bounds2<T> {
        Bounds2::new_point(self)
    }
}

impl<T: Number> Into<Bounds3<T>> for Point3<T> {
    fn into(self) -> Bounds3<T> {
        Bounds3::new_point(self)
    }
}