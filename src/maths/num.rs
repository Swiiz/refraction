use std::ops::{Add, AddAssign, Mul, MulAssign, Div, DivAssign, Sub, SubAssign};

use super::Float;

pub fn max<T: Number>(a: T, b: T) -> T {
    if a.greater(b) { a } else { b }
}
pub fn min<T: Number>(a: T, b: T) -> T {
    if a.lower(b) { a } else { b }
}

pub trait Number
    where Self: MaybeNeg
    + Copy
    + Add<Output=Self>
    + AddAssign
    + Sub<Output=Self>
    + SubAssign
    + Mul<Output=Self>
    + MulAssign
    + Div<Output=Self>
    + DivAssign {
    fn zero() -> Self;
    fn one() -> Self;
    fn equal(self, other: Self) -> bool;
    fn greater(self, other: Self) -> bool;
    fn greater_equal(self, other: Self) -> bool {
        self.greater(other) || self.equal(other)
    }
    fn lower(self, other: Self) -> bool {
        !self.greater(other)
    }
    fn lower_equal(self, other: Self) -> bool {
        self.lower(other) || self.equal(other)
    }
    fn sqrt(self) -> Float;
    fn abs(self) -> Self;
}

macro_rules! impl_number_for {
    ($($t:ty),+) => {
        $(
            impl Number for $t {
                fn zero() -> Self {
                    0 as $t
                }

                fn one() -> Self {
                    1 as $t
                }

                fn equal(self, other: Self) -> bool {
                    self == other
                }

                fn greater(self, other: Self) -> bool {
                    self > other
                }

                fn sqrt(self) -> Float {
                    (self as Float).sqrt()
                }

                fn abs(self) -> Self {
                    if self.greater(0 as $t) || !self.equal(0 as $t) { self } else { self * <Self as MaybeNeg>::minus_one().unwrap() }
                }
            }
        )+
    };
}

impl_number_for!(f32, f64, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

pub trait MaybeNeg {
    fn minus_one() -> Option<Self> where Self: Sized;
}

macro_rules! impl_trynegate_for {
    ($e:expr; $f:expr; $($t:ty),+) => {
        $(
            impl MaybeNeg for $t {
                fn minus_one() -> Option<Self> {
                    ($e).map(|_| $f as $t)
                }
            }
        )+
    };
}

impl_trynegate_for!(Some(Self::zero()); -1; f32, f64, i8, i16, i32, i64, i128);
impl_trynegate_for!(None::<Self>; 0; u8, u16, u32, u64, u128);