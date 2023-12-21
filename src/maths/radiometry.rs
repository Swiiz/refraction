use super::Float;

pub type Spectrum = RGBSpectrum;

pub type RGBSpectrum = CoefSpectrum<3>;

impl RGBSpectrum {
    pub fn rgb(&self) -> [Float; 3] {
        [
            self.coefficients[0], self.coefficients[1], self.coefficients[2], 
        ]
    }
}

#[derive(Clone)]
pub struct CoefSpectrum<const N_COEFS: usize> {
    coefficients: [Float; N_COEFS], 
}

impl<const N_COEFS: usize> CoefSpectrum<N_COEFS> {
    pub fn new_black() -> Self {
        Self {
            coefficients: [0.0; N_COEFS]
        }
    }
}

impl<const N_COEFS: usize> std::ops::Add<CoefSpectrum<N_COEFS>> for CoefSpectrum<N_COEFS> {
    type Output = Self;
    fn add(self, rhs: CoefSpectrum<N_COEFS>) -> Self::Output {
        let mut coefficients = [0.0; N_COEFS];
        for i in 0..N_COEFS {
            coefficients[i] = self.coefficients[i] + rhs.coefficients[i];
        }
        CoefSpectrum {
            coefficients
        }
    }
}

impl<const N_COEFS: usize> std::ops::AddAssign<CoefSpectrum<N_COEFS>> for CoefSpectrum<N_COEFS> {
    fn add_assign(&mut self, rhs: CoefSpectrum<N_COEFS>) {
        for i in 0..N_COEFS {
            self.coefficients[i] += rhs.coefficients[i];
        }
    }
}

impl<const N_COEFS: usize> std::ops::Mul<CoefSpectrum<N_COEFS>> for CoefSpectrum<N_COEFS> {
    type Output = Self;
    fn mul(self, rhs: CoefSpectrum<N_COEFS>) -> Self::Output {
        let mut coefficients = [0.0; N_COEFS];
        for i in 0..N_COEFS {
            coefficients[i] = self.coefficients[i] * rhs.coefficients[i];
        }
        CoefSpectrum {
            coefficients
        }
    }
}

impl<const N_COEFS: usize> std::ops::MulAssign<CoefSpectrum<N_COEFS>> for CoefSpectrum<N_COEFS> {
    fn mul_assign(&mut self, rhs: CoefSpectrum<N_COEFS>) {
        for i in 0..N_COEFS {
            self.coefficients[i] *= rhs.coefficients[i];
        }
    }
}

impl<const N_COEFS: usize> std::ops::Mul<Float> for CoefSpectrum<N_COEFS> {
    type Output = Self;
    fn mul(self, rhs: Float) -> Self::Output {
        let mut coefficients = [0.0; N_COEFS];
        for i in 0..N_COEFS {
            coefficients[i] = self.coefficients[i] * rhs;
        }
        CoefSpectrum {
            coefficients
        }
    }
}

impl<const N_COEFS: usize> std::ops::MulAssign<Float> for CoefSpectrum<N_COEFS> {
    fn mul_assign(&mut self, rhs: Float) {
        for i in 0..N_COEFS {
            self.coefficients[i] *= rhs;
        }
    }
}