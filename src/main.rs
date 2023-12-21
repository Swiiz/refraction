use image::{ImageBuffer, Rgb};
use refraction::Refraction;

fn main() {
    let width = 720;
    let height = 480;
    let raw_img = Refraction::new().render(width, height);
    let img = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        let rgb_f = raw_img[y as usize][x as usize].rgb();
        Rgb([(rgb_f[0] * 255.999) as u8, (rgb_f[1] * 255.999) as u8, (rgb_f[2] * 255.999) as u8])
    });
    img.save("output.png").unwrap();
}