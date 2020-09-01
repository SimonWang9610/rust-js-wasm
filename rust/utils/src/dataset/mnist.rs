extern crate ndarray;

use crate::utils::one_hot;
use ndarray::{Array, Array2, Ix2};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub fn load_mnist<P: AsRef<Path>>(
    path: Vec<P>,
) -> ((Array2<f64>, Array2<f64>), (Array2<f64>, Array2<f64>)) {
    let mut path_iter = path.into_iter();
    let (train_x, num_image_train): (Vec<f64>, usize) = load_images(path_iter.next().unwrap());
    let (train_y, num_label_train): (Vec<f64>, usize) = load_labels(path_iter.next().unwrap());

    let (test_x, num_image_test): (Vec<f64>, usize) = load_images(path_iter.next().unwrap());

    let (test_y, num_label_test): (Vec<f64>, usize) = load_labels(path_iter.next().unwrap());

    let as_arr = Array::from_shape_vec; // create am array withe the given shape from a vector
    let x_train = as_arr(Ix2(num_image_train, 28 * 28), train_x).unwrap();
    let y_train = as_arr(Ix2(num_label_train, 1), train_y).unwrap();
    let x_test = as_arr(Ix2(num_image_test, 28 * 28), test_x).unwrap();
    let y_test = as_arr(Ix2(num_label_test, 1), test_y).unwrap();
    (
        (x_train / 255., one_hot(y_train, 10)),
        (x_test / 255., one_hot(y_test, 10)),
    )
}

fn load_images<P: AsRef<Path>>(path: P) -> (Vec<f64>, usize) {
    let file = File::open(path).expect("please sure the data file exists");
    let ref mut buf_reader = io::BufReader::new(file);
    let magic = read_be_u32(buf_reader);

    if magic != 2051 {
        panic!("Invalid magic number, expect 2051, got {}", magic)
    }

    let num_image = read_be_u32(buf_reader) as usize;
    let rows = read_be_u32(buf_reader) as usize;
    let cols = read_be_u32(buf_reader) as usize;

    assert!(rows == 28 && cols == 28);

    let mut buf: Vec<u8> = vec![0 as u8; num_image * rows * cols];
    let _ = buf_reader.read_exact(buf.as_mut());
    let ret: Vec<f64> = buf.into_iter().map(|x| (x as f64) / 255.).collect();
    (ret, num_image)
}

fn load_labels<P: AsRef<Path>>(path: P) -> (Vec<f64>, usize) {
    let ref mut buf_reader =
        io::BufReader::new(File::open(path).expect("please sure label data exists"));
    let magic = read_be_u32(buf_reader);

    if magic != 2049 {
        panic!("invalid magic number, expect 2049, got {}", magic)
    }

    let num_label = read_be_u32(buf_reader) as usize;
    let mut buf: Vec<u8> = vec![0 as u8; num_label];
    let _ = buf_reader.read_exact(buf.as_mut());
    let ret: Vec<f64> = buf.into_iter().map(|x| x as f64).collect();
    (ret, num_label)
}

fn read_be_u32<T: Read>(reader: &mut T) -> u32 {
    let mut buf = [0 as u8; 4];
    let _ = reader.read_exact(&mut buf);
    u32::from_be_bytes(buf)
}