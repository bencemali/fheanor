use feanor_math::rings::matrix::MatrixRing;

use crate::NiceZn;

pub trait LWEMatrixRing: PartialEq + Clone + MatrixRing {
    type BaseRing: NiceZn;

    // TODO(bence): implement things that are needed by MatrixBFVInstantiation
}