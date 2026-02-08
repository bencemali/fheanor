use feanor_math::{ring::RingBase, rings::matrix::MatrixRing};

use crate::number_ring::{AbstractNumberRing, NumberRingQuotient};

pub trait ModuleLWEMatrixRing: PartialEq + Clone + MatrixRing {

    type NumberRing: AbstractNumberRing;

    type ScalarRing: NumberRingQuotient<NumberRing = Self::NumberRing>;

    // TODO(bence): implement things that are needed by MatrixBFVInstantiation

    fn from_scalar_entries(&self, entries: impl IntoIterator<Item = <Self::ScalarRing as RingBase>::Element>) -> Self::Element;
}