use std::fmt::Debug;

use feanor_math::algorithms::discrete_log::{multiplicative_order, Subgroup};
use feanor_math::algorithms::eea::inv_crt;
use feanor_math::algorithms::int_factor::{factor, is_prime_power};
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::group::{AbelianGroupBase, AbelianGroupStore, GroupValue, MultGroup, MultGroupEl};
use feanor_math::integer::int_cast;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::zn::zn_64::{Zn, ZnEl};
use feanor_math::ring::*;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::serialization::{DeserializeWithRing, SerializeOwnedWithRing};
use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{euler_phi, ZZbig, ZZi64};

///
/// Represents the Galois group of the `m`-th cyclotomic number field by
/// its isomorphic subgroup of `(Z/mZ)*`.
/// 
#[derive(Clone, Serialize, Deserialize)]
pub struct CyclotomicGaloisGroupBase {
    group: MultGroup<Zn>,
    order: usize
}

pub type CyclotomicGaloisGroup = GroupValue<CyclotomicGaloisGroupBase>;

impl CyclotomicGaloisGroupBase {

    pub fn new(m: u64) -> GroupValue<Self> {
        let ring = Zn::new(m);
        let group = MultGroup::new(ring);
        let order = euler_phi(&factor(ZZi64, m as i64)) as usize;
        return GroupValue::from(Self {
            order: order,
            group: group
        });
    }

    pub fn subgroup<I>(self, generators: I) -> Subgroup<CyclotomicGaloisGroup>
        where I: IntoIterator<Item = GaloisGroupEl>
    {
        let order = int_cast(self.order as i64, ZZbig, ZZi64);
        Subgroup::new(GroupValue::from(self), order, generators.into_iter().collect())
    }

    pub fn full_subgroup(self) -> Subgroup<CyclotomicGaloisGroup> {
        let order = int_cast(self.order as i64, ZZbig, ZZi64);
        let mut generators = Vec::new();
        for (p, e) in factor(ZZi64, self.m() as i64) {
            let pe = ZZi64.pow(p, e);
            let rest = ZZi64.checked_div(&(self.m() as i64), &pe).unwrap();
            if p == 2 {
                generators.push(inv_crt(-1, 1, &pe, &rest, ZZi64));
                generators.push(inv_crt(5, 1, &pe, &rest, ZZi64));
            } else {
                let Zpe = Zn::new(pe as u64);
                let gen = get_multiplicative_generator(Zpe);
                generators.push(inv_crt(Zpe.smallest_lift(gen), 1, &pe, &rest, ZZi64));
            }
        }
        return Subgroup::new(GroupValue::from(self.clone()), order, generators.into_iter().map(|x| self.from_ring_el(self.underlying_ring().coerce(&ZZi64, x))).collect());
    }

    pub fn underlying_ring(&self) -> &Zn {
        self.group.underlying_ring()
    }

    pub fn as_ring_el<'a>(&self, value: &'a GaloisGroupEl) -> &'a ZnEl {
        self.group.as_ring_el(&value.value)
    }

    pub fn from_ring_el(&self, el: ZnEl) -> GaloisGroupEl {
        GaloisGroupEl { value: self.group.from_ring_el(el).unwrap() }
    }

    pub fn m(&self) -> u64 {
        *self.underlying_ring().modulus() as u64
    }

    pub fn element_order(&self, value: &GaloisGroupEl) -> usize {
        multiplicative_order(
            *self.as_ring_el(value),
            self.underlying_ring()
        ) as usize
    }

    pub fn from_representative(&self, x: i64) -> GaloisGroupEl {
        self.from_ring_el(self.underlying_ring().coerce(&ZZi64, x))
    }

    pub fn representative(&self, x: &GaloisGroupEl) -> u64 {
        self.underlying_ring().smallest_positive_lift(*self.as_ring_el(x)) as u64
    }
}

impl PartialEq for CyclotomicGaloisGroupBase {

    fn eq(&self, other: &Self) -> bool {
        self.group.get_group() == other.group.get_group()
    }
}

impl Eq for CyclotomicGaloisGroupBase {}

impl AbelianGroupBase for CyclotomicGaloisGroupBase {

    type Element = GaloisGroupEl;

    fn clone_el(&self, x: &Self::Element) -> Self::Element {
        x.clone()
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.group.eq_el(&lhs.value, &rhs.value)
    }

    fn hash<H: std::hash::Hasher>(&self, x: &Self::Element, hasher: &mut H) {
        self.group.hash(&x.value, hasher)
    }

    fn inv(&self, x: &Self::Element) -> Self::Element {
        GaloisGroupEl { value: self.group.inv(&x.value) }
    }

    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        GaloisGroupEl { value: self.group.op(lhs.value, rhs.value) }
    }

    fn identity(&self) -> Self::Element {
        GaloisGroupEl { value: self.group.identity() }
    }
}

impl Debug for CyclotomicGaloisGroupBase {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(Z/{}Z)*", self.underlying_ring().modulus())
    }
}

pub struct SerializableCyclotomicGaloisGroupEl<'a>(&'a CyclotomicGaloisGroupBase, GaloisGroupEl);

impl<'a> SerializableCyclotomicGaloisGroupEl<'a> {
    pub fn new(galois_group: &'a CyclotomicGaloisGroupBase, el: GaloisGroupEl) -> Self {
        Self(galois_group, el)
    }
}

impl<'a> Serialize for SerializableCyclotomicGaloisGroupEl<'a> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SerializableNewtypeStruct::new("CyclotomicGaloisGroupEl", &SerializeOwnedWithRing::new(*self.0.as_ring_el(&self.1), self.0.underlying_ring())).serialize(serializer)
    }
}

#[derive(Copy, Clone)]
pub struct DeserializeSeedCyclotomicGaloisGroupEl<'a>(&'a CyclotomicGaloisGroupBase);

impl<'a> DeserializeSeedCyclotomicGaloisGroupEl<'a> {
    pub fn new(galois_group: &'a CyclotomicGaloisGroupBase) -> Self {
        Self(galois_group)
    }
}

impl<'a, 'de> DeserializeSeed<'de> for DeserializeSeedCyclotomicGaloisGroupEl<'a> {
    type Value = GaloisGroupEl;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        Ok(DeserializeSeedNewtypeStruct::new("CyclotomicGaloisGroupEl", DeserializeWithRing::new(self.0.underlying_ring())).deserialize(deserializer).map(|g| self.0.from_ring_el(g)).unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct GaloisGroupEl {
    value: MultGroupEl<Zn>
}

pub fn get_multiplicative_generator(ring: Zn) -> ZnEl {
    let (p, e) = is_prime_power(ZZi64, ring.modulus()).unwrap();
    assert!(*ring.modulus() % 2 == 1, "for even m, Z/mZ* is either equal to Z/(m/2)Z* or not cyclic");
    let mut rng = oorandom::Rand64::new(ring.integer_ring().default_hash(ring.modulus()) as u128);
    let order = ZZi64.pow(p, e - 1) * (p - 1);
    let order_factorization = factor(&ZZi64, order);
    'test_generator: loop {
        let potential_generator = ring.random_element(|| rng.rand_u64());
        if !ring.is_unit(&potential_generator) {
            continue 'test_generator;
        }
        for (p, _) in &order_factorization {
            if ring.is_one(&ring.pow(potential_generator, (order / p) as usize)) {
                continue 'test_generator;
            }
        }
        return potential_generator;
    }
}
