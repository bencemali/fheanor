use std::fmt::Debug;

use feanor_math::algorithms::discrete_log::*;
use feanor_math::algorithms::eea::inv_crt;
use feanor_math::integer::int_cast;
use feanor_math::ring::*;
use feanor_math::delegate;
use feanor_math::rings::extension::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::algorithms::int_factor::factor;
use feanor_math::seq::VectorView;
use feanor_math::serialization::*;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use serde::de::DeserializeSeed;
use serde::Deserialize;
use serde::Serialize;

use crate::euler_phi;
use crate::number_ring::hypercube::structure::get_multiplicative_generator;
use crate::ZZbig;
use crate::ZZi64;

///
/// Represents the Galois group of the `m`-th cyclotomic number field by
/// its isomorphic subgroup of `(Z/mZ)*`.
/// 
#[derive(Clone)]
pub struct CyclotomicGaloisGroup {
    group: MultGroup<Zn>,
    order: usize,
    generating_set: GeneratingSet<MultGroup<Zn>>
}

impl PartialEq for CyclotomicGaloisGroup {

    fn eq(&self, other: &Self) -> bool {
        self.underlying_ring().get_ring() == other.group.0.get_ring()
    }
}

impl Eq for CyclotomicGaloisGroup {}

impl CyclotomicGaloisGroup {

    pub fn new(m: u64) -> Self {
        let ring = Zn::new(m);
        let group = MultGroup(ring);
        let m_factorization = factor(ZZi64, m as i64);
        let order = euler_phi(&m_factorization);
        let mut generators = Vec::new();
        for (p, e) in &m_factorization {
            let pe = ZZi64.pow(*p, *e);
            let rest = ZZi64.checked_div(&(m as i64), &pe).unwrap();
            if *p == 2 {
                generators.push(inv_crt(-1, 1, &pe, &rest, ZZi64));
                generators.push(inv_crt(5, 1, &pe, &rest, ZZi64));
            } else {
                let Zpe = Zn::new(pe as u64);
                let gen = get_multiplicative_generator(Zpe);
                generators.push(inv_crt(Zpe.smallest_lift(gen), 1, &pe, &rest, ZZi64));
            }
        }
        return Self {
            generating_set: GeneratingSet::for_zn(&group, generators.into_iter().map(|x| group.from_ring_el(ring.coerce(&ZZi64, x)).unwrap()).collect()),
            order: order as usize,
            group: group
        };
    }

    pub fn subgroup(&self, gens: &[CyclotomicGaloisGroupEl]) -> Self {
        let generating_set = GeneratingSet::for_zn(&self.group, gens.iter().map(|x| x.clone().value).collect());
        return Self {
            generating_set: generating_set,
            group: self.group.clone(),
            order: self.order
        }
    }

    pub fn identity(&self) -> CyclotomicGaloisGroupEl {
        CyclotomicGaloisGroupEl { value: self.group.identity() }
    }

    pub fn mul(&self, lhs: CyclotomicGaloisGroupEl, rhs: &CyclotomicGaloisGroupEl) -> CyclotomicGaloisGroupEl {
        CyclotomicGaloisGroupEl { value: self.group.op(lhs.value, &rhs.value) }
    }

    pub fn invert(&self, value: &CyclotomicGaloisGroupEl) -> CyclotomicGaloisGroupEl {
        CyclotomicGaloisGroupEl { value: self.group.inv(&value.value) }
    }

    pub fn representative(&self, value: &CyclotomicGaloisGroupEl) -> usize {
        self.underlying_ring().smallest_positive_lift(self.underlying_ring().clone_el(self.group.as_ring_el(&value.value))) as usize
    }

    pub fn from_representative(&self, value: i64) -> CyclotomicGaloisGroupEl {
        self.from_ring_el(self.underlying_ring().coerce(&ZZi64, value))
    }

    pub fn from_ring_el(&self, value: ZnEl) -> CyclotomicGaloisGroupEl {
        let group_el = self.group.from_ring_el(value).unwrap();
        assert!(self.generating_set.dlog(&self.group, &group_el).is_some());
        CyclotomicGaloisGroupEl { value: group_el }
    }

    pub fn negate(&self, value: &CyclotomicGaloisGroupEl) -> CyclotomicGaloisGroupEl {
        self.from_ring_el(self.underlying_ring().negate(self.underlying_ring().clone_el(self.group.as_ring_el(&value.value))))
    }

    pub fn prod<I>(&self, it: I) -> CyclotomicGaloisGroupEl
        where I: IntoIterator<Item = CyclotomicGaloisGroupEl>
    {
        it.into_iter().fold(self.identity(), |a, b| self.mul(a, &b))
    }

    pub fn pow(&self, base: &CyclotomicGaloisGroupEl, power: i64) -> CyclotomicGaloisGroupEl {
        CyclotomicGaloisGroupEl { value: self.group.pow(&base.value, power) }
    }

    pub fn is_identity(&self, value: &CyclotomicGaloisGroupEl) -> bool {
        self.group.is_identity(&value.value)
    }

    pub fn eq_el(&self, lhs: &CyclotomicGaloisGroupEl, rhs: &CyclotomicGaloisGroupEl) -> bool {
        self.group.eq_el(&lhs.value, &rhs.value)
    }

    ///
    /// Returns `m` such that this group is a subgroup of the Galois group of the
    /// `m`-th cyclotomic number field `Q[𝝵]`, where `𝝵` is an `m`-th primitive root of unity.
    /// 
    pub fn m(&self) -> usize {
        *self.underlying_ring().modulus() as usize
    }

    pub fn to_ring_el(&self, value: &CyclotomicGaloisGroupEl) -> ZnEl {
        self.underlying_ring().clone_el(self.group.as_ring_el(&value.value))
    }

    pub fn underlying_ring(&self) -> &Zn {
        &self.group.0
    }

    pub fn ambient_order(&self) -> usize {
        self.order
    }

    pub fn group_order(&self) -> usize {
        int_cast(self.generating_set.subgroup_order(), ZZi64, ZZbig) as usize
    }

    pub fn element_order(&self, value: &CyclotomicGaloisGroupEl) -> usize {
        multiplicative_order(
            self.to_ring_el(value),
            self.underlying_ring()
        ) as usize
    }
}

impl Debug for CyclotomicGaloisGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Subgroup of (Z/{}Z)* generated by {:?}", self.underlying_ring().modulus(), self.generating_set.as_iter().map(|g| self.underlying_ring().format(self.group.as_ring_el(g))).collect::<Vec<_>>())
    }
}

pub struct SerializableCyclotomicGaloisGroupEl<'a>(&'a CyclotomicGaloisGroup, CyclotomicGaloisGroupEl);

impl<'a> SerializableCyclotomicGaloisGroupEl<'a> {
    pub fn new(galois_group: &'a CyclotomicGaloisGroup, el: CyclotomicGaloisGroupEl) -> Self {
        Self(galois_group, el)
    }
}

impl<'a> Serialize for SerializableCyclotomicGaloisGroupEl<'a> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SerializableNewtypeStruct::new("CyclotomicGaloisGroupEl", &SerializeOwnedWithRing::new(self.0.to_ring_el(&self.1), self.0.underlying_ring())).serialize(serializer)
    }
}

#[derive(Copy, Clone)]
pub struct DeserializeSeedCyclotomicGaloisGroupEl<'a>(&'a CyclotomicGaloisGroup);

impl<'a> DeserializeSeedCyclotomicGaloisGroupEl<'a> {
    pub fn new(galois_group: &'a CyclotomicGaloisGroup) -> Self {
        Self(galois_group)
    }
}

impl<'a, 'de> DeserializeSeed<'de> for DeserializeSeedCyclotomicGaloisGroupEl<'a> {
    type Value = CyclotomicGaloisGroupEl;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        Ok(DeserializeSeedNewtypeStruct::new("CyclotomicGaloisGroupEl", DeserializeWithRing::new(self.0.underlying_ring())).deserialize(deserializer).map(|g| self.0.from_ring_el(g)).unwrap())
    }
}

impl Serialize for CyclotomicGaloisGroup {
    
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        #[derive(Serialize)]
        #[serde(rename = "CyclotomicGaloisGroup")]
        struct SerializableCyclotomicGaloisGroup {
            m: u64,
            generators: Vec<u64>
        }
        SerializableCyclotomicGaloisGroup {
            m: self.m() as u64,
            generators: self.generating_set.as_iter().map(|g| self.underlying_ring().smallest_positive_lift(*self.group.as_ring_el(g)) as u64).collect()
        }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CyclotomicGaloisGroup {

    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        #[derive(Deserialize)]
        #[serde(rename = "CyclotomicGaloisGroup")]
        struct DeserializableCyclotomicGaloisGroup {
            m: u64,
            generators: Vec<u64>
        }
        DeserializableCyclotomicGaloisGroup::deserialize(deserializer).map(|res| {
            let parent = CyclotomicGaloisGroup::new(res.m);
            parent.subgroup(&res.generators.iter().map(|g| parent.from_representative(*g as i64)).collect::<Vec<_>>())
        })
    }
}

#[derive(Debug, Clone)]
pub struct CyclotomicGaloisGroupEl {
    value: MultGroupEl<Zn>
}

///
/// Trait for rings generated by an `m`-th root of unity. It should be returned by
/// [`feanor_math::rings::extension::FreeAlgebra::canonical_gen()`].
/// 
/// Such rings are always isomorphic to `R/I`, where `R` is the ring of integers in the
/// `m`-th cyclotomic number field, and `I` is any ideal of `R`. We define the Galois
/// group of this ring to be the group of Galois automorphisms of `R` that fix `I`, i.e. 
/// `{ σ in Gal(R/Z) | σ(I) = I }`.
/// 
/// # Multiple possible `m`
/// 
/// There is some ambiguity, as the ring in question might have generators of different multiplicative
/// orders, thus being a valid [`CyclotomicQuotient`] for multiple `m` (of course, the Galois groups are
/// always isomorphic). It is up to implementations which `m` they expose. Naturally, this should be 
/// consistent - i.e. `self.canonical_gen()` should always return a `self.m()`-th root of unity.
/// 
/// # Nontrivial automorphisms
/// 
/// See [`feanor_math::rings::extension::FreeAlgebra`].
/// 
/// Note that computing this particular map when given arbitrary isomorphic instances `R`, `S`
/// can be difficult for specific implementations, hence it is not required that for all isomorphic
/// instances `R, S: RingType` with `RingType: CyclotomicRing` we have `R.has_canonical_hom(S).is_some()`.
/// Naturally, it is always required that `R.has_canonical_iso(R).is_some()`.
/// 
pub trait CyclotomicQuotient: FreeAlgebra {

    ///
    /// The cyclotomic order, i.e. the multiplicative order of `self.canonical_gen()`.
    ///
    fn m(&self) -> usize {
        self.galois_group().m()
    }

    ///
    /// Returns the Galois group of this ring. This is not a standard notion, we define it
    /// as the group of Galois automorphisms of `R` that fix `I`, i.e. `{ σ in Gal(R/Z) | σ(I) = I }`.
    /// 
    /// If this ring is a finite field, then this agrees with the standard Galois group of the
    /// field over the prime field. Indeed, if `I` is a prime ideal over a prime number `p`, then
    /// the Galois group is the decomposition group, which is isomorphic to `Gal(Fq/Fp)` via
    /// ```text
    ///   Gal(R/I) -> Gal(Fq/Fp),  σ -> (x mod I -> σx mod I)
    /// ```
    /// (assuming that `p` is unramified in `R`).
    /// 
    fn galois_group(&self) -> &CyclotomicGaloisGroup;

    ///
    /// Computes `g(x)` for the given Galois automorphism `g`.
    /// 
    fn apply_galois_action(&self, x: &Self::Element, g: &CyclotomicGaloisGroupEl) -> Self::Element;

    ///
    /// Computes `g(x)` for each Galois automorphism `g` in the given list.
    /// 
    /// This may be faster than using [`CyclotomicQuotient::apply_galois_action()`] multiple times.
    /// 
    fn apply_galois_action_many(&self, x: &Self::Element, gs: &[CyclotomicGaloisGroupEl]) -> Vec<Self::Element> {
        gs.iter().map(move |g| self.apply_galois_action(&x, g)).collect()
    }
}

///
/// The [`RingStore`] belonging to [`CyclotomicQuotient`]
/// 
pub trait CyclotomicQuotientStore: RingStore
    where Self::Type: CyclotomicQuotient
{
    delegate!{ CyclotomicQuotient, fn m(&self) -> usize }
    delegate!{ CyclotomicQuotient, fn galois_group(&self) -> &CyclotomicGaloisGroup }
    delegate!{ CyclotomicQuotient, fn apply_galois_action(&self, el: &El<Self>, s: &CyclotomicGaloisGroupEl) -> El<Self> }
    delegate!{ CyclotomicQuotient, fn apply_galois_action_many(&self, el: &El<Self>, gs: &[CyclotomicGaloisGroupEl]) -> Vec<El<Self>> }
}

impl<R: RingStore> CyclotomicQuotientStore for R
    where R::Type: CyclotomicQuotient
{}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_cyclotomic_ring_axioms<R: CyclotomicQuotientStore>(ring: R)
    where R::Type: CyclotomicQuotient
{
    use feanor_math::assert_el_eq;
    use feanor_math::primitive_int::*;
    use feanor_math::algorithms::int_factor::factor;
    use feanor_math::algorithms::cyclotomic::cyclotomic_polynomial;
    use feanor_math::rings::poly::*;
    use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
    use feanor_math::seq::*;
    use feanor_math::homomorphism::Homomorphism;

    let zeta = ring.canonical_gen();
    let m = ring.m();
    
    assert_el_eq!(&ring, &ring.one(), &ring.pow(ring.clone_el(&zeta), m as usize));
    for (p, _) in factor(&StaticRing::<i64>::RING, m as i64) {
        assert!(!ring.eq_el(&ring.one(), &ring.pow(ring.clone_el(&zeta), m as usize / p as usize)));
    }

    // test minimal polynomial of zeta
    let poly_ring = SparsePolyRing::new(&StaticRing::<i64>::RING, "X");
    let cyclo_poly = cyclotomic_polynomial(&poly_ring, m as usize);

    let x = ring.pow(ring.clone_el(&zeta), ring.rank());
    let x_vec = ring.wrt_canonical_basis(&x);
    assert_eq!(ring.rank(), x_vec.len());
    for i in 0..x_vec.len() {
        assert_el_eq!(ring.base_ring(), &ring.base_ring().negate(ring.base_ring().int_hom().map(*poly_ring.coefficient_at(&cyclo_poly, i) as i32)), &x_vec.at(i));
    }
    assert_el_eq!(&ring, &x, &ring.from_canonical_basis((0..x_vec.len()).map(|i| x_vec.at(i))));

    // test basis wrt_root_of_unity_basis linearity and compatibility from_canonical_basis/wrt_root_of_unity_basis
    for i in (0..ring.rank()).step_by(5) {
        for j in (1..ring.rank()).step_by(7) {
            if i == j {
                continue;
            }
            let element = ring.from_canonical_basis((0..ring.rank()).map(|k| if k == i { ring.base_ring().one() } else if k == j { ring.base_ring().int_hom().map(2) } else { ring.base_ring().zero() }));
            let expected = ring.add(ring.pow(ring.clone_el(&zeta), i), ring.int_hom().mul_map(ring.pow(ring.clone_el(&zeta), j), 2));
            assert_el_eq!(&ring, &expected, &element);
            let element_vec = ring.wrt_canonical_basis(&expected);
            for k in 0..ring.rank() {
                if k == i {
                    assert_el_eq!(ring.base_ring(), &ring.base_ring().one(), &element_vec.at(k));
                } else if k == j {
                    assert_el_eq!(ring.base_ring(), &ring.base_ring().int_hom().map(2), &element_vec.at(k));
                } else {
                    assert_el_eq!(ring.base_ring(), &ring.base_ring().zero(), &element_vec.at(k));
                }
            }
        }
    }
}
