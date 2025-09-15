use std::alloc::Allocator;
use std::alloc::Global;

use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::algorithms::convolution::KaratsubaAlgorithm;
use feanor_math::algorithms::discrete_log::Subgroup;
use feanor_math::algorithms::eea::*;
use feanor_math::algorithms::resultant::ComputeResultantRing;
use feanor_math::delegate::DelegateRing;
use feanor_math::delegate::DelegateRingImplFiniteRing;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::field::FieldStore;
use feanor_math::rings::extension::*;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::rings::zn::*;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::pid::PrincipalIdealRingStore;
use feanor_math::ring::*;
use feanor_math::rings::poly::*;
use feanor_math::rings::poly::dense_poly::*;
use feanor_math::rings::rational::RationalField;
use feanor_math::integer::*;
use tracing::instrument;

use crate::number_ring::galois::*;
use crate::number_ring::quotient_by_ideal::*;
use crate::number_ring::*;
use crate::NiceZn;
use crate::{log_time, ZZbig, ZZi64};

pub struct CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    ZZX: DensePolyRing<BigIntRing>,
    base: NumberRingQuotientByIdeal<NumberRing, ZnTy, A, C>,
    t: El<DensePolyRing<BigIntRing>>,
    /// The (algebraic) norm `N(t)` of `t`, which is equivalent to `Res(t(X), Phi_m(X))`
    normt: El<BigIntRing>,
    /// the value `N(t) t^-1`, which is an element of `Z[𝝵]`
    normt_t_inv: El<DensePolyRing<BigIntRing>>,
}

pub type CLPXPlaintextRing<NumberRing, ZnTy, A = Global, C = KaratsubaAlgorithm> = RingValue<CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>>;

impl<NumberRing, ZnTy, A, C> CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    #[instrument(skip_all)]
    pub fn create<const LOG: bool>(number_ring: NumberRing, base_ring: ZnTy, poly_ring: DensePolyRing<BigIntRing>, t: El<DensePolyRing<BigIntRing>>, acting_galois_group: Subgroup<CyclotomicGaloisGroup>, allocator: A, convolution: C) -> RingValue<Self> {
        let QQX = DensePolyRing::new(RationalField::new(ZZbig), "X");
        let QQ = QQX.base_ring();

        let gen_poly = number_ring.generating_poly(&poly_ring);

        // we compute `N(t) = Res(t(X), Phi_m(X))`; this is large, so use big integers
        let norm = log_time::<_, _, LOG, _>("Compute Resultant", |[]| 
            ZZbig.abs(<_ as ComputeResultantRing>::resultant(&poly_ring, poly_ring.clone_el(&gen_poly), poly_ring.clone_el(&t)))
        );
        let rest = ZZbig.checked_div(&norm, &ZZbig.pow(base_ring.characteristic(ZZbig).unwrap(), int_cast(acting_galois_group.subgroup_order(), ZZi64, ZZbig) as usize)).unwrap();
        assert!(ZZbig.is_one(&signed_gcd(rest, base_ring.characteristic(ZZbig).unwrap(), ZZbig)));

        // compute the inverse of `t(X)` modulo `Phi_m(X)`, which is required for encoding
        let ZZX_to_QQX = QQX.lifted_hom(&poly_ring, QQ.inclusion());
        let (mut s, _, d) = log_time::<_, _, LOG, _>("Compute Inverse", |[]| 
            QQX.extended_ideal_gen(&ZZX_to_QQX.map_ref(&t), &ZZX_to_QQX.map_ref(&gen_poly))
        );
        assert_eq!(0, QQX.degree(&d).unwrap());
        QQX.inclusion().mul_assign_map(&mut s, QQ.div(&QQ.inclusion().map_ref(&norm), QQX.coefficient_at(&d, 0)));
        let normt_t_inv = poly_ring.from_terms(QQX.terms(&s).map(|(c, i)| (
            ZZbig.checked_div(QQ.num(c), QQ.den(c)).unwrap(),
            i
        )));

        let base_ringX = DensePolyRing::new(base_ring, "X");
        let hom = base_ringX.base_ring().can_hom(&ZZbig).unwrap();
        let ideal_generator = base_ringX.lifted_hom(&poly_ring, &hom).map_ref(&t);
        let base = NumberRingQuotientByIdealBase::create(number_ring, base_ringX, ideal_generator, acting_galois_group, allocator, convolution);
        return RingValue::from(Self {
            base: base,
            normt: norm,
            normt_t_inv: normt_t_inv,
            t: t,
            ZZX: poly_ring
        });
    }

    pub fn t(&self) -> &El<DensePolyRing<BigIntRing>> {
        &self.t
    }

    pub fn ZZX(&self) -> &DensePolyRing<BigIntRing> {
        &self.ZZX
    }

    #[instrument(skip_all)]
    pub fn reduce_mod_t(&self, el: El<DensePolyRing<BigIntRing>>) -> <Self as RingBase>::Element {
        unimplemented!()
    }

    #[instrument(skip_all)]
    pub fn small_lift(&self, el: &<Self as RingBase>::Element) -> El<DensePolyRing<BigIntRing>> {
        unimplemented!()
    }

    #[instrument(skip_all)]
    pub fn encode<S>(&self, target: S, el: &<Self as RingBase>::Element) -> El<S>
        where S: RingStore, 
            S::Type: FreeAlgebra,
            <<S::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing
    {
        unimplemented!()
    }

    #[instrument(skip_all)]
    pub fn decode<S>(&self, from: S, el: &El<S>) -> <Self as RingBase>::Element
        where S: RingStore, 
            S::Type: FreeAlgebra,
            <<S::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing
    {
        unimplemented!()
    }
}

impl<NumberRing, ZnTy, A, C> PartialEq for CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

impl<NumberRing, ZnTy, A, C> DelegateRing for CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type Base = NumberRingQuotientByIdealBase<NumberRing, ZnTy, A, C>;
    type Element = El<NumberRingQuotientByIdeal<NumberRing, ZnTy, A, C>>;

    fn get_delegate(&self) -> &Self::Base {
        self.base.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<NumberRing, ZnTy, A, C> DelegateRingImplFiniteRing for CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{}

impl<NumberRing, ZnTy, A, C> NumberRingQuotient for CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type NumberRing = NumberRing;

    fn acting_galois_group(&self) -> &Subgroup<CyclotomicGaloisGroup> {
        self.base.acting_galois_group()
    }

    fn apply_galois_action(&self, x: &Self::Element, g: &GaloisGroupEl) -> Self::Element {
        self.base.apply_galois_action(x, g)
    }

    fn number_ring(&self) -> &Self::NumberRing {
        self.base.number_ring()
    }

    fn apply_galois_action_many(&self, x: &Self::Element, gs: &[GaloisGroupEl]) -> Vec<Self::Element> {
        self.base.apply_galois_action_many(x, gs)
    }
}

#[cfg(test)]
use feanor_math::algorithms::convolution::STANDARD_CONVOLUTION;
#[cfg(test)]
use feanor_math::group::AbelianGroupStore;
#[cfg(test)]
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::number_ring::general_cyclotomic::OddSquarefreeCyclotomicNumberRing;
#[cfg(test)]
use crate::number_ring::galois::CyclotomicGaloisGroupOps;
#[cfg(test)]
use std::array::from_fn;
#[cfg(test)]
use crate::ciphertext_ring::double_rns_ring::DoubleRNSRingBase;

#[cfg(test)]
fn test_rns_base() -> zn_rns::Zn<zn_64::Zn, BigIntRing> {
    zn_rns::Zn::create_from_primes(vec![4915200001, 4920115201, 4925030401, 4944691201, 5018419201, 5028249601, 5042995201, 5052825601, 5092147201, 5141299201], ZZbig)
}

#[cfg(test)]
fn test_ring1() -> (CLPXPlaintextRing<Pow2CyclotomicNumberRing, zn_64::Zn>, Vec<El<CLPXPlaintextRing<Pow2CyclotomicNumberRing, zn_64::Zn>>>) {
    let ZZX = DensePolyRing::new(ZZbig, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X - 2]);
    let number_ring = Pow2CyclotomicNumberRing::new(32);
    let acting_galois_group = number_ring.galois_group().get_group().clone().subgroup([]);
    let result = CLPXPlaintextRingBase::create::<true>(number_ring, zn_64::Zn::new(65537), ZZX, t, acting_galois_group, Global, STANDARD_CONVOLUTION);
    let elements = (0..31).map(|i| result.int_hom().map(1 << i)).collect();
    return (result, elements);
}

#[cfg(test)]
fn test_ring2() -> (CLPXPlaintextRing<Pow2CyclotomicNumberRing, zn_64::Zn>, Vec<El<CLPXPlaintextRing<Pow2CyclotomicNumberRing, zn_64::Zn>>>) {
    let ZZX = DensePolyRing::new(ZZbig, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + X - 2]);
    let number_ring = Pow2CyclotomicNumberRing::new(64);
    let acting_galois_group = number_ring.galois_group().get_group().clone().subgroup([]);
    let result = CLPXPlaintextRingBase::create::<true>(number_ring, zn_64::Zn::new(65537), ZZX, t, acting_galois_group, Global, STANDARD_CONVOLUTION);
    let elements = (0..31).map(|i| result.int_hom().map(1 << i)).collect();
    return (result, elements);
}

#[cfg(test)]
fn test_ring3() -> (CLPXPlaintextRing<Pow2CyclotomicNumberRing, zn_64::Zn>, Vec<El<CLPXPlaintextRing<Pow2CyclotomicNumberRing, zn_64::Zn>>>) {
    let ZZX = DensePolyRing::new(ZZbig, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 2]);
    let number_ring = Pow2CyclotomicNumberRing::new(64);
    let acting_galois_group = number_ring.galois_group().get_group().clone().subgroup([number_ring.galois_group().from_representative(33)]);
    let result = CLPXPlaintextRingBase::create::<true>(number_ring, zn_64::Zn::new(257), ZZX, t, acting_galois_group, Global, STANDARD_CONVOLUTION);
    let elements = (0..31).map(|i| result.int_hom().map(1 << i))
        .chain((0..31).map(|i| result.mul(result.canonical_gen(), result.int_hom().map(1 << i)))).collect();
    return (result, elements);
}

#[cfg(test)]
fn test_ring4() -> (CLPXPlaintextRing<OddSquarefreeCyclotomicNumberRing, zn_64::Zn>, Vec<El<CLPXPlaintextRing<OddSquarefreeCyclotomicNumberRing, zn_64::Zn>>>) {
    let ZZX = DensePolyRing::new(ZZbig, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(5) - 2]);
    let number_ring = OddSquarefreeCyclotomicNumberRing::new(75);
    let acting_galois_group = number_ring.galois_group().get_group().clone().subgroup([number_ring.galois_group().from_representative(16)]);
    let result = CLPXPlaintextRingBase::create::<true>(number_ring, zn_64::Zn::new(151), ZZX, t, acting_galois_group, Global, STANDARD_CONVOLUTION);
    let elements = (0..15).flat_map(|i| from_fn::<_, 4, _>(|j| result.mul(result.pow(result.canonical_gen(), j), result.int_hom().map(1 << i))).into_iter()).collect();
    return (result, elements);
}

#[test]
fn test_clpx_plaintext_ring_create() {
    let ZZX = DensePolyRing::new(ZZbig, "X");
    let (ring, _) = test_ring1();
    assert_eq!(1, ring.rank());
    assert_el_eq!(&ring, ring.int_hom().map(2), ring.canonical_gen());

    let (ring, _) = test_ring2();
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + X - 2]);
    assert_eq!(1, ring.rank());
    assert_el_eq!(&ring, ring.zero(), ZZX.evaluate(&t, &ring.canonical_gen(), ring.inclusion().compose(ring.base_ring().can_hom(&ZZbig).unwrap())));

    let (ring, _) = test_ring3();
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 2]);
    assert_eq!(4, ring.rank());
    assert_el_eq!(&ring, ring.zero(), ZZX.evaluate(&t, &ring.canonical_gen(), ring.inclusion().compose(ring.base_ring().can_hom(&ZZbig).unwrap())));

    let (ring, _) = test_ring4();
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(5) - 2]);
    assert_eq!(4, ring.rank());
    assert_el_eq!(&ring, ring.zero(), ZZX.evaluate(&t, &ring.canonical_gen(), ring.inclusion().compose(ring.base_ring().can_hom(&ZZbig).unwrap())));
}

#[test]
fn test_clpx_plaintext_ring_small_lift() {
    let ZZX = DensePolyRing::new(ZZbig, "X");

    let (ring, elements) = test_ring1();
    for a in &elements {
        assert_el_eq!(&ring, a, ring.get_ring().reduce_mod_t(ring.get_ring().small_lift(a)));
    }
    for a in &elements {
        assert!(ZZX.terms(&ring.get_ring().small_lift(a)).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 1));
    }

    let (ring, elements) = test_ring2();
    for a in &elements {
        assert_el_eq!(&ring, a, ring.get_ring().reduce_mod_t(ring.get_ring().small_lift(a)));
    }
    for a in &elements {
        assert!(ZZX.terms(&ring.get_ring().small_lift(a)).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 1));
    }

    let (ring, elements) = test_ring3();
    for a in &elements {
        assert_el_eq!(&ring, a, ring.get_ring().reduce_mod_t(ring.get_ring().small_lift(a)));
    }
    for a in &elements {
        assert!(ZZX.terms(&ring.get_ring().small_lift(a)).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 1));
    }

    let (ring, elements) = test_ring4();
    for a in &elements {
        assert_el_eq!(&ring, a, ring.get_ring().reduce_mod_t(ring.get_ring().small_lift(a)));
    }
    for a in &elements {
        assert!(ZZX.terms(&ring.get_ring().small_lift(a)).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 1));
    }
}

#[test]
fn test_clpx_plaintext_ring_encode_decode() {
    let ZQ = test_rns_base();

    let (ring, elements) = test_ring1();
    let RQ = DoubleRNSRingBase::new(ring.number_ring().clone(), ZQ.clone());
    for a in &elements {
        assert_el_eq!(&ring, a, ring.get_ring().decode(&RQ, &ring.get_ring().encode(&RQ, a)));
    }

    let (ring, elements) = test_ring2();
    let RQ = DoubleRNSRingBase::new(ring.number_ring().clone(), ZQ.clone());
    for a in &elements {
        assert_el_eq!(&ring, a, ring.get_ring().decode(&RQ, &ring.get_ring().encode(&RQ, a)));
    }

    let (ring, elements) = test_ring3();
    let RQ = DoubleRNSRingBase::new(ring.number_ring().clone(), ZQ.clone());
    for a in &elements {
        assert_el_eq!(&ring, a, ring.get_ring().decode(&RQ, &ring.get_ring().encode(&RQ, a)));
    }

    let (ring, elements) = test_ring4();
    let RQ = DoubleRNSRingBase::new(ring.number_ring().clone(), ZQ.clone());
    for a in &elements {
        assert_el_eq!(&ring, a, ring.get_ring().decode(&RQ, &ring.get_ring().encode(&RQ, a)));
    }
}
