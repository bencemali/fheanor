use std::alloc::Allocator;

use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::algorithms::discrete_log::Subgroup;
use feanor_math::algorithms::eea::*;
use feanor_math::algorithms::int_factor::factor;
use feanor_math::algorithms::resultant::ComputeResultantRing;
use feanor_math::delegate::DelegateRing;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::field::FieldStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::pid::PrincipalIdealRingStore;
use feanor_math::ring::*;
use feanor_math::algorithms::cyclotomic::cyclotomic_polynomial;
use feanor_math::rings::extension::extension_impl::FreeAlgebraImpl;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::field::*;
use feanor_math::rings::poly::*;
use feanor_math::rings::poly::dense_poly::*;
use feanor_math::rings::poly::sparse_poly::*;
use feanor_math::primitive_int::*;
use feanor_math::rings::rational::RationalField;
use feanor_math::rings::zn::*;
use feanor_math::integer::*;
use feanor_math::seq::sparse::SparseMapVector;
use feanor_math::seq::VectorFn;
use feanor_math::seq::VectorView;
use feanor_math::seq::VectorViewMut;

use crate::ciphertext_ring::BGFVCiphertextRing;
use crate::number_ring::galois::CyclotomicGaloisGroup;
use crate::number_ring::quotient_by_ideal::*;
use crate::number_ring::AbstractNumberRing;
use crate::NiceZn;
use crate::{euler_phi, log_time, ZZbig, ZZi64};

pub struct CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    base: NumberRingQuotientByIdeal<NumberRing, ZnTy, A, C>,
    t: El<NumberRingQuotientByIdeal<NumberRing, ZnTy, A, C>>,
    /// The (algebraic) norm `N(t)` of `t`, which is equivalent to `Res(t(X), Phi_m(X))`
    normt: El<BigIntRing>,
    /// the value `N(t) t^-1`, which is an element of `Z[𝝵]`
    normt_t_inv: Vec<El<BigIntRing>>,
}

impl<NumberRing, ZnTy, A, C> CLPXPlaintextRingBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    pub fn new<const LOG: bool>(number_ring: NumberRing, base_ring: ZnTy, poly_ring: DensePolyRing<BigIntRing>, t: El<DensePolyRing<BigIntRing>>, prime: El<BigIntRing>, acting_galois_group: Subgroup<CyclotomicGaloisGroup>) -> Self {
        let QQX = DensePolyRing::new(RationalField::new(ZZbig), "X");
        let QQ = QQX.base_ring();

        let gen_poly = number_ring.generating_poly(&poly_ring);

        // we compute `N(t) = Res(t(X), Phi_m(X))`; this is large, so use big integers
        let norm = log_time::<_, _, LOG, _>("Compute Resultant", |[]| 
            ZZbig.abs(<_ as ComputeResultantRing>::resultant(&poly_ring, poly_ring.clone_el(&gen_poly), poly_ring.clone_el(&t)))
        );
        let rest = ZZbig.checked_div(&norm, &ZZbig.pow(ZZbig.clone_el(&prime), acting_galois_group.group_order())).unwrap();
        assert!(!ZZbig.divides(&rest, &prime));

        // compute the inverse of `t(X)` modulo `Phi_m(X)`, which is required for encoding
        let ZZX_to_QQX = QQX.lifted_hom(&poly_ring, QQ.inclusion());
        let (mut s, _, d) = log_time::<_, _, LOG, _>("Compute Inverse", |[]| 
            eea(ZZX_to_QQX.map_ref(&t), ZZX_to_QQX.map_ref(&gen_poly), &QQX)
        );
        assert_eq!(0, QQX.degree(&d).unwrap());
        QQX.inclusion().mul_assign_map(&mut s, QQ.div(&QQ.inclusion().map_ref(&norm), QQX.coefficient_at(&d, 0)));
        let normt_t_inv = poly_ring.from_terms(QQX.terms(&s).map(|(c, i)| (
            ZZbig.checked_div(QQ.num(c), QQ.den(c)).unwrap(),
            i
        )));

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

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::number_ring::composite_cyclotomic::CompositeCyclotomicNumberRing;
#[cfg(test)]
use crate::ciphertext_ring::double_rns_managed::*;

#[cfg(test)]
fn test_rns_base() -> zn_rns::Zn<zn_64::Zn, BigIntRing> {
    zn_rns::Zn::create_from_primes(vec![167116801, 200540161, 284098561, 317521921, 384368641, 451215361, 501350401, 651755521, 752025601, 802160641], ZZbig)
}

#[test]
fn test_clpx_base_encoding_new() {
    let ZZX = DensePolyRing::new(ZZi64, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X - 2]);
    let m = 32;
    let encoding = CLPXBaseEncoding::new::<true>(m, ZZX.clone(), t, ZZbig.int_hom().map(65537));
    let Fp = encoding.Fp();
    assert_el_eq!(Fp, Fp.int_hom().map(2), &encoding.zeta_im);

    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + X - 2]);
    let m = 64;
    let encoding = CLPXBaseEncoding::new::<true>(m, ZZX.clone(), ZZX.clone_el(&t), ZZbig.int_hom().map(6700417));
    let Fp = encoding.Fp();
    assert_el_eq!(Fp, Fp.zero(), ZZX.evaluate(&t, &encoding.zeta_im, Fp.can_hom(&ZZi64).unwrap()));
}

#[test]
fn test_clpx_base_encoding_map() {
    let ZZX = DensePolyRing::new(ZZi64, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X - 2]);
    let m = 32;
    let encoding = CLPXBaseEncoding::new::<true>(m, ZZX.clone(), t, ZZbig.int_hom().map(65537));
    let Fp = encoding.Fp();
    let ZZX = encoding.ZZX();
    let elements = (0..16).map(|i| Fp.int_hom().map(1 << i)).collect::<Vec<_>>();
    for a in &elements {
        for b in &elements {
            assert_el_eq!(Fp, Fp.mul_ref(a, b), encoding.map(&ZZX.mul(encoding.small_preimage(Fp.clone_el(a)), encoding.small_preimage(Fp.clone_el(b)))));
        }
    }
    for a in &elements {
        assert!(ZZX.terms(&encoding.small_preimage(Fp.clone_el(a))).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 1));
    }

    let ZZX = DensePolyRing::new(ZZi64, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + X - 2]);
    let m = 64;
    let encoding = CLPXBaseEncoding::new::<true>(m, ZZX.clone(), ZZX.clone_el(&t), ZZbig.int_hom().map(6700417));
    let Fp = encoding.Fp();
    let ZZX = encoding.ZZX();
    let elements = (0..30).map(|i| Fp.int_hom().map(1 << i)).collect::<Vec<_>>();
    for a in &elements {
        for b in &elements {
            assert_el_eq!(Fp, Fp.mul_ref(a, b), encoding.map(&ZZX.mul(encoding.small_preimage(Fp.clone_el(a)), encoding.small_preimage(Fp.clone_el(b)))));
        }
    }
    for a in &elements {
        assert!(ZZX.terms(&encoding.small_preimage(Fp.clone_el(a))).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 1));
    }
}

#[test]
fn test_clpx_encoding_map() {
    let ZZX = DensePolyRing::new(ZZi64, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X - 2]);
    let m1 = 17;
    let m2 = 15;
    let base_encoding = CLPXBaseEncoding::new::<false>(m1, ZZX.clone(), t, ZZbig.int_hom().map(131071));
    let encoding = CLPXEncoding::new::<false>(m2, base_encoding);
    let P = encoding.plaintext_ring();
    let ZZX = encoding.ZZX();
    let rank = encoding.plaintext_ring().rank();
    let elements = [
        P.zero(),
        P.one(),
        P.int_hom().map(363),
        P.canonical_gen(),
        P.int_hom().mul_map(P.canonical_gen(), 363),
        P.add(P.canonical_gen(), P.one()),
        P.pow(P.canonical_gen(), rank - 1),
        P.int_hom().mul_map(P.pow(P.canonical_gen(), rank - 1), 363),
    ];
    for a in &elements {
        for b in &elements {
            assert_el_eq!(P, P.mul_ref(a, b), encoding.map(&ZZX.mul(encoding.small_preimage(a), encoding.small_preimage(b))));
        }
    }
    for a in &elements {
        assert!(ZZX.terms(&encoding.small_preimage(a)).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 3));
    }

    let ZZX = DensePolyRing::new(ZZi64, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + X - 2]);
    let m1 = 17;
    let m2 = 15;
    let base_encoding = CLPXBaseEncoding::new::<false>(m1, ZZX.clone(), t, ZZbig.int_hom().map(43691));
    let encoding = CLPXEncoding::new::<false>(m2, base_encoding);
    let P = encoding.plaintext_ring();
    let ZZX = encoding.ZZX();
    let rank = encoding.plaintext_ring().rank();
    let elements = [
        P.zero(),
        P.one(),
        P.int_hom().map(363),
        P.canonical_gen(),
        P.int_hom().mul_map(P.canonical_gen(), 363),
        P.add(P.canonical_gen(), P.one()),
        P.pow(P.canonical_gen(), rank - 1),
        P.int_hom().mul_map(P.pow(P.canonical_gen(), rank - 1), 363),
    ];
    for a in &elements {
        for b in &elements {
            assert_el_eq!(P, P.mul_ref(a, b), encoding.map(&ZZX.mul(encoding.small_preimage(a), encoding.small_preimage(b))));
        }
    }
    for a in &elements {
        assert!(ZZX.terms(&encoding.small_preimage(a)).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 3));
    }
}

#[test]
fn test_clpx_encoding_not_coprime_map() {
    let ZZX = DensePolyRing::new(ZZi64, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X - 2]);
    let m1 = 10;
    let m2 = 15;
    let base_encoding = CLPXBaseEncoding::new::<false>(m1, ZZX.clone(), t, ZZbig.int_hom().map(11));
    let encoding = CLPXEncoding::new::<false>(m2, base_encoding);
    let P = encoding.plaintext_ring();
    let ZZX = encoding.ZZX();
    let rank = encoding.plaintext_ring().rank();
    assert_eq!(10, rank);
    let elements = [
        P.zero(),
        P.one(),
        P.int_hom().map(5),
        P.canonical_gen(),
        P.int_hom().mul_map(P.canonical_gen(), 5),
        P.add(P.canonical_gen(), P.one()),
        P.pow(P.canonical_gen(), rank - 1),
        P.int_hom().mul_map(P.pow(P.canonical_gen(), rank - 1), 5),
    ];
    assert_el_eq!(P, P.one(), P.pow(encoding.map(&ZZX.indeterminate()), 150));
    assert_el_eq!(P, P.inclusion().map_ref(&encoding.base_encoding().zeta_im), P.pow(encoding.map(&ZZX.indeterminate()), 15));
    assert!(!P.is_one(&P.pow(encoding.map(&ZZX.indeterminate()), 50)));
    assert!(!P.is_one(&P.pow(encoding.map(&ZZX.indeterminate()), 75)));
    assert!(!P.is_one(&P.pow(encoding.map(&ZZX.indeterminate()), 30)));
    for a in &elements {
        for b in &elements {
            assert_el_eq!(P, P.mul_ref(a, b), encoding.map(&ZZX.mul(encoding.small_preimage(a), encoding.small_preimage(b))));
        }
    }
    for a in &elements {
        assert!(ZZX.terms(&encoding.small_preimage(a)).all(|(c, _)| int_cast(ZZbig.clone_el(c), ZZi64, ZZbig).abs() <= 3));
    }
}

#[test]
fn test_clpx_base_encoding_encode_decode() {
    let ZZX = DensePolyRing::new(ZZi64, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X - 2]);
    let m = 32;
    let encoding = CLPXBaseEncoding::new::<true>(m, ZZX.clone(), t, ZZbig.int_hom().map(65537));
    let Fp = encoding.Fp();
    let elements = (0..16).map(|i| Fp.int_hom().map(1 << i)).collect::<Vec<_>>();
    let ZQ = test_rns_base();
    for a in &elements {
        assert_el_eq!(&Fp, a, encoding.decode_impl(&ZQ, encoding.encode_impl(&ZQ, Fp.clone_el(a))));
    }

    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + X - 2]);
    let m = 17;
    let encoding = CLPXBaseEncoding::new::<true>(m, ZZX.clone(), ZZX.clone_el(&t), ZZbig.int_hom().map(43691));
    let Fp = encoding.Fp();
    let elements = (0..20).map(|i| Fp.int_hom().map(1 << i)).collect::<Vec<_>>();
    let ZQ = test_rns_base();
    for a in &elements {
        assert_el_eq!(&Fp, a, encoding.decode_impl(&ZQ, encoding.encode_impl(&ZQ, Fp.clone_el(a))));
    }
}

#[test]
fn test_clpx_encoding_encode_decode() {
    let ZZX = DensePolyRing::new(ZZi64, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + X - 2]);
    let m1 = 17;
    let m2 = 15;
    let base_encoding = CLPXBaseEncoding::new::<false>(m1, ZZX.clone(), t, ZZbig.int_hom().map(43691));
    let encoding = CLPXEncoding::new::<false>(m2, base_encoding);
    let P = encoding.plaintext_ring();
    let rank = encoding.plaintext_ring().rank();
    let elements = [
        P.zero(),
        P.one(),
        P.int_hom().map(363),
        P.canonical_gen(),
        P.int_hom().mul_map(P.canonical_gen(), 363),
        P.add(P.canonical_gen(), P.one()),
        P.pow(P.canonical_gen(), rank - 1),
        P.int_hom().mul_map(P.pow(P.canonical_gen(), rank - 1), 363),
    ];
    let C = ManagedDoubleRNSRingBase::new(CompositeCyclotomicNumberRing::new(17, 15), test_rns_base());
    for a in &elements {
        assert_el_eq!(P, a, encoding.decode(&C, &encoding.encode(&C, a)));
    }
}