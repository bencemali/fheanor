use std::alloc::{Allocator, Global};
use std::sync::Arc;

use feanor_math::algorithms::convolution::fft::{FFTConvolution, FFTConvolutionZn};
use feanor_math::algorithms::convolution::rns::{RNSConvolution, RNSConvolutionZn};
use feanor_math::algorithms::convolution::{ConvolutionAlgorithm, STANDARD_CONVOLUTION};
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity;
use feanor_math::algorithms::poly_gcd::hensel::hensel_lift_factorization;
use feanor_math::reduce_lift::poly_factor_gcd::PolyGCDLocallyIntermediateReductionMap;
use feanor_math::computation::{DontObserve, no_error};
use feanor_math::field::Field;
use feanor_math::rings::field::AsField;
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::pid::PrincipalIdealRingStore;
use feanor_math::divisibility::*;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::rings::extension::extension_impl::*;
use feanor_math::rings::extension::galois_field::GaloisField;
use feanor_math::rings::extension::*;
use feanor_math::rings::local::{AsLocalPIR, AsLocalPIRBase};
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::PolyRingStore;
use feanor_math::group::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::delegate::{WrapHom, UnwrapHom};
use feanor_math::ring::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::*;
use feanor_math::serialization::SerializableElementRing;
use tracing::instrument;

use crate::cache::{create_cached, SerializeDeserializeWith, StoreAs};
use crate::number_ring::galois::*;
use crate::number_ring::*;
use crate::*;
use crate::ntt::dyn_convolution::*;
use crate::number_ring::hypercube::interpolate::FastPolyInterpolation;
use crate::number_ring::hypercube::structure::*;

pub use crate::number_ring::hypercube::serialization::{DeserializeSeedHypercubeIsomorphismWithoutRing, SerializableHypercubeIsomorphismWithoutRing};

#[instrument(skip_all)]
fn hensel_lift_factor<R1, R2, A1, A2, C1, C2>(from_ring: &DensePolyRing<R1, A1, C1>, to_ring: &DensePolyRing<R2, A2, C2>, f: &El<DensePolyRing<R1, A1, C1>>, g: El<DensePolyRing<R2, A2, C2>>) -> El<DensePolyRing<R1, A1, C1>>
    where R1: RingStore,
        R1::Type: ZnRing,
        R2: RingStore,
        R2::Type: ZnRing + Field,
        A1: Allocator + Clone,
        A2: Allocator + Clone,
        C1: ConvolutionAlgorithm<R1::Type>,
        C2: ConvolutionAlgorithm<R2::Type>,
{
    let ZZ = StaticRing::<i64>::RING;
    let p = int_cast(to_ring.base_ring().integer_ring().clone_el(to_ring.base_ring().modulus()), ZZ, to_ring.base_ring().integer_ring());
    let (from_p, e) = is_prime_power(ZZ, &int_cast(from_ring.base_ring().integer_ring().clone_el(from_ring.base_ring().modulus()), ZZ, from_ring.base_ring().integer_ring())).unwrap();
    assert_eq!(p, from_p);

    let Zpe = zn_big::Zn::new(BigIntRing::RING, BigIntRing::RING.pow(int_cast(p, BigIntRing::RING, ZZ), e));
    let Zp = zn_big::Zn::new(BigIntRing::RING, int_cast(p, BigIntRing::RING, ZZ));
    let Fp = Zn::new(p as u64).as_field().ok().unwrap();
    let red_map = PolyGCDLocallyIntermediateReductionMap::new(ZZ.get_ring(), &p, &Zpe, e, &Zp, 1, 0);
    let ZpeX = DensePolyRing::new(&Zpe, "X");
    let FpX = DensePolyRing::new(&Fp, "X");
    let to_ZpeX = ZpeX.lifted_hom(from_ring, ZnReductionMap::new(from_ring.base_ring(), ZpeX.base_ring()).unwrap());
    let from_ZpeX = from_ring.lifted_hom(&ZpeX, ZnReductionMap::new(ZpeX.base_ring(), from_ring.base_ring()).unwrap());
    let to_FpX = FpX.lifted_hom(to_ring, ZnReductionMap::new(to_ring.base_ring(), FpX.base_ring()).unwrap());

    let f_mod_p = FpX.lifted_hom(&ZpeX, Fp.can_hom(&Zp).unwrap().compose(&red_map)).compose(&to_ZpeX).map_ref(f);
    let f_over_g = FpX.checked_div(&f_mod_p, &to_FpX.map_ref(&g)).unwrap();
    let lifted = hensel_lift_factorization(&red_map, &ZpeX, &FpX, &to_ZpeX.map_ref(f), &[to_FpX.map(g), f_over_g][..], DontObserve);
    return from_ZpeX.map(lifted.into_iter().next().unwrap());
}

fn create_convolution<R>(d: usize, log2_input_size: usize) -> DynConvolutionAlgorithmConvolution<R, Arc<dyn DynConvolutionAlgorithm<R>>>
    where R: ?Sized + ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>
{
    let fft_convolution = FFTConvolution::new();
    let max_log2_len = ZZi64.abs_log2_ceil(&(d as i64)).unwrap() + 1;
    if d <= 30 {
        DynConvolutionAlgorithmConvolution::new(Arc::new(STANDARD_CONVOLUTION))
    } else if fft_convolution.has_sufficient_precision(max_log2_len, log2_input_size) {
        DynConvolutionAlgorithmConvolution::new(Arc::new(FFTConvolutionZn::from(fft_convolution)))
    } else {
        DynConvolutionAlgorithmConvolution::new(Arc::new(RNSConvolutionZn::from(RNSConvolution::new(max_log2_len))))
    }
}

pub type SlotRingOver<R> = AsLocalPIR<FreeAlgebraImpl<R, Vec<El<R>>, Global, DynConvolutionAlgorithmConvolution<<R as RingStore>::Type, Arc<dyn DynConvolutionAlgorithm<<R as RingStore>::Type>>>>>;

///
/// Type of the slot ring used to represent the decomposition into
/// slots of the given ring `R`.
/// 
pub type SlotRingOf<R> = SlotRingOver<RingValue<BaseRing<R>>>;

///
/// Shortcut to access the base ring of a ring extension `R`.
/// 
pub type BaseRing<R> = <<<R as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type;

///
/// Shortcut to access the base ring of a ring extension `R` and
/// wrap it in a [`AsLocalPIRBase`].
/// 
pub type DecoratedBaseRingBase<R> = AsLocalPIRBase<RingValue<BaseRing<R>>>;

///
/// Represents the isomorphism `Fp[X]/(Phi_m(X)) -> F_(p^d)^domain(h)` for a
/// [`HypercubeStructure`] `h`, defined by
/// ```text
///   Fp[X]/(Phi_m(X))  ->  F_(p^d)^domain(h)
///          a(X)       ->  ( a(𝝵^(h(i)^-1)) )_(i in domain(h))
/// ```
/// where `𝝵` is an `m`-th primitive root of unity over `Fp`.
/// 
/// In fact, the more general case of `(Z/p^eZ)[X]/(Phi_m(X))` is supported.
///  
/// # Serialization
/// 
/// There are two ways of serializing/deserializing a [`HypercubeIsomorphism`]
///  - you can serialize the isomorphism including the implementation of `Fp[X]/(Phi_m(X))`
///    (if the latter is serializable) using the implementation of [`serde::Serialize`] and
///    [`serde::Deserialize`]
///  - alternatively, you can serialize the isomorphism without the ring implementation
///    using [`SerializableHypercubeIsomorphismWithoutRing`] and 
///    [`DeserializeSeedHypercubeIsomorphismWithoutRing`]; 
///    this of course requires that the ring is provided at deserialization time
/// 
pub struct HypercubeIsomorphism<R>
    where R: RingStore,
        R::Type: NumberRingQuotient,
        BaseRing<R>: NiceZn,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    ring: R,
    e: usize,
    slot_rings: Vec<SlotRingOf<R>>,
    slot_to_ring_interpolation: FastPolyInterpolation<DensePolyRing<RingValue<DecoratedBaseRingBase<R>>, Global>>,
    hypercube_structure: HypercubeStructure,
}

impl<R> HypercubeIsomorphism<R>
    where R: RingStore,
        R::Type: NumberRingQuotient,
        BaseRing<R>: NiceZn,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    ///
    /// Most general way to create a new [`HypercubeIsomorphism`].
    /// 
    /// The parameters are as follows:
    ///  - `ring` is the ring `Fp[X]/(Phi_m(X))` that is the domain of the isomorphism
    ///  - `hypercube_structure` is the [`HypercubeStructure`] that induces the isomorphism
    ///  - `slot_ring_moduli` is a list of all factors of `Phi_m(X) mod p`, ordered such
    ///    as to be compatible with the ordering of the slots, as given by [`HypercubeStructure`].
    /// 
    #[instrument(skip_all)]
    pub fn create<const LOG: bool>(ring: R, hypercube_structure: HypercubeStructure, ZpeX: DensePolyRing<AsLocalPIR<RingValue<BaseRing<R>>>>, slot_ring_moduli: Vec<El<DensePolyRing<AsLocalPIR<RingValue<BaseRing<R>>>>>>) -> Self {
        assert!(ring.acting_galois_group().get_group() == hypercube_structure.galois_group().get_group());
        let frobenius = hypercube_structure.frobenius(1);
        let d = hypercube_structure.d();
        
        // for quotients of cyclotomic number rings, the frobenius associated to a prime ideal
        // is always a nontrivial element of `<p> <= (Z/mZ)*`, where the characteristic of the
        // quotient is a power of `p`
        let (p, e) = is_prime_power(&ZZbig, &ring.characteristic(&ZZbig).unwrap()).unwrap();
        assert!(hypercube_structure.galois_group().eq_el(&frobenius, &hypercube_structure.galois_group().from_representative(int_cast(ZZbig.clone_el(&p), ZZi64, ZZbig))));

        let ring_ref = &ring;
        let slot_rings: Vec<SlotRingOf<R>> = log_time::<_, _, LOG, _>("[HypercubeIsomorphism::new_small_slot_ring] Computing slot rings", |[]| slot_ring_moduli.iter().map(|f| {
            let unwrap = UnwrapHom::new(ZpeX.base_ring().get_ring());
            let modulus = (0..d).map(|i| ring_ref.base_ring().negate(unwrap.map_ref(ZpeX.coefficient_at(f, i)))).collect::<Vec<_>>();
            let slot_ring = FreeAlgebraImpl::new_with_convolution(
                RingValue::from(ring_ref.base_ring().get_ring().clone()),
                d,
                modulus,
                "𝝵",
                Global,
                create_convolution(d, ring_ref.base_ring().integer_ring().abs_log2_ceil(ring_ref.base_ring().modulus()).unwrap())
            );
            let max_ideal_gen = slot_ring.inclusion().map(slot_ring.base_ring().coerce(&ZZbig, ZZbig.clone_el(&p)));
            return SlotRingOf::<R>::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(e)));
        }).collect::<Vec<_>>());

        let interpolation = log_time::<_, _, LOG, _>("[HypercubeIsomorphism::new_small_slot_ring] Computing interpolation data", |[]|
            FastPolyInterpolation::new(ZpeX, slot_ring_moduli)
        );

        return Self {
            hypercube_structure: hypercube_structure,
            ring: ring,
            e: e,
            slot_to_ring_interpolation: interpolation,
            slot_rings: slot_rings
        };
    }

    pub fn change_modulus<RNew>(&self, new_ring: RNew) -> HypercubeIsomorphism<RNew>
        where RNew: RingStore,
            RNew::Type: NumberRingQuotient,
            BaseRing<RNew>: NiceZn,
            DecoratedBaseRingBase<RNew>: CanIsoFromTo<BaseRing<RNew>>
    {
        let (p, e) = is_prime_power(&ZZbig, &new_ring.characteristic(&ZZbig).unwrap()).unwrap();
        let d = self.hypercube().d();
        let red_map = ZnReductionMap::new(self.ring().base_ring(), new_ring.base_ring()).unwrap();
        let poly_ring = DensePolyRing::new(new_ring.base_ring(), "X");
        let slot_rings = self.slot_rings.iter().map(|slot_ring| {
            let gen_poly = slot_ring.generating_poly(&poly_ring, &red_map);
            let new_slot_ring = FreeAlgebraImpl::new_with_convolution(
                RingValue::from(new_ring.base_ring().get_ring().clone()),
                d,
                (0..d).map(|i| new_ring.base_ring().negate(new_ring.base_ring().clone_el(poly_ring.coefficient_at(&gen_poly, i)))).collect::<Vec<_>>(),
                "𝝵",
                Global,
                create_convolution(d, ZZbig.abs_log2_ceil(&p).unwrap())
            );
            let max_ideal_gen = new_slot_ring.inclusion().map(new_slot_ring.base_ring().coerce(&ZZbig, ZZbig.clone_el(&p)));
            return AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(new_slot_ring, max_ideal_gen, Some(e)));
        }).collect::<Vec<_>>();

        let decorated_base_ring: RingValue<DecoratedBaseRingBase<RNew>> = AsLocalPIR::from_zn(RingValue::from(new_ring.base_ring().get_ring().clone())).unwrap();
        let base_poly_ring = DensePolyRing::new(decorated_base_ring, "X");
        return HypercubeIsomorphism {
            slot_to_ring_interpolation: self.slot_to_ring_interpolation.change_modulus(base_poly_ring),
            e: e,
            hypercube_structure: self.hypercube().clone(),
            ring: new_ring,
            slot_rings: slot_rings,
        };
    }

    pub fn hypercube(&self) -> &HypercubeStructure {
        &self.hypercube_structure
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn slot_ring_at<'a>(&'a self, i: usize) -> &'a SlotRingOf<R>
        where R: 'a
    {
        &self.slot_rings[i]
    }

    pub fn slot_ring<'a>(&'a self) -> &'a SlotRingOf<R>
        where R: 'a
    {
        self.slot_ring_at(0)
    }

    pub fn p(&self) -> &GaloisGroupEl {
        self.hypercube().p()
    }

    pub fn e(&self) -> usize {
        self.e
    }

    pub fn d(&self) -> usize {
        self.hypercube_structure.d()
    }

    pub fn galois_group(&self) -> &Subgroup<CyclotomicGaloisGroup> {
        self.hypercube_structure.galois_group()
    }

    pub fn slot_count(&self) -> usize {
        self.hypercube_structure.element_count()
    }
    
    #[instrument(skip_all)]
    pub fn get_slot_value(&self, el: &El<R>, slot_index: &GaloisGroupEl) -> El<SlotRingOf<R>> {
        let el = self.ring().apply_galois_action(el, &self.galois_group().inv(slot_index));
        let poly_ring = DensePolyRing::new(self.ring.base_ring(), "X");
        let el_as_poly = self.ring().poly_repr(&poly_ring, &el, self.ring.base_ring().identity());
        let poly_modulus = self.slot_ring().generating_poly(&poly_ring, self.ring.base_ring().identity());
        let (_, rem) = poly_ring.div_rem_monic(el_as_poly, &poly_modulus);
        self.slot_ring().from_canonical_basis((0..self.d()).map(|i| poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&rem, i))))
    }

    #[instrument(skip_all)]
    pub fn get_slot_values<'a>(&'a self, el: &'a El<R>) -> impl ExactSizeIterator<Item = El<SlotRingOf<R>>> + use<'a, R> {
        self.hypercube_structure.element_iter().map(move |g| self.get_slot_value(el, &g))
    }

    #[instrument(skip_all)]
    fn slot_ring_el_to_coset_X_repr<I>(&self, values: I) -> Vec<El<DensePolyRing<RingValue<DecoratedBaseRingBase<R>>>>>
        where I: IntoIterator<Item = El<SlotRingOf<R>>>
    {
        let poly_ring = self.slot_to_ring_interpolation.poly_ring();
        let wrap = WrapHom::new(poly_ring.base_ring().get_ring());
        let unwrap = UnwrapHom::new(poly_ring.base_ring().get_ring());
        let first_slot_ring: &SlotRingOf<R> = self.slot_ring();
        let mut values_it = values.into_iter();
        let result = values_it.by_ref().zip(self.hypercube_structure.element_iter()).enumerate().map(|(i, (a, g))| {
            let f = first_slot_ring.poly_repr(&poly_ring, &a, &wrap);
            let local_slot_ring = self.slot_ring_at(i);
            let image_zeta = local_slot_ring.pow(local_slot_ring.canonical_gen(), self.galois_group().representative(&g) as usize);
            return local_slot_ring.poly_repr(&poly_ring, &poly_ring.evaluate(&f, &image_zeta, local_slot_ring.inclusion().compose(&unwrap)), &wrap);
        }).collect::<Vec<_>>();
        assert!(values_it.next().is_none(), "iterator should only have {} elements", self.slot_count());
        return result;
    }

    #[instrument(skip_all)]
    pub fn from_slot_values<'a, I>(&self, values: I) -> El<R>
        where I: IntoIterator<Item = El<SlotRingOf<R>>>
    {
        let poly_ring = self.slot_to_ring_interpolation.poly_ring();
        let remainders = self.slot_ring_el_to_coset_X_repr(values);
        debug_assert!(remainders.iter().all(|r| poly_ring.degree(r).unwrap_or(0) < self.d()));

        let unreduced_result = self.slot_to_ring_interpolation.interpolate_unreduced(remainders);
        let unreduced_result = (0..=poly_ring.degree(&unreduced_result).unwrap_or(0)).map(|i| poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&unreduced_result, i))).collect::<Vec<_>>();

        let canonical_gen_pow_rank = self.ring().mul(self.ring().canonical_gen(), self.ring().from_canonical_basis((1..self.ring().rank()).map(|_| self.ring().base_ring().zero()).chain([self.ring().base_ring().one()].into_iter())));
        let unwrap = UnwrapHom::new(poly_ring.base_ring().get_ring());
        let mut current = self.ring().one();
        return <_ as RingStore>::sum(&self.ring, unreduced_result.chunks(self.ring.rank()).map(|chunk| self.ring.from_canonical_basis(
            chunk.iter().map(|a| poly_ring.base_ring().clone_el(a)).chain((0..(self.ring.rank() - chunk.len())).map(|_| poly_ring.base_ring().zero()))
                .map(|x| unwrap.map(x))
        )).map(|x| {
            let result = self.ring().mul_ref_snd(x, &current);
            self.ring().mul_assign_ref(&mut current, &canonical_gen_pow_rank);
            return result;
        }));
    }

    ///
    /// Tries to load a [`HypercubeIsomorphism`] from the corresponding file
    /// in the given directory. If the file does not exist, a new
    /// [`HypercubeIsomorphism`] is created and stored in the file.
    /// 
    pub fn new<const LOG: bool>(ring: &R, hypercube_structure: HypercubeStructure, cache_dir: Option<&str>) -> Self
        where R: Clone,
            BaseRing<R>: SerializableElementRing
    {
        let (p, e) = is_prime_power(&ZZbig, &ring.characteristic(&ZZbig).unwrap()).unwrap();
        let result = create_cached::<_, R, _, LOG>(
            &ring,
            || {
                // let d = hypercube_structure.d();  
                // if d * d < hypercube_structure.m() as usize {
                //     return Self::new_small_slot_ring::<LOG>(ring.clone(), hypercube_structure.clone());
                // } else {
                    return Self::new_large_slot_ring::<LOG>(ring.clone(), hypercube_structure.clone());
                // }
            },
            &filename_keys![hypercube, m: ring.number_ring().galois_group().m(), o: hypercube_structure.galois_group().subgroup_order(), p: p, e: e],
            cache_dir,
            if cache_dir.is_none() { StoreAs::None } else { StoreAs::AlwaysJson }
        );
        assert!(result.hypercube_structure == hypercube_structure, "hypercube structure mismatch");

        return result;
    }

    ///
    /// Computes an irrreducible factor of the generating polynomial of the given ring.
    /// 
    /// Currently, we assume that this root is an `m`-th root of unity. This makes sense,
    /// since the Galois group currently is a subgroup of `(Z/mZ)*`, thus the ring always
    /// has a generator which is a root of unity.
    /// 
    #[instrument(skip_all)]
    fn compute_factor_of_generating_poly_mod_p<const LOG: bool>(ring: &R, hypercube_structure: &HypercubeStructure) -> (DensePolyRing<AsField<RingValue<BaseRing<R>>>>, El<DensePolyRing<AsField<RingValue<BaseRing<R>>>>>) {
        let m = ring.acting_galois_group().m() as usize;
        assert!(ring.is_one(&ring.pow(ring.canonical_gen(), m)), "HypercubeIsomorphism currently assumes that the generator of the ring is an m-th root of unity");

        let d = hypercube_structure.d();
        let (p, _) = is_prime_power(&ZZbig, &ring.characteristic(&ZZbig).unwrap()).unwrap();
        let Fp = RingValue::from(<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as FromModulusCreateableZnRing>::from_modulus::<_, !>(|ZZ| 
            Ok(int_cast(ZZbig.clone_el(&p), RingRef::new(ZZ), ZZbig))
        ).unwrap_or_else(no_error)).as_field().ok().unwrap();
        let FpX = DensePolyRing::new(Fp, "X");
        let Fp = FpX.base_ring();
        let Fq = log_time::<_, _, LOG, _>("[HypercubeIsomorphism] Creating temporary slot ring", |[]|
            GaloisField::new_with_convolution(Fp, d, Global, create_convolution(d, ZZbig.abs_log2_ceil(&p).unwrap()))
        );
        let FqX = DensePolyRing::new(&Fq, "X");
        
        let root_of_unity = get_prim_root_of_unity(&Fq, m).unwrap();
        let gen_poly = ring.generating_poly(&FpX, ZnReductionMap::new(ring.base_ring(), FpX.base_ring()).unwrap());
        let root = (0..m).scan(Fq.one(), |state, _| {
            let result = Fq.clone_el(state);
            Fq.mul_assign_ref(state, &root_of_unity);
            Some(result)
        }).filter(|x| Fq.is_zero(&FpX.evaluate(&gen_poly, x, Fq.inclusion()))).next().unwrap();

        let mut result = FqX.prod((0..d).scan(
            root, 
            |current_root, _| {
                let result = FqX.sub(FqX.indeterminate(), FqX.inclusion().map_ref(current_root));
                *current_root = Fq.pow_gen(Fq.clone_el(current_root), &p, ZZbig);
                return Some(result);
            }
        ));
        let normalization_factor = FqX.base_ring().invert(FqX.lc(&result).unwrap()).unwrap();
        FqX.inclusion().mul_assign_map(&mut result, normalization_factor);

        let result = FpX.from_terms(FqX.terms(&result).map(|(c, i)| {
            let c_wrt_basis = Fq.wrt_canonical_basis(c);
            debug_assert!(c_wrt_basis.iter().skip(1).all(|c| Fp.is_zero(&c)));
            (c_wrt_basis.at(0), i)
        }));
        return (FpX, result);
    }

    ///
    /// Creates a new [`HypercubeIsomorphism`], using algorithms that are
    /// optimized for few large slots.
    /// 
    #[instrument(skip_all)]
    fn new_large_slot_ring<const LOG: bool>(ring: R, hypercube_structure: HypercubeStructure) -> Self {

        // in case that the slot ring is large, it can actually be faster to compute in the
        // original ring, since that can use the structure of the cyclotomic polynomial for
        // a more efficient implementation. Hence, we only compute a single factor of the 
        // cyclotomic polynomial using computations in the slot ring, and then compute the
        // other factors as the suitable Galois conjugates of the first factor, which can
        // be done using arithmetic in the full ring.

        let (FpX, factor) = Self::compute_factor_of_generating_poly_mod_p::<LOG>(&ring, &hypercube_structure);

        let decorated_base_ring: RingValue<DecoratedBaseRingBase<R>> = AsLocalPIR::from_zn(RingValue::from(ring.base_ring().get_ring().clone())).unwrap();
        let ZpeX = DensePolyRing::new(decorated_base_ring, "X");
        let ZZX = SparsePolyRing::new(ZZi64, "X");
        let gen_poly = ring.number_ring().generating_poly(&ZZX);
        let gen_poly_mod_pe = ZpeX.lifted_hom(&ZZX, ZpeX.base_ring().can_hom(ring.base_ring()).unwrap().compose(ring.base_ring().can_hom(&ZZX.base_ring()).unwrap())).map_ref(&gen_poly);
        let gen_poly_mod_p = FpX.lifted_hom(&ZZX, FpX.base_ring().can_hom(ZZX.base_ring()).unwrap()).map(gen_poly);
        
        let slot_ring_moduli = log_time::<_, _, LOG, _>("[HypercubeIsomorphism::new_large_slot_ring] Computing complete factorization of cyclotomic polynomial", |[]| {
            let mut result = Vec::new();
            let Zm = ring.number_ring().galois_group().underlying_ring();
            for g in hypercube_structure.element_iter() {
                let factor_mod_p = FpX.from_terms(
                    FpX.terms(&factor).map(|(c, i)| (
                        FpX.base_ring().clone_el(c),
                        Zm.smallest_positive_lift(Zm.mul(*ring.number_ring().galois_group().as_ring_el(&g), Zm.coerce(&ZZi64, i as i64))) as usize
                    ))
                );
                result.push(hensel_lift_factor(&ZpeX, &FpX, &gen_poly_mod_pe, FpX.normalize(FpX.ideal_gen(&factor_mod_p, &gen_poly_mod_p))));
            }
            return result;
        });

        return Self::create::<LOG>(ring, hypercube_structure, ZpeX, slot_ring_moduli);
    }
    
}

impl<R> SerializeDeserializeWith<R> for HypercubeIsomorphism<R>
    where R: RingStore + Clone,
        R::Type: NumberRingQuotient,
        BaseRing<R>: SerializableElementRing,
        BaseRing<R>: NiceZn,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    type SerializeWithData<'a> = SerializableHypercubeIsomorphismWithoutRing<'a, R> where Self: 'a, R: 'a;
    type DeserializeWithData<'a> = DeserializeSeedHypercubeIsomorphismWithoutRing<R> where Self: 'a, R: 'a;

    fn serialize_with<'a>(&'a self, ring: &'a R) -> Self::SerializeWithData<'a> {
        assert!(self.ring.get_ring() == ring.get_ring());
        SerializableHypercubeIsomorphismWithoutRing::new(self)
    }

    fn deserialize_with<'a>(ring: &'a R) -> Self::DeserializeWithData<'a> {
        DeserializeSeedHypercubeIsomorphismWithoutRing::new(ring.clone())
    }
}

#[cfg(test)]
use feanor_math::rings::finite::*;
#[cfg(test)]
use crate::number_ring::composite_cyclotomic::CompositeCyclotomicNumberRing;
#[cfg(test)]
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
#[cfg(test)]
use serde::de::DeserializeSeed;
#[cfg(test)]
use serde::Serialize;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::number_ring::quotient_by_int::{NumberRingQuotientByInt, NumberRingQuotientByIntBase};

#[cfg(test)]
fn test_ring1() -> (NumberRingQuotientByInt<Pow2CyclotomicNumberRing, Zn>, HypercubeStructure) {

    let galois_group = CyclotomicGaloisGroupBase::new(32);
    let p = galois_group.from_representative(7);
    let gs = vec![galois_group.from_representative(5)];
    let hypercube_structure = HypercubeStructure::new(galois_group.into().full_subgroup(), p, 4, vec![4], gs);
    let ring = NumberRingQuotientByIntBase::new(Pow2CyclotomicNumberRing::new(32), Zn::new(7));
    return (ring, hypercube_structure);
}

#[cfg(test)]
fn test_ring2() -> (NumberRingQuotientByInt<Pow2CyclotomicNumberRing, Zn>, HypercubeStructure) {
    let galois_group = CyclotomicGaloisGroupBase::new(32);
    let gs = vec![galois_group.from_representative(5), galois_group.from_representative(-1)];
    let p = galois_group.from_representative(17);
    let hypercube_structure = HypercubeStructure::new(galois_group.into().full_subgroup(), p, 2, vec![4, 2], gs);
    let ring = NumberRingQuotientByIntBase::new(Pow2CyclotomicNumberRing::new(32), Zn::new(17));
    return (ring, hypercube_structure);
}

#[cfg(test)]
fn test_ring3() -> (NumberRingQuotientByInt<CompositeCyclotomicNumberRing, Zn>, HypercubeStructure) {
    let galois_group = CyclotomicGaloisGroupBase::new(11 * 13);
    let p = galois_group.from_representative(3);
    let gs = vec![galois_group.from_representative(79), galois_group.from_representative(67)];
    let hypercube_structure = HypercubeStructure::new(
        galois_group.into().full_subgroup(),
        p,
        15,
        vec![2, 4],
        gs
    );
    let ring = NumberRingQuotientByIntBase::new(CompositeCyclotomicNumberRing::new(11, 13), Zn::new(3));
    return (ring, hypercube_structure);
}

#[test]
fn test_hypercube_isomorphism_from_to_slot_vector() {
    let mut rng = oorandom::Rand64::new(0);

    let (ring, hypercube) = test_ring1();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    assert_eq!(4, isomorphism.slot_count());
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let expected = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let element = isomorphism.from_slot_values(expected.iter().map(|a| slot_ring.clone_el(a)));
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring2();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    assert_eq!(8, isomorphism.slot_count());
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let expected = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let element = isomorphism.from_slot_values(expected.iter().map(|a| slot_ring.clone_el(a)));
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring3();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    assert_eq!(8, isomorphism.slot_count());
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let expected = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let element = isomorphism.from_slot_values(expected.iter().map(|a| slot_ring.clone_el(a)));
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }
}

#[test]
fn test_hypercube_isomorphism_is_isomorphic() {
    let mut rng = oorandom::Rand64::new(1);

    let (ring, hypercube) = test_ring1();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let lhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let rhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let expected = (0..isomorphism.slot_count()).map(|i| slot_ring.mul_ref(&lhs[i], &rhs[i])).collect::<Vec<_>>();
        let element = isomorphism.ring().mul(
            isomorphism.from_slot_values(lhs.iter().map(|a| slot_ring.clone_el(a))),
            isomorphism.from_slot_values(rhs.iter().map(|a| slot_ring.clone_el(a)))
        );
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring2();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let lhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let rhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let expected = (0..isomorphism.slot_count()).map(|i| slot_ring.mul_ref(&lhs[i], &rhs[i])).collect::<Vec<_>>();
        let element = isomorphism.ring().mul(
            isomorphism.from_slot_values(lhs.iter().map(|a| slot_ring.clone_el(a))),
            isomorphism.from_slot_values(rhs.iter().map(|a| slot_ring.clone_el(a)))
        );
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring3();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let lhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let rhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let expected = (0..isomorphism.slot_count()).map(|i| slot_ring.mul_ref(&lhs[i], &rhs[i])).collect::<Vec<_>>();
        let element = isomorphism.ring().mul(
            isomorphism.from_slot_values(lhs.iter().map(|a| slot_ring.clone_el(a))),
            isomorphism.from_slot_values(rhs.iter().map(|a| slot_ring.clone_el(a)))
        );
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }
}

#[test]
fn test_hypercube_isomorphism_rotation() {
    let mut rng = oorandom::Rand64::new(1);

    let (ring, hypercube) = test_ring1();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    let ring = isomorphism.ring();
    let hypercube = isomorphism.hypercube();
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let a = slot_ring.random_element(|| rng.rand_u64());

        let mut input = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        input[0] = slot_ring.clone_el(&a);

        let mut expected = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        expected[hypercube.dim_length(0) - 1] = slot_ring.clone_el(&a);

        let actual = ring.apply_galois_action(
            &isomorphism.from_slot_values(input.into_iter()),
            &hypercube.galois_group().pow(hypercube.dim_generator(0), &int_cast(hypercube.dim_length(0) as i64 - 1, ZZbig, ZZi64))
        );
        let actual = isomorphism.get_slot_values(&actual);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring2();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    let ring = isomorphism.ring();
    let hypercube = isomorphism.hypercube();
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let a = slot_ring.random_element(|| rng.rand_u64());
        
        let mut input = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        input[0] = slot_ring.clone_el(&a);

        let mut expected = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        expected[(hypercube.dim_length(0) - 1) * hypercube.dim_length(1)] = slot_ring.clone_el(&a);

        let actual = ring.apply_galois_action(
            &isomorphism.from_slot_values(input.into_iter()),
            &hypercube.galois_group().pow(hypercube.dim_generator(0), &int_cast(hypercube.dim_length(0) as i64 - 1, ZZbig, ZZi64))
        );
        let actual = isomorphism.get_slot_values(&actual).collect::<Vec<_>>();
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring3();
    let isomorphism = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
    let ring = isomorphism.ring();
    let hypercube = isomorphism.hypercube();
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let a = slot_ring.random_element(|| rng.rand_u64());
        
        let mut input = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        input[0] = slot_ring.clone_el(&a);

        let mut expected = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        expected[(hypercube.dim_length(0) - 1) * hypercube.dim_length(1)] = slot_ring.clone_el(&a);

        let actual = ring.apply_galois_action(
            &isomorphism.from_slot_values(input.into_iter()),
            &hypercube.galois_group().pow(hypercube.dim_generator(0), &int_cast(hypercube.dim_length(0) as i64 - 1, ZZbig, ZZi64))
        );
        let actual = isomorphism.get_slot_values(&actual);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }
}

// #[test]
// #[ignore]
// fn time_from_slot_values_large() {
//     use tracing_subscriber::prelude::*;
//     let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
//     tracing_subscriber::registry().with(chrome_layer).init();

//     let mut rng = oorandom::Rand64::new(1);

//     let allocator = feanor_mempool::AllocRc(Rc::new(feanor_mempool::dynsize::DynLayoutMempool::<Global>::new(Alignment::of::<u64>())));
//     let ring = RingValue::from(NumberRingQuotientImplBase::new(CompositeCyclotomicNumberRing::new(337, 127), Zn::new(65536)).into().with_allocator(allocator));
//     let galois_group = CyclotomicGaloisGroupBase::new(337 * 127);
//     let gs = vec![galois_group.from_representative(37085), galois_group.from_representative(25276)];
//     let p = galois_group.from_representative(2);
//     let hypercube = HypercubeStructure::new(galois_group, p, 21, vec![16, 126], gs);
//     let H = HypercubeIsomorphism::new::<true>(&ring, hypercube, None);
//     let slot_ring = H.slot_ring();

//     let value = log_time::<_, _, true, _>("from_slot_values", |[]| {
//         H.from_slot_values((0..H.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())))
//     });
//     std::hint::black_box(value);
// }

#[test]
fn test_serialization() {

    fn test_with_test_ring<R>((ring, hypercube_structure): (R, HypercubeStructure))
        where R: RingStore + Clone,
            R::Type: NumberRingQuotient,
            BaseRing<R>: NiceZn + SerializableElementRing + CanIsoFromTo<ZnBase>,
            DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
    {
        let hypercube = HypercubeIsomorphism::new::<false>(&ring, hypercube_structure, None);
        let serializer = serde_assert::Serializer::builder().is_human_readable(true).build();
        let tokens = SerializableHypercubeIsomorphismWithoutRing::new(&hypercube).serialize(&serializer).unwrap();
        let mut deserializer = serde_assert::Deserializer::builder(tokens).is_human_readable(true).build();
        let deserialized_hypercube = DeserializeSeedHypercubeIsomorphismWithoutRing::new(&ring).deserialize(&mut deserializer).unwrap();
        assert!(hypercube.slot_ring().get_ring() == deserialized_hypercube.slot_ring().get_ring());
        assert_el_eq!(hypercube.ring(), 
            hypercube.from_slot_values((0..hypercube.slot_count()).map(|i| hypercube.slot_ring().int_hom().map(i as i32))), 
            deserialized_hypercube.from_slot_values((0..deserialized_hypercube.slot_count()).map(|i| deserialized_hypercube.slot_ring().int_hom().map(i as i32)))
        );

        let serializer = serde_assert::Serializer::builder().is_human_readable(false).build();
        let tokens = SerializableHypercubeIsomorphismWithoutRing::new(&hypercube).serialize(&serializer).unwrap();
        let mut deserializer = serde_assert::Deserializer::builder(tokens).is_human_readable(false).build();
        let deserialized_hypercube = DeserializeSeedHypercubeIsomorphismWithoutRing::new(&ring).deserialize(&mut deserializer).unwrap();
        assert!(hypercube.slot_ring().get_ring() == deserialized_hypercube.slot_ring().get_ring());
        assert_el_eq!(hypercube.ring(), 
            hypercube.from_slot_values((0..hypercube.slot_count()).map(|i| hypercube.slot_ring().int_hom().map(i as i32))), 
            deserialized_hypercube.from_slot_values((0..deserialized_hypercube.slot_count()).map(|i| deserialized_hypercube.slot_ring().int_hom().map(i as i32)))
        );
    }
    test_with_test_ring(test_ring1());
    test_with_test_ring(test_ring2());
    test_with_test_ring(test_ring3());
}