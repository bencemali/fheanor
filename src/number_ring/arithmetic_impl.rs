use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::cmp::max;

use feanor_math::algorithms::convolution::{ConvolutionAlgorithm, KaratsubaAlgorithm};
use feanor_math::algorithms::discrete_log::Subgroup;
use feanor_math::divisibility::DivisibilityRing;
use feanor_math::field::Field;
use feanor_math::homomorphism::{CanHomFrom, CanIsoFromTo, Homomorphism};
use feanor_math::algorithms::linsolve::LinSolveRing;
use feanor_math::integer::{int_cast, BigIntRing, BigIntRingBase, IntegerRing, IntegerRingStore};
use feanor_math::iters::{multi_cartesian_product, MultiProduct};
use feanor_math::matrix::OwnedMatrix;
use feanor_math::pid::PrincipalIdealRing;
use feanor_math::rings::extension::{create_multiplication_matrix, FreeAlgebra, FreeAlgebraStore};
use feanor_math::rings::finite::FiniteRing;
use feanor_math::group::*;
use feanor_math::rings::poly::PolyRingStore;
use feanor_math::pid::PrincipalIdealRingStore;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::PolyRing;
use feanor_math::rings::zn::*;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;
use feanor_math::seq::*;
use feanor_math::serialization::{SerializableElementRing, SerializeWithRing, DeserializeWithRing};
use feanor_math::rings::finite::*;
use feanor_math::specialization::{FiniteRingOperation, FiniteRingSpecializable};

use feanor_serde::newtype_struct::*;
use feanor_serde::seq::*;
use serde::{Deserializer, Serialize, Serializer};

use tracing::instrument;

use crate::number_ring::galois::{CyclotomicGaloisGroup, GaloisGroupEl};
use crate::number_ring::poly_remainder::BarettPolyReducer;
use crate::number_ring::{largest_prime_leq_congruent_to_one, AbstractNumberRing, NumberRingQuotient, NumberRingQuotientBases};
use crate::prepared_mul::PreparedMultiplicationRing;
use crate::serde::de::DeserializeSeed;
use crate::{ZZbig, ZZi64};

///
/// Implementation of `R/tR` for any modulus `t in R` (without restriction on the
/// splitting behaviour of `t` over `R`).
/// 
pub struct NumberRingQuotientImplBase<NumberRing, ZnTy, A = Global, C = KaratsubaAlgorithm> 
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    number_ring: NumberRing,
    acting_galois_group: Subgroup<CyclotomicGaloisGroup>,
    generator_galois_conjugates: Vec<(GaloisGroupEl, NumberRingQuotientEl<NumberRing, ZnTy, A, C>)>,
    allocator: A,
    reducer: BarettPolyReducer<ZnTy, C>,
}

///
/// [`RingStore`] for [`NumberRingQuotientImplBase`]
/// 
pub type NumberRingQuotientImpl<NumberRing, ZnTy, A = Global, C = KaratsubaAlgorithm> = RingValue<NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>>;

pub struct NumberRingQuotientEl<NumberRing, ZnTy, A = Global, C = KaratsubaAlgorithm> 
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    ring: PhantomData<NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>>,
    data: Vec<El<ZnTy>, A>
}

pub struct NumberRingQuotientPreparedMultiplicant<NumberRing, ZnTy, A = Global, C = KaratsubaAlgorithm> 
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    ring: PhantomData<NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>>,
    data: C::PreparedConvolutionOperand
}

impl<NumberRing, ZnTy, A, C> NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    #[instrument(skip_all)]
    pub fn quotient_by_integer(number_ring: NumberRing, base_ring: ZnTy, allocator: A, convolution: C) -> RingValue<Self> {
        let poly_ring = DensePolyRing::new(base_ring, "X");
        let ZZX = DensePolyRing::new(ZZbig, "X");
        let gen_poly = number_ring.generating_poly(&ZZX);
        let modulus = poly_ring.lifted_hom(&ZZX, &poly_ring.base_ring().can_hom(&ZZbig).unwrap()).map(gen_poly);
        let acting_galois_group = number_ring.galois_group().clone().into().full_subgroup();
        return Self::create(
            number_ring,
            poly_ring,
            modulus,
            acting_galois_group,
            allocator,
            convolution
        );
    }

    ///
    /// Creates the ring `R/I`, where `R` is the given number ring and `I = (p, f(ϑ))`,
    /// where `p` is the characteristic of the given polynomial ring (a prime) and `f(X)`
    /// is the given polynomial.
    ///  
    /// # Algorithm
    /// 
    /// Our assumption that the given ideal is `I = (p, f(ϑ))` makes things relatively simple.
    /// First, observe that then we have an isomorphism
    /// ```text
    ///   Fp[X]/(gcd_p(f, MiPo(ϑ))) -> R/I,  X -> ϑ
    /// ```
    /// We prove this now. We have `MiPo(ϑ) = f1 ... fr` modulo `p`. Thus, we find
    /// ```text
    ///   I = prod_i (p, fi(ϑ))^ki
    /// ```
    /// Since `p in I`, we see that every `ki in {0, 1}`, and since `p in (p^2, fi(ϑ)fj(ϑ))`,
    /// it follows that `I = (p, f(ϑ)) = (p, f'(ϑ))` for `f(X)' = prod_i fi(X)^ki`. Clearly,
    /// this implies an isomorphism `Fp[X]/(f'(X)) -> R/I`, so it is left to show that
    /// `f' = gcd_p(f, MiPo(ϑ))`. From `(p, f(ϑ)) = (p, f'(ϑ))`, we see that
    /// ```text
    ///   f = p a + b f' + c MiPo(ϑ)    and    f' = p a' + b' f + c' MiPo(ϑ)
    /// ```
    /// Thus `gcd_p(f, MiPo(ϑ)) | f'` and `f' | f` (since `f' | MiPo(ϑ)`) modulo `p`. The claim
    /// follows.
    /// 
    /// # Problem with non-prime characteristics
    /// 
    /// Even the prime-power case `I = (p^e, f(ϑ))` is a problem: We would like an isomorphism
    /// `R/I ~ (Z/p^eZ)[X]/(f'(X))`, but in general that doesn't exist. To see that, note that
    /// ` (Z/p^eZ)[X]/(f'(X))` is always generated by a single element over `Z/p^eZ`, while
    /// `R/I` might be isomorphic to, say, `Z/pZ x Z/p^2Z`, which does not allow a single generator
    /// over `Z/p^2Z`.
    /// 
    #[instrument(skip_all)]
    pub fn quotient_by_ideal<P>(number_ring: NumberRing, base_ring: ZnTy, poly_ring: P, ideal_generator: El<P>, acting_galois_group: Subgroup<CyclotomicGaloisGroup>, allocator: A, convolution: C) -> RingValue<Self>
        where P: RingStore,
            P::Type: PolyRing + PrincipalIdealRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field
    {
        let p = int_cast(poly_ring.base_ring().integer_ring().clone_el(poly_ring.base_ring().modulus()), ZZbig, poly_ring.base_ring().integer_ring());
        assert_el_eq!(ZZbig, &p, int_cast(base_ring.integer_ring().clone_el(base_ring.modulus()), ZZbig, base_ring.integer_ring()));

        let ZZX = DensePolyRing::new(ZZbig, "X");
        let gen_mipo = number_ring.generating_poly(&ZZX);
        let ZZX_to_poly_ring = poly_ring.lifted_hom(&ZZX, poly_ring.base_ring().can_hom(poly_ring.base_ring().integer_ring()).unwrap().compose(poly_ring.base_ring().integer_ring().can_hom(&ZZbig).unwrap()));
        let modulus = poly_ring.ideal_gen(&ideal_generator, &ZZX_to_poly_ring.map(gen_mipo));

        let FpX = DensePolyRing::new(base_ring, "X");
        let Fp = FpX.base_ring();
        let hom = ZnReductionMap::new(poly_ring.base_ring(), &Fp).unwrap();
        let modulus_FpX = FpX.lifted_hom(&poly_ring, &hom).map(modulus);
        return Self::create(number_ring, FpX, modulus_FpX, acting_galois_group, allocator, convolution);
    }

    #[instrument(skip_all)]
    pub fn create(number_ring: NumberRing, poly_ring: DensePolyRing<ZnTy>, modulus: El<DensePolyRing<ZnTy>>, acting_galois_group: Subgroup<CyclotomicGaloisGroup>, allocator: A, convolution: C) -> RingValue<Self> {
        let rank = poly_ring.degree(&modulus).unwrap();
        assert!(rank >= 1);
        assert_eq!(rank, int_cast(acting_galois_group.subgroup_order(), ZZi64, ZZbig) as usize);

        let mut result = Self {
            acting_galois_group: acting_galois_group,
            allocator: allocator,
            generator_galois_conjugates: Vec::new(),
            number_ring: number_ring,
            reducer: BarettPolyReducer::new(poly_ring, &modulus, 2 * rank - 2, convolution)
        };
        result.init_galois_conjugates();
        let result = RingValue::from(result);
        let poly_ring = DensePolyRing::new(result.base_ring(), "X");
        let modulus = result.get_ring().reducer.modulus(&poly_ring);

        // check that the galois group indeed fixes the ideal
        for (_, conjugate) in &result.get_ring().generator_galois_conjugates {
            assert_el_eq!(
                &result,
                result.zero(),
                poly_ring.evaluate(&modulus, conjugate, result.inclusion())
            );
        }

        return result;
    }

    #[instrument(skip_all)]
    fn init_galois_conjugates(&mut self) {
        let number_ring = self.number_ring();
        let acting_galois_group = self.acting_galois_group();
        let hom = self.base_ring().can_hom(self.base_ring().integer_ring()).unwrap().compose(self.base_ring().integer_ring().can_hom(&ZZi64).unwrap());
        
        let galois_output_expansion = number_ring.can_to_inf_norm_expansion_factor() * number_ring.inf_to_can_norm_expansion_factor();
        let tmp_p = largest_prime_leq_congruent_to_one(1 << 57, number_ring.mod_p_required_root_of_unity() as i64).unwrap();
        assert!((tmp_p as f64) > 2. * galois_output_expansion);
        let tmp_Fp = zn_64::Zn::new(tmp_p as u64);
        let tmp_p_bases = number_ring.bases_mod_p(tmp_Fp);
        let x_mult_basis = {
            let mut data = (0..number_ring.rank()).map(|i| if i == 1 { tmp_Fp.one() } else { tmp_Fp.zero() }).collect::<Vec<_>>();
            tmp_p_bases.coeff_basis_to_small_basis(&mut data);
            tmp_p_bases.small_basis_to_mult_basis(&mut data);
            data
        };
        let mut tmp = (0..number_ring.rank()).map(|_| tmp_Fp.zero()).collect::<Vec<_>>();
        let galois_conjugates = acting_galois_group.enumerate_elements().map(|g| {
            tmp_p_bases.permute_galois_action(&x_mult_basis, &mut tmp, &g);
            tmp_p_bases.mult_basis_to_small_basis(&mut tmp);
            tmp_p_bases.small_basis_to_coeff_basis(&mut tmp);
            return (
                g,
                self.from_canonical_basis_extended(tmp.iter().copied().map(|x| hom.map(tmp_Fp.smallest_lift(x))))
            )
        }).collect::<Vec<_>>();

        self.generator_galois_conjugates = galois_conjugates;
    }
}

impl<NumberRing, ZnTy, A, C> NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    fn convolution(&self) -> &C {
        self.reducer.convolution()
    }
}

impl<NumberRing, ZnTy, A, C> Clone for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore + Clone,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type> + Clone
{
    fn clone(&self) -> Self {
        Self {
            allocator: self.allocator.clone(),
            number_ring: self.number_ring.clone(),
            reducer: self.reducer.clone(),
            acting_galois_group: self.acting_galois_group.clone(),
            generator_galois_conjugates: self.generator_galois_conjugates.iter().map(|(g, x)| (g.clone(), self.clone_el(x))).collect()
        }
    }
}

impl<NumberRing, ZnTy, A, C> NumberRingQuotient for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type NumberRing = NumberRing;

    fn number_ring(&self) -> &Self::NumberRing {
        &self.number_ring
    }

    fn acting_galois_group(&self) -> &Subgroup<CyclotomicGaloisGroup> {
        &self.acting_galois_group
    }

    fn apply_galois_action(&self, x: &Self::Element, g: &GaloisGroupEl) -> Self::Element {
        assert!(self.acting_galois_group().dlog(g).is_some());
        let gen_conjugate = &self.generator_galois_conjugates.iter().filter(|(h, _)| self.acting_galois_group().parent().eq_el(h, g)).next().unwrap().1;
        let gen_conjugate_prepared = self.prepare_multiplicant(&gen_conjugate);
        let result = x.data.iter().rev().fold(self.zero(), |current, next| {
            let mut result = self.mul_prepared(&gen_conjugate, &gen_conjugate_prepared, &current, &self.prepare_multiplicant(&current));
            self.base_ring().add_assign_ref(&mut result.data[0], next);
            return result;
        });
        return result;
    }
}

impl<NumberRing, ZnTy, A, C> PreparedMultiplicationRing for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type PreparedMultiplicant = NumberRingQuotientPreparedMultiplicant<NumberRing, ZnTy, A, C>;

    fn prepare_multiplicant(&self, x: &Self::Element) -> Self::PreparedMultiplicant {
        NumberRingQuotientPreparedMultiplicant {
            ring: PhantomData,
            data: self.convolution().prepare_convolution_operand(&x.data, Some(2 * self.rank()), self.base_ring())
        }
    }

    fn mul_prepared(&self, lhs: &Self::Element, lhs_prep: &Self::PreparedMultiplicant, rhs: &Self::Element, rhs_prep: &Self::PreparedMultiplicant) -> Self::Element {
        assert_eq!(self.rank(), lhs.data.len());
        assert_eq!(self.rank(), rhs.data.len());
        let mut result = Vec::with_capacity_in(2 * self.rank(), self.allocator.clone());
        result.resize_with(2 * self.rank(), || self.base_ring().zero());
        self.convolution().compute_convolution_prepared(&lhs.data, Some(&lhs_prep.data), &rhs.data, Some(&rhs_prep.data), &mut result, self.base_ring());
        self.reducer.remainder(&mut result);
        result.truncate(self.rank());
        return NumberRingQuotientEl {
            ring: PhantomData,
            data: result
        };
    }

    fn inner_product_prepared<'a, I>(&self, parts: I) -> Self::Element
        where I: IntoIterator<Item = (&'a Self::Element, &'a Self::PreparedMultiplicant, &'a Self::Element, &'a Self::PreparedMultiplicant)>,
            I::IntoIter: ExactSizeIterator,
            Self: 'a
    {
        let mut result = Vec::with_capacity_in(2 * self.rank(), self.allocator.clone());
        result.resize_with(2 * self.rank(), || self.base_ring().zero());
        self.convolution().compute_convolution_sum(parts.into_iter().map(|(lhs, lhs_prep, rhs, rhs_prep)| {
            assert_eq!(self.rank(), lhs.data.len());
            assert_eq!(self.rank(), rhs.data.len());
            (&lhs.data, Some(&lhs_prep.data), &rhs.data, Some(&rhs_prep.data))
        }), &mut result, self.base_ring());
        self.reducer.remainder(&mut result);
        result.truncate(self.rank());
        return NumberRingQuotientEl {
            ring: PhantomData,
            data: result
        };
    }
}

impl<NumberRing, ZnTy, A, C> PartialEq for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    fn eq(&self, other: &Self) -> bool {
        self.number_ring == other.number_ring && self.base_ring().get_ring() == other.base_ring().get_ring()
    }
}

impl<NumberRing, ZnTy, A, C> RingBase for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type Element = NumberRingQuotientEl<NumberRing, ZnTy, A, C>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend(val.data.iter().map(|x| self.base_ring().clone_el(x)));
        return NumberRingQuotientEl {
            data: result,
            ring: PhantomData
        };
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        for (i, x) in rhs.data.into_iter().enumerate() {
            self.base_ring().add_assign(&mut lhs.data[i], x)
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        for (i, x) in (&rhs.data).into_iter().enumerate() {
            self.base_ring().add_assign_ref(&mut lhs.data[i], x)
        }
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        for (i, x) in (&rhs.data).into_iter().enumerate() {
            self.base_ring().sub_assign_ref(&mut lhs.data[i], x)
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        assert_eq!(lhs.data.len(), self.rank());
        for i in 0..self.rank() {
            self.base_ring().negate_inplace(&mut lhs.data[i]);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = self.mul_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(2 * self.rank(), self.allocator.clone());
        result.resize_with(2 * self.rank(), || self.base_ring().zero());
        self.convolution().compute_convolution_prepared(&lhs.data, None, &rhs.data, None, &mut result, self.base_ring());
        self.reducer.remainder(&mut result);
        result.truncate(self.rank());
        return NumberRingQuotientEl {
            ring: PhantomData,
            data: result
        };
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        for i in 0..self.rank() {
            if !self.base_ring().eq_el(&lhs.data[i], &rhs.data[i]) {
                return false;
            }
        }
        return true;
    }

    fn zero(&self) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|_| self.base_ring().zero()));
        return NumberRingQuotientEl {
            data: result,
            ring: PhantomData
        };
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        assert_eq!(value.data.len(), self.rank());
        value.data.iter().all(|x| self.base_ring().is_zero(x))
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        assert_eq!(value.data.len(), self.rank());
        self.base_ring().is_one(&value.data[0]) && value.data[1..].iter().all(|x| self.base_ring().is_zero(x))
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        assert_eq!(value.data.len(), self.rank());
        self.base_ring().is_neg_one(&value.data[0]) && value.data[1..].iter().all(|x| self.base_ring().is_zero(x))
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    fn is_approximate(&self) -> bool { false }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _env: EnvBindingStrength) -> std::fmt::Result {
        let poly_ring = DensePolyRing::new(self.base_ring(), "X");
        poly_ring.get_ring().dbg(&RingRef::new(self).poly_repr(&poly_ring, value, self.base_ring().identity()), out)
    }

    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)
    }
}

impl<NumberRing, ZnTy, A, C> RingExtension for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type BaseRing = ZnTy;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.reducer.base_ring()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        result.data[0] = x;
        return result;
    }
}

impl<NumberRing, ZnTy, A, C> FreeAlgebra for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type VectorRepresentation<'a> = CloneElFn<&'a [El<ZnTy>], El<ZnTy>, CloneRingEl<&'a ZnTy>>
        where Self: 'a;

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: IntoIterator<Item = El<Self::BaseRing>>
    {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend(vec);
        assert_eq!(result.len(), self.rank());
        return NumberRingQuotientEl {
            data: result,
            ring: PhantomData
        };
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        (&el.data[..]).clone_ring_els(self.base_ring())
    }

    fn canonical_gen(&self) -> Self::Element {
        let mut result = self.zero();
        if result.data.len() > 1 {
            result.data[1] = self.base_ring().one();
        } else {
            result.data[0] = self.base_ring().negate(self.base_ring().clone_el(&self.reducer.modulus_coefficients()[0]));
        }
        return result;
    }

    fn rank(&self) -> usize {
        self.reducer.modulus_deg()
    }
}

pub struct WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    ring: &'a NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>,
}

impl<'a, NumberRing, ZnTy, A, C> Copy for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{}

impl<'a, NumberRing, ZnTy, A, C> Clone for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, NumberRing, ZnTy, A, C> FnOnce<(&'b [El<ZnTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type Output = El<NumberRingQuotientImpl<NumberRing, ZnTy, A, C>>;

    extern "rust-call" fn call_once(self, args: (&'b [El<ZnTy>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, ZnTy, A, C> FnMut<(&'b [El<ZnTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<ZnTy>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, ZnTy, A, C> Fn<(&'b [El<ZnTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    extern "rust-call" fn call(&self, args: (&'b [El<ZnTy>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<NumberRing, ZnTy, A, C> FiniteRingSpecializable for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.execute()
    }
}

impl<NumberRing, ZnTy, A, C> FiniteRing for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type ElementsIter<'a> = MultiProduct<<ZnTy::Type as FiniteRing>::ElementsIter<'a>, WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A, C>, CloneRingEl<&'a ZnTy>, Self::Element>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..self.rank()).map(|_| self.base_ring().elements()), WRTCanonicalBasisElementCreator { ring: self }, CloneRingEl(self.base_ring()))
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|_| self.base_ring().random_element(&mut rng)));
        return NumberRingQuotientEl {
            data: result,
            ring: PhantomData
        };
    }

    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        let characteristic = self.base_ring().size(ZZ)?;
        if ZZ.get_ring().representable_bits().is_none() || ZZ.get_ring().representable_bits().unwrap() >= self.rank() * ZZ.abs_log2_ceil(&characteristic).unwrap() {
            Some(ZZ.pow(characteristic, self.rank()))
        } else {
            None
        }
    }
}

impl<NumberRing, ZnTy, A, C> DivisibilityRing for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let mut mul_matrix: OwnedMatrix<_> = create_multiplication_matrix(RingRef::new(self), rhs, Global);
        let data = self.wrt_canonical_basis(&lhs);
        let mut lhs_matrix: OwnedMatrix<_> = OwnedMatrix::from_fn(self.rank(), 1, |i, _| data.at(i));

        let mut solution: OwnedMatrix<_> = OwnedMatrix::zero(self.rank(), 1, self.base_ring());
        let has_sol = self.base_ring().get_ring().solve_right(mul_matrix.data_mut(), lhs_matrix.data_mut(), solution.data_mut(), Global);
        if has_sol.is_solved() {
            return Some(self.from_canonical_basis((0..self.rank()).map(|i| self.base_ring().clone_el(solution.at(i, 0)))));
        } else {
            return None;
        }
    }
}

impl<NumberRing, ZnTy, A, C> SerializableElementRing for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: SerializableElementRing + ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("RingEl", SerializableSeq::new_with_len(el.data.iter().map(|x| SerializeWithRing::new(x, self.base_ring())), el.data.len())).serialize(serializer)
    }

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de> 
    {
        let result = DeserializeSeedNewtypeStruct::new("RingEl", DeserializeSeedSeq::new(
            (0..(self.rank() + 1)).map(|_| DeserializeWithRing::new(self.base_ring())),
            Vec::with_capacity_in(self.rank(), self.allocator.clone()),
            |mut current, next| { current.push(next); current }
        )).deserialize(deserializer)?;
        if result.len() != self.rank() {
            return Err(serde::de::Error::invalid_length(result.len(), &format!("expected {} elements", self.rank()).as_str()));
        }
        return Ok(NumberRingQuotientEl {
            data: result,
            ring: PhantomData
        });
    }
}

impl<NumberRing, ZnTy, A, C> CanHomFrom<BigIntRingBase> for NumberRingQuotientImplBase<NumberRing, ZnTy, A, C>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    type Homomorphism = <ZnTy::Type as CanHomFrom<BigIntRingBase>>::Homomorphism;

    fn has_canonical_hom(&self, from: &BigIntRingBase) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &BigIntRingBase, el: El<BigIntRing>, hom: &Self::Homomorphism) -> Self::Element {
        self.from(self.base_ring().get_ring().map_in(from, el, hom))
    }

    fn mul_assign_map_in(&self, from: &BigIntRingBase, lhs: &mut Self::Element, rhs: <BigIntRingBase as RingBase>::Element, hom: &Self::Homomorphism) {
        self.mul_assign_base(lhs, &self.base_ring().get_ring().map_in(from, rhs, hom));
    }

    fn mul_assign_map_in_ref(&self, from: &BigIntRingBase, lhs: &mut Self::Element, rhs: &<BigIntRingBase as RingBase>::Element, hom: &Self::Homomorphism) {
        self.mul_assign_base(lhs, &self.base_ring().get_ring().map_in_ref(from, rhs, hom));
    }
}

impl<NumberRing, ZnTy1, ZnTy2, A1, A2, C1, C2> CanHomFrom<NumberRingQuotientImplBase<NumberRing, ZnTy2, A2, C2>> for NumberRingQuotientImplBase<NumberRing, ZnTy1, A1, C1>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy1: RingStore,
        ZnTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        C1: ConvolutionAlgorithm<ZnTy1::Type> + Clone,
        ZnTy2: RingStore,
        ZnTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        C2: ConvolutionAlgorithm<ZnTy2::Type> + Clone,
        ZnTy1::Type: CanHomFrom<ZnTy2::Type>
{
    type Homomorphism = <ZnTy1::Type as CanHomFrom<ZnTy2::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &NumberRingQuotientImplBase<NumberRing, ZnTy2, A2, C2>) -> Option<Self::Homomorphism> {
        if self.number_ring == from.number_ring {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &NumberRingQuotientImplBase<NumberRing, ZnTy2, A2, C2>, el: <NumberRingQuotientImplBase<NumberRing, ZnTy2, A2, C2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        assert_eq!(el.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(&el.data[i]), hom)));
        return NumberRingQuotientEl {
            data: result,
            ring: PhantomData
        };
    }
}

impl<NumberRing, ZnTy1, ZnTy2, A1, A2, C1, C2> CanIsoFromTo<NumberRingQuotientImplBase<NumberRing, ZnTy2, A2, C2>> for NumberRingQuotientImplBase<NumberRing, ZnTy1, A1, C1>
    where NumberRing: AbstractNumberRing + Clone,
        ZnTy1: RingStore,
        ZnTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        C1: ConvolutionAlgorithm<ZnTy1::Type> + Clone,
        ZnTy2: RingStore,
        ZnTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        C2: ConvolutionAlgorithm<ZnTy2::Type> + Clone,
        ZnTy1::Type: CanIsoFromTo<ZnTy2::Type>
{
    type Isomorphism = <ZnTy1::Type as CanIsoFromTo<ZnTy2::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &NumberRingQuotientImplBase<NumberRing, ZnTy2, A2, C2>) -> Option<Self::Isomorphism> {
        if self.number_ring == from.number_ring {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &NumberRingQuotientImplBase<NumberRing, ZnTy2, A2, C2>, el: Self::Element, iso: &Self::Isomorphism) -> <NumberRingQuotientImplBase<NumberRing, ZnTy2, A2, C2> as RingBase>::Element {
        assert_eq!(el.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(self.rank(), from.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.data[i]), iso)));
        return NumberRingQuotientEl {
            data: result,
            ring: PhantomData
        };
    }
}

#[cfg(test)]
use feanor_math::algorithms::convolution::STANDARD_CONVOLUTION;
#[cfg(test)]
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;

#[cfg(test)]
pub fn test_with_number_ring<NumberRing: AbstractNumberRing>(number_ring: NumberRing) {

    let base_ring = zn_64::Zn::new(5);
    let rank = number_ring.rank();
    let ring = NumberRingQuotientImplBase::quotient_by_integer(number_ring, base_ring, Global, STANDARD_CONVOLUTION);

    let elements = vec![
        ring.zero(),
        ring.one(),
        ring.neg_one(),
        ring.int_hom().map(2),
        ring.canonical_gen(),
        ring.pow(ring.canonical_gen(), 2),
        ring.pow(ring.canonical_gen(), rank - 1),
        ring.int_hom().mul_map(ring.canonical_gen(), 2),
        ring.int_hom().mul_map(ring.pow(ring.canonical_gen(), rank - 1), 2)
    ];

    feanor_math::ring::generic_tests::test_ring_axioms(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::ring::generic_tests::test_self_iso(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(&ring);
    feanor_math::serialization::generic_tests::test_serialization(&ring, elements.iter().map(|x| ring.clone_el(x)));
}

#[test]
fn test_quotient_by_ideal() {
    let number_ring: Pow2CyclotomicNumberRing = Pow2CyclotomicNumberRing::new(8);
    let base_ring = zn_big::Zn::new(ZZbig, int_cast(17, ZZbig, ZZi64)).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new((&base_ring).as_field().ok().unwrap(), "X");
    let [t] = poly_ring.with_wrapped_indeterminate(|X| [X - 2]);
    let acting_galois_group = number_ring.galois_group().clone().into().subgroup([]);
    let ring = NumberRingQuotientImplBase::quotient_by_ideal(
        number_ring, 
        &base_ring, 
        poly_ring, 
        t, 
        acting_galois_group, 
        Global, 
        STANDARD_CONVOLUTION
    );
    assert_eq!(1, ring.rank());
    let galois_group = ring.get_ring().acting_galois_group().parent();
    assert_eq!(17, ring.elements().count());
    feanor_math::ring::generic_tests::test_ring_axioms(&ring, ring.elements());
    assert_el_eq!(&ring, ring.one(), ring.get_ring().apply_galois_action(&ring.one(), &galois_group.identity()));

    let number_ring: Pow2CyclotomicNumberRing = Pow2CyclotomicNumberRing::new(8);
    let galois_group = number_ring.galois_group().get_group();
    let base_ring = zn_big::Zn::new(ZZbig, int_cast(17, ZZbig, ZZi64)).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new((&base_ring).as_field().ok().unwrap(), "X");
    let [t] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 4]);
    let acting_galois_group = galois_group.clone().subgroup([galois_group.from_representative(5)]);
    let ring = NumberRingQuotientImplBase::quotient_by_ideal(
        number_ring, 
        &base_ring, 
        poly_ring, 
        t, 
        acting_galois_group, 
        Global, 
        STANDARD_CONVOLUTION
    );
    assert_eq!(2, ring.rank());
    let galois_group = ring.get_ring().acting_galois_group().parent().get_group();
    assert_el_eq!(ZZbig, int_cast(2, ZZbig, ZZi64), ring.get_ring().acting_galois_group().subgroup_order());
    assert_el_eq!(&ring, ring.negate(ring.canonical_gen()), ring.get_ring().apply_galois_action(&ring.canonical_gen(), &galois_group.from_representative(5)));
}