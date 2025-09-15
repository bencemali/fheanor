use std::alloc::Allocator;

use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::algorithms::poly_gcd::hensel::hensel_lift_factorization;
use feanor_math::computation::DontObserve;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::field::Field;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, BigIntRing};
use feanor_math::primitive_int::StaticRing;
use feanor_math::reduce_lift::poly_factor_gcd::PolyGCDLocallyIntermediateReductionMap;
use feanor_math::ring::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::PolyRingStore;
use feanor_math::rings::zn::*;
use tracing::instrument;

#[instrument(skip_all)]
pub fn hensel_lift_factor<R1, R2, A1, A2, C1, C2>(from_ring: &DensePolyRing<R1, A1, C1>, to_ring: &DensePolyRing<R2, A2, C2>, f: &El<DensePolyRing<R1, A1, C1>>, g: El<DensePolyRing<R2, A2, C2>>) -> El<DensePolyRing<R1, A1, C1>>
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
    let Fp = zn_64::Zn::new(p as u64).as_field().ok().unwrap();
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
