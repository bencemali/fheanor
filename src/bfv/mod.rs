#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::sync::Arc;
use std::ops::Range;
use std::cell::RefCell;
use std::fmt::Display;

use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::algorithms::miller_rabin::prev_prime;
use feanor_math::matrix::OwnedMatrix;
use feanor_math::primitive_int::StaticRingBase;
use feanor_math::ring::*;
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::primitive_int::StaticRing;
use feanor_math::integer::*;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::seq::*;
use feanor_math::rings::finite::FiniteRingStore;
use tracing::instrument;

use crate::ciphertext_ring::perform_rns_op;
use crate::ciphertext_ring::BGFVCiphertextRing;
use crate::circuit::evaluator::DefaultCircuitEvaluator;
use crate::circuit::{Coefficient, PlaintextCircuit};
use crate::cyclotomic::*;
use crate::gadget_product::{GadgetProductLhsOperand, GadgetProductRhsOperand};
use crate::gadget_product::digits::*;
use crate::ntt::{HERingNegacyclicNTT, HERingConvolution};
use crate::ciphertext_ring::double_rns_managed::*;
use crate::number_ring::hypercube::isomorphism::*;
use crate::number_ring::hypercube::structure::HypercubeStructure;
use crate::number_ring::{largest_prime_leq_congruent_to_one, sample_primes, extend_sampled_primes, HECyclotomicNumberRing, HENumberRing};
use crate::number_ring::quotient::*;
use crate::number_ring::pow2_cyclotomic::*;
use crate::number_ring::composite_cyclotomic::*;
use crate::ciphertext_ring::single_rns_ring::SingleRNSRingBase;
use crate::rnsconv::bfv_rescale::{AlmostExactRescaling, AlmostExactRescalingConvert};
use crate::rnsconv::RNSOperation;
use crate::rnsconv::shared_lift::AlmostExactSharedBaseConversion;
use crate::DefaultCiphertextAllocator;
use crate::*;

use rand::{Rng, CryptoRng};
use rand_distr::StandardNormal;

///
/// Contains the implementation of bootstrapping for BFV.
/// 
pub mod bootstrap;

pub type NumberRing<Params: BFVInstantiation> = <<Params as BFVInstantiation>::CiphertextRing as BGFVCiphertextRing>::NumberRing;
pub type PlaintextRing<Params: BFVInstantiation> = RingValue<<Params as BFVInstantiation>::PlaintextRing>;
pub type SecretKey<Params: BFVInstantiation> = El<CiphertextRing<Params>>;
pub type KeySwitchKey<'a, Params: BFVInstantiation> = (GadgetProductOperand<'a, Params>, GadgetProductOperand<'a, Params>);
pub type RelinKey<'a, Params: BFVInstantiation> = KeySwitchKey<'a, Params>;
pub type CiphertextRing<Params: BFVInstantiation> = RingValue<Params::CiphertextRing>;
pub type Ciphertext<Params: BFVInstantiation> = (El<CiphertextRing<Params>>, El<CiphertextRing<Params>>);
pub type GadgetProductOperand<'a, Params: BFVInstantiation> = GadgetProductRhsOperand<Params::CiphertextRing>;

///
/// Trait for types that represent an instantiation of BFV.
/// 
/// For a few more details on how this works, see [`crate::examples::bfv_basics`].
/// 
/// # Design
/// 
/// Generally speaking, Fheanor tries to avoid storing parameters, in particular
/// plaintext and ciphertext, in a single object. This allows users to work with
/// multiple different plaintext and ciphertext moduli and rings in a single context.
/// 
/// In a sense, the optimal design would thus be for the HE schemes to just be a
/// collection of global functions, accepting the plaintext and ciphertext rings as
/// parameters. 
/// 
/// However, we don't go quite that far, since this approach would make
/// it very hard to provide optimized specializations for certain settings.
/// Instead, we bundle just the information on the number ring in a [`BFVInstantiation`],
/// which then has all the BFV-related functionality as associated functions. A single
/// [`BFVInstantiation`] still supports construction many plaintext and ciphertext rings
/// with different moduli, as well as keys w.r.t. different ring and parameters, as long
/// as all of them live over the same number ring. Since most optimizations are designed
/// for certain classes of number rings, this seems like a reasonable compromise.
/// 
/// Note that it is still supported and valid to exchange data between different number
/// rings, by using functionality provided by the plaintext and ciphertext rings.
/// 
pub trait BFVInstantiation {

    ///
    /// Type of the ciphertext ring `R/qR`.
    /// 
    type CiphertextRing: BGFVCiphertextRing + CyclotomicRing + FiniteRing;

    ///
    /// Type of the plaintext base ring `Z/tZ`.
    /// 
    type PlaintextZnRing: NiceZn;
    
    ///
    /// Type of the plaintext ring `R/tR`.
    /// 
    type PlaintextRing: CyclotomicRing<BaseRing = RingValue<Self::PlaintextZnRing>>;

    ///
    /// The number ring `R` we work in, i.e. the ciphertext ring is `R/qR` and
    /// the plaintext ring is `R/tR`.
    /// 
    fn number_ring(&self) -> NumberRing<Self>;

    ///
    /// Creates the ciphertext ring `R/qR` and the extended-modulus ciphertext ring
    /// `R/qq'R` that is necessary for homomorphic multiplication.
    /// 
    /// The modulus for `q` is chosen such that its bitlength is within `log2_q`.
    /// The modulus `q'` is chosen so that `R/qq'R` can represent the result of
    /// the intermediate product of the shortest lifts of two elements of `R/qR`.
    /// 
    fn create_ciphertext_rings(&self, log2_q: Range<usize>) -> (CiphertextRing<Self>, CiphertextRing<Self>);

    ///
    /// Creates the plaintext ring `R/tR` for the given plaintext modulus `t`.
    /// 
    fn create_plaintext_ring(&self, t: El<BigIntRing>) -> PlaintextRing<Self>;

    ///
    /// Generates a secret key, using the randomness of the given rng.
    /// 
    /// If `sk_hwt` is set, the secret will be a random ring element with exactly `sk_hwt` entries (w.r.t.
    /// coefficient basis) in `{-1, 1}`, and the others as `0`. If `hwt` is not set, the secret will be
    /// a ring element whose coefficient basis coefficients are drawn uniformly at random from `{-1, 0, 1}`.
    /// 
    /// If you need another kind of secret, consider creating the ring element yourself using `C.from_canonical_basis()`.
    /// 
    #[instrument(skip_all)]
    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk_hwt: Option<usize>) -> SecretKey<Self> {
        assert!(sk_hwt.is_none() || sk_hwt.unwrap() * 3 <= C.rank() * 2, "it does not make sense to take more than 2/3 of secret key entries in {{-1, 1}}");
        if let Some(hwt) = sk_hwt {
            let mut result_data = (0..C.rank()).map(|_| 0).collect::<Vec<_>>();
            for _ in 0..hwt {
                let mut i = rng.next_u32() as usize % C.rank();
                while result_data[i] != 0 {
                    i = rng.next_u32() as usize % C.rank();
                }
                result_data[i] = (rng.next_u32() % 2) as i32 * 2 - 1;
            }
            let result = C.from_canonical_basis(result_data.into_iter().map(|c| C.base_ring().int_hom().map(c)));
            return result;
        } else {
            let result = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((rng.next_u32() % 3) as i32 - 1)));
            return result;
        }
    }
    
    ///
    /// Generates a new encryption of zero using the secret key and the randomness of the given rng.
    /// 
    /// The noise is chosen according to the rounded Gaussian distribution with standard deviation `noise_sigma`.
    /// 
    #[instrument(skip_all)]
    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>, noise_sigma: f64) -> Ciphertext<Self> {
        let a = C.random_element(|| rng.next_u64());
        let mut b = C.negate(C.mul_ref(&a, &sk));
        let e = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((rng.sample::<f64, _>(StandardNormal) * noise_sigma).round() as i32)));
        C.add_assign(&mut b, e);
        return (b, a);
    }
    
    ///
    /// Creates a transparent encryption of zero, i.e. a noiseless encryption that does not hide
    /// the encrypted value - everyone can read it, even without access to the secret key.
    /// 
    /// Often used to initialize an accumulator (or similar) during algorithms. 
    /// 
    #[instrument(skip_all)]
    fn transparent_zero(C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        (C.zero(), C.zero())
    }

    ///
    /// Encrypts the given value, using the randomness of the given rng.
    /// 
    /// The noise is chosen according to the rounded Gaussian distribution with standard deviation `noise_sigma`.
    /// 
    #[instrument(skip_all)]
    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, m: &El<PlaintextRing<Self>>, sk: &SecretKey<Self>, noise_sigma: f64) -> Ciphertext<Self> {
        Self::hom_add_plain(P, C, m, Self::enc_sym_zero(C, rng, sk, noise_sigma))
    }

    ///
    /// Creates an encryption of the secret key - this is always easily possible in BFV.
    /// 
    #[instrument(skip_all)]
    fn enc_sk(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        let ZZ = P.base_ring().integer_ring();
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(ZZ.clone_el(P.base_ring().modulus()), ZZbig, ZZ)
        );
        (C.zero(), C.inclusion().map(C.base_ring().coerce(&ZZbig, Delta)))
    }
    
    ///
    /// Given `q/t m + e`, removes the noise term `e`, thus returns `q/t m`.
    /// 
    /// Used during [`BFVInstantiation::dec()`] and [`BFVInstantiation::noise_budget()`].
    /// 
    #[instrument(skip_all)]
    fn remove_noise(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, c: &El<CiphertextRing<Self>>) -> El<PlaintextRing<Self>> {
        let coefficients = C.wrt_canonical_basis(c);
        let ZZ = P.base_ring().integer_ring();
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(ZZ.clone_el(P.base_ring().modulus()), ZZbig, ZZ)
        );
        let modulo = P.base_ring().can_hom(&ZZbig).unwrap();
        return P.from_canonical_basis((0..coefficients.len()).map(|i| modulo.map(ZZbig.rounded_div(C.base_ring().smallest_lift(coefficients.at(i)), &Delta))));
    }
    
    ///
    /// Decrypts a given ciphertext.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is a valid
    /// BFV encryption w.r.t. the given plaintext modulus.
    /// 
    #[instrument(skip_all)]
    fn dec(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, sk: &SecretKey<Self>) -> El<PlaintextRing<Self>> {
        let (c0, c1) = ct;
        let noisy_m = C.add(c0, C.mul_ref_snd(c1, sk));
        return Self::remove_noise(P, C, &noisy_m);
    }
    
    ///
    /// Decrypts a given ciphertext and prints its value to stdout.
    /// Designed for debugging purposes.
    /// 
    #[instrument(skip_all)]
    fn dec_println(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        println!("ciphertext (noise budget: {}):", Self::noise_budget(P, C, ct, sk));
        P.println(&m);
        println!();
    }
    
    ///
    /// Decrypts a given ciphertext and prints the values in its slots to stdout.
    /// Designed for debugging purposes.
    /// 
    #[instrument(skip_all)]
    fn dec_println_slots(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>, dir: Option<&str>)
        where DecoratedBaseRingBase<PlaintextRing<Self>>: CanIsoFromTo<BaseRing<PlaintextRing<Self>>>
    {
        let ZZ = P.base_ring().integer_ring();
        let (p, _e) = is_prime_power(ZZ, P.base_ring().modulus()).unwrap();
        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(P.m() as u64), int_cast(p, ZZbig, ZZ));

        let H = if let Some(dir) = dir {
            HypercubeIsomorphism::new_cache_file::<false>(P, hypercube, dir)
        } else {
            HypercubeIsomorphism::new::<false>(P, hypercube)
        };
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        println!("ciphertext (noise budget: {}):", Self::noise_budget(P, C, ct, sk));
        for a in H.get_slot_values(&m) {
            H.slot_ring().println(&a);
        }
        println!();
    }
    
    ///
    /// Computes an encryption of the sum of two encrypted messages.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertexts are defined over the given ring, and are
    /// BFV encryptions w.r.t. compatible plaintext moduli.
    /// 
    #[instrument(skip_all)]
    fn hom_add(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        let (lhs0, lhs1) = lhs;
        let (rhs0, rhs1) = rhs;
        return (C.add_ref(&lhs0, &rhs0), C.add_ref(&lhs1, &rhs1));
    }
    
    ///
    /// Computes an encryption of the difference of two encrypted messages.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertexts are defined over the given ring, and are
    /// BFV encryptions w.r.t. compatible plaintext moduli.
    /// 
    #[instrument(skip_all)]
    fn hom_sub(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        let (lhs0, lhs1) = lhs;
        let (rhs0, rhs1) = rhs;
        return (C.sub_ref(&lhs0, rhs0), C.sub_ref(&lhs1, rhs1));
    }
    
    ///
    /// Copies a ciphertext.
    /// 
    #[instrument(skip_all)]
    fn clone_ct(C: &CiphertextRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
        (C.clone_el(&ct.0), C.clone_el(&ct.1))
    }
    
    ///
    /// Computes an encryption of the sum of an encrypted message and a plaintext.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is a valid
    /// BFV encryption w.r.t. the given plaintext modulus.
    /// 
    #[instrument(skip_all)]
    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        let ZZ = P.base_ring().integer_ring();
        let mut m = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
        let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(ZZ.clone_el(P.base_ring().modulus()), ZZbig, ZZ)
        ));
        C.inclusion().mul_assign_ref_map(&mut m, &Delta);
        return (C.add(ct.0, m), ct.1);
    }
    
    ///
    /// Computes an encryption of the product of an encrypted message and a plaintext.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is
    /// BFV encryption w.r.t. the given plaintext modulus.
    /// 
    #[instrument(skip_all)]
    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        Self::hom_mul_plain_encoded(P, C, &Self::encode_plain_multiplicant(P, C, m), ct)
    }
    
    ///
    /// Computes the smallest lift of the plaintext ring element to the ciphertext
    /// ring. The result can be used in [`BFVInstantiation::hom_mul_plain_encoded()`]
    /// to compute plaintext-ciphertext multiplication faster.
    /// 
    /// Note that (as opposed to BFV), encoding of plaintexts that are used as multiplicants
    /// (i.e. multiplied to ciphertexts) is different than for plaintexts that are used as summands
    /// (i.e. added to ciphertexts). Currently only the former is supported for BFV.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the plaintext ring element is defined over the given plaintext ring.
    /// 
    #[instrument(skip_all)]
    fn encode_plain_multiplicant(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>) -> El<CiphertextRing<Self>> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        return C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
    }

    ///
    /// Returns an encryption of the product of the encrypted input and the given plaintext,
    /// which has already been lifted/encoded into the ciphertext ring.
    /// 
    /// When the plaintext is given as an element of `P`, use [`BFVInstantiation::hom_mul_plain()`]
    /// instead. However, internally, the plaintext will be lifted into the ciphertext ring during
    /// the multiplication, and if this is performed in advance (via [`BFVInstantiation::encode_plain_multiplicant()`]),
    /// multiplication will be faster.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is a valid
    /// BFV encryptions w.r.t. compatible plaintext moduli.
    /// 
    #[instrument(skip_all)]
    fn hom_mul_plain_encoded(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<CiphertextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        (C.mul_ref_snd(ct.0, m), C.mul_ref_snd(ct.1, m))
    }

    ///
    /// Computes an encryption of the product of an encrypted message and an integer plaintext.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is a valid
    /// BFV encryption w.r.t. the given plaintext modulus.
    /// 
    #[instrument(skip_all)]
    fn hom_mul_plain_i64(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: i64, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        (C.int_hom().mul_map(ct.0, m as i32), C.int_hom().mul_map(ct.1, m as i32))
    }
    
    ///
    /// Computes the "noise budget" of a given ciphertext.
    /// 
    /// Concretely, the noise budget is `log(q/(t|e|))`, where `t` is the plaintext modulus
    /// and `|e|` is the `l_inf`-norm of the noise term. This will decrease during homomorphic
    /// operations, and if it reaches zero, decryption may yield incorrect results.
    /// 
    #[instrument(skip_all)]
    fn noise_budget(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) -> usize {
        let (c0, c1) = Self::clone_ct(C, ct);
        let noisy_m = C.add(c0, C.mul_ref_snd(c1, sk));
        let coefficients = C.wrt_canonical_basis(&noisy_m);
        let ZZ = P.base_ring().integer_ring();
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(ZZ.clone_el(P.base_ring().modulus()), ZZbig, ZZ)
        );
        let log2_size_of_noise = <_ as Iterator>::max((0..coefficients.len()).map(|i| {
            let c = C.base_ring().smallest_lift(coefficients.at(i));
            let size = ZZbig.abs_log2_ceil(&ZZbig.sub_ref_fst(&c, ZZbig.mul_ref_snd(ZZbig.rounded_div(ZZbig.clone_el(&c), &Delta), &Delta)));
            return size.unwrap_or(0);
        })).unwrap();
        return ZZbig.abs_log2_ceil(C.base_ring().modulus()).unwrap().saturating_sub(log2_size_of_noise + P.base_ring().integer_ring().abs_log2_ceil(P.base_ring().modulus()).unwrap() + 1);
    }
    
    ///
    /// Generates a relinearization key, necessary to compute homomorphic multiplications.
    /// 
    /// The parameter `digits` defined the RNS-based gadget vector to use for the gadget product
    /// during key-switching. More concretely, when performing key-switching, the ciphertext
    /// will be decomposed into multiple small parts, which are then multiplied with the components
    /// of the key-switching key. Thus, a large number of small digits will result in lower (additive)
    /// noise growth during key-switching, at the cost of higher performance.
    /// 
    /// The noise is chosen according to the rounded Gaussian distribution with standard deviation `noise_sigma`.
    /// 
    #[instrument(skip_all)]
    fn gen_rk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, digits: &RNSGadgetVectorDigitIndices, noise_sigma: f64) -> RelinKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk, digits, noise_sigma)
    }
    
    ///
    /// Computes an encryption of the product of two encrypted messages.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertexts are defined over the given ring, and are
    /// BFV encryptions w.r.t. the given plaintext modulus.
    /// 
    /// As opposed to BGV, hybrid key switching is currently not implemented for BFV.
    /// You can achieve the same effect by manually modulus-switching ciphertext to a higher
    /// modulus before calling `hom_mul()` (although this will be less efficient than 
    /// performing only the key-switch modulo the larger modulus).
    /// 
    #[instrument(skip_all)]
    fn hom_mul<'a>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, C_mul: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c00, c01) = lhs;
        let (c10, c11) = rhs;

        let mut lift = lift_to_Cmul::<Self>(C, C_mul);
        let c00_lifted = lift(&c00);
        let c01_lifted = lift(&c01);
        let c10_lifted = lift(&c10);
        let c11_lifted = lift(&c11);

        let [lifted0, lifted1, lifted2] = C_mul.get_ring().two_by_two_convolution([&c00_lifted, &c01_lifted], [&c10_lifted, &c11_lifted]);

        let mut scale_down = rescale_to_C::<Self>(P, C, C_mul);
        let res0 = scale_down(&lifted0);
        let res1 = scale_down(&lifted1);
        let res2 = scale_down(&lifted2);

        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &res2, rk.0.gadget_vector_digits());
        let (s0, s1) = rk;
        return (C.add_ref(&res0, &op.gadget_product(s0, C.get_ring())), C.add_ref(&res1, &op.gadget_product(s1, C.get_ring())));
        
    }
    
    ///
    /// Computes an encryption of the square of an encrypted messages.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertexts are defined over the given ring, and are
    /// BFV encryptions w.r.t. the given plaintext modulus.
    /// 
    /// As opposed to BGV, hybrid key switching is currently not implemented for BFV.
    /// You can achieve the same effect by manually modulus-switching ciphertext to a higher
    /// modulus before calling `hom_square()` (although this will be less efficient than 
    /// performing only the key-switch modulo the larger modulus).
    /// 
    #[instrument(skip_all)]
    fn hom_square<'a>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, C_mul: &CiphertextRing<Self>, val: Ciphertext<Self>, rk: &RelinKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c0, c1) = val;

        let mut lift = lift_to_Cmul::<Self>(C, C_mul);
        let c0_lifted = lift(&c0);
        let c1_lifted = lift(&c1);

        let [lifted0, lifted1, lifted2] = C_mul.get_ring().two_by_two_convolution([&c0_lifted, &c1_lifted], [&c0_lifted, &c1_lifted]);

        let mut scale_down = rescale_to_C::<Self>(P, C, C_mul);
        let res0 = scale_down(&lifted0);
        let res1 = scale_down(&lifted1);
        let res2 = scale_down(&lifted2);

        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &res2, rk.0.gadget_vector_digits());
        let (s0, s1) = rk;
        return (C.add_ref(&res0, &op.gadget_product(s0, C.get_ring())), C.add_ref(&res1, &op.gadget_product(s1, C.get_ring())));
        
    }
    
    ///
    /// Generates a key-switch key. 
    /// 
    /// In particular, this is used to generate relinearization keys (via [`BFVInstantiation::gen_rk()`])
    /// or Galois keys (via [`BFVInstantiation::gen_gk()`]).
    /// 
    /// The parameter `digits` defined the RNS-based gadget vector to use for the gadget product
    /// during key-switching. More concretely, when performing key-switching, the ciphertext
    /// will be decomposed into multiple small parts, which are then multiplied with the components
    /// of the key-switching key. Thus, a large number of small digits will result in lower (additive)
    /// noise growth during key-switching, at the cost of higher performance.
    /// 
    /// The noise is chosen according to the rounded Gaussian distribution with standard deviation `noise_sigma`.
    /// 
    #[instrument(skip_all)]
    fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: &RNSGadgetVectorDigitIndices, noise_sigma: f64) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let mut res0 = GadgetProductRhsOperand::new_with(C.get_ring(), digits.to_owned());
        let mut res1 = GadgetProductRhsOperand::new_with(C.get_ring(), digits.to_owned());
        for (i, digit) in digits.iter().enumerate() {
            let (c0, c1) = Self::enc_sym_zero(C, &mut rng, new_sk, noise_sigma);
            let factor = C.base_ring().get_ring().from_congruence((0..C.base_ring().len()).map(|i2| {
                let Fp = C.base_ring().at(i2);
                if digit.contains(&i2) { Fp.one() } else { Fp.zero() } 
            }));
            let mut payload = C.clone_el(&old_sk);
            C.inclusion().mul_assign_ref_map(&mut payload, &factor);
            C.add_assign_ref(&mut payload, &c0);
            res0.set_rns_factor(C.get_ring(), i, payload);
            res1.set_rns_factor(C.get_ring(), i, c1);
        }
        return (res0, res1);
    }
    
    ///
    /// Using a key-switch key, computes an encryption encrypting the same message as the
    /// given ciphertext under a different secret key.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is a valid
    /// BFV encryptions w.r.t. the given plaintext modulus.
    /// 
    /// As opposed to BGV, hybrid key switching is currently not implemented for BFV.
    /// You can achieve the same effect by manually modulus-switching ciphertext to a higher
    /// modulus before calling `key_switch()`.
    /// 
    #[instrument(skip_all)]
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c0, c1) = ct;
        let (s0, s1) = switch_key;
        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &c1, switch_key.0.gadget_vector_digits());
        return (
            C.add_ref_snd(c0, &op.gadget_product(s0, C.get_ring())),
            op.gadget_product(s1, C.get_ring())
        );
    }
    
    ///
    /// Modulus-switches from `R/qR` to `R/tR`, where the latter one is given as a plaintext ring.
    /// In particular, this is necessary during bootstrapping.
    /// 
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>);
        
    ///
    /// Modulus-switches a ciphertext.
    /// 
    /// More concretely, we have the ring `Cold = R/qR` and `Cnew = R/q'R`.
    /// Given a ciphertext `ct` over `R/qR`, this function then computes a ciphertext 
    /// encrypting the same message over `R/q'R` (w.r.t. the secret key `sk mod q'`).
    /// 
    #[instrument(skip_all)]
    fn mod_switch_ct(_P: &PlaintextRing<Self>, Cnew: &CiphertextRing<Self>, Cold: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let num_moduli = Cnew.base_ring().as_iter().filter(|Zp| Cold.base_ring().as_iter().all(|other_Zp| other_Zp.modulus() != Zp.modulus())).cloned().collect::<Vec<_>>();
        let den_moduli_indices = Cold.base_ring().as_iter().enumerate().filter(|(_, Zp)| Cnew.base_ring().as_iter().all(|other_Zp| other_Zp.modulus() != Zp.modulus())).map(|(i, _)| i).collect::<Vec<_>>();
        let mod_switch = AlmostExactRescaling::new_with(
            Cold.base_ring().as_iter().map(|Zp| *Zp).collect(),
            num_moduli,
            den_moduli_indices,
            Global
        );
        assert!(Cnew.base_ring().as_iter().zip(mod_switch.output_rings()).all(|(l, r)| l.get_ring() == r.get_ring()), "invalid modulus switch");
        return (
            perform_rns_op(Cnew.get_ring(), Cold.get_ring(), &ct.0, &mod_switch),
            perform_rns_op(Cnew.get_ring(), Cold.get_ring(), &ct.1, &mod_switch)
        );
    }
    
    ///
    /// Generates a Galois key, usable for homomorphically applying the Galois automorphisms
    /// defined by the given element of the Galois group.
    /// 
    /// The parameter `digits` defined the RNS-based gadget vector to use for the gadget product
    /// during key-switching. More concretely, when performing key-switching, the ciphertext
    /// will be decomposed into multiple small parts, which are then multiplied with the components
    /// of the key-switching key. Thus, a large number of small digits will result in lower (additive)
    /// noise growth during key-switching, at the cost of higher performance.
    /// 
    /// The noise is chosen according to the rounded Gaussian distribution with standard deviation `noise_sigma`.
    /// 
    #[instrument(skip_all)]
    fn gen_gk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, g: CyclotomicGaloisGroupEl, digits: &RNSGadgetVectorDigitIndices, noise_sigma: f64) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.get_ring().apply_galois_action(sk, g), sk, digits, noise_sigma)
    }
    
    ///
    /// Computes an encryption of `sigma(x)`, where `x` is the message encrypted by the given ciphertext
    /// and `sigma` is the given Galois automorphism.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is a valid
    /// BFV encryptions w.r.t. the given plaintext modulus.
    /// 
    /// As opposed to BGV, hybrid key switching is currently not implemented for BFV.
    /// You can achieve the same effect by manually modulus-switching ciphertext to a higher
    /// modulus before calling `hom_galois()`.
    /// 
    #[instrument(skip_all)]
    fn hom_galois<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, g: CyclotomicGaloisGroupEl, gk: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        Self::key_switch(C, (
            C.get_ring().apply_galois_action(&ct.0, g),
            C.get_ring().apply_galois_action(&ct.1, g)
        ), gk)
    }
    
    ///
    /// Homomorphically applies multiple Galois automorphisms at once.
    /// Functionally, this is equivalent to calling [`BFVInstantiation::hom_galois()`]
    /// multiple times, but can be faster.
    /// 
    /// All used Galois keys must use the same digits, i.e. the same RNS-based
    /// gadget vector.
    /// 
    /// This function does not perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is a valid
    /// BFV encryptions w.r.t. the given plaintext modulus.
    /// 
    /// As opposed to BGV, hybrid key switching is currently not implemented for BFV.
    /// You can achieve the same effect by manually modulus-switching ciphertext to a higher
    /// modulus before calling `hom_galois()`.
    /// 
    #[instrument(skip_all)]
    fn hom_galois_many<'a, 'b, V>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, gs: &[CyclotomicGaloisGroupEl], gks: V) -> Vec<Ciphertext<Self>>
        where V: VectorFn<&'b KeySwitchKey<'a, Self>>,
            KeySwitchKey<'a, Self>: 'b,
            'a: 'b,
            Self: 'a
    {
        let digits = gks.at(0).0.gadget_vector_digits();
        let has_same_digits = |gk: &GadgetProductRhsOperand<_>| gk.gadget_vector_digits().len() == digits.len() && gk.gadget_vector_digits().iter().zip(digits.iter()).all(|(l, r)| l == r);
        assert!(gks.iter().all(|gk| has_same_digits(&gk.0) && has_same_digits(&gk.1)));
        let (c0, c1) = ct;
        let c1_op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &c1, digits);
        let c1_op_gs = c1_op.apply_galois_action_many(C.get_ring(), gs);
        let c0_gs = C.get_ring().apply_galois_action_many(&c0, gs).into_iter();
        assert_eq!(gks.len(), c1_op_gs.len());
        assert_eq!(gks.len(), c0_gs.len());
        return c0_gs.zip(c1_op_gs.iter()).enumerate().map(|(i, (c0_g, c1_g))| {
            let (s0, s1) = gks.at(i);
            let r0 = c1_g.gadget_product(s0, C.get_ring());
            let r1 = c1_g.gadget_product(s1, C.get_ring());
            return (C.add_ref(&r0, &c0_g), r1);
        }).collect();
    }
}

///
/// Instantiation of BFV in power-of-two cyclotomic rings `Z[X]/(X^N + 1)` for `N`
/// a power of two.
/// 
/// For these rings, using a `DoubleRNSRing` as ciphertext ring is always the best
/// (i.e. fastest) solution.
/// 
#[derive(Debug)]
pub struct Pow2BFV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator, C: Send + Sync + HERingNegacyclicNTT<Zn> = DefaultNegacyclicNTT> {
    pub log2_N: usize,
    pub ciphertext_allocator: A,
    pub negacyclic_ntt: PhantomData<C>
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> Display for Pow2BFV<A, C> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BFV(m = 2^{})", self.log2_N + 1)
    }
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> Clone for Pow2BFV<A, C> {

    fn clone(&self) -> Self {
        Self {
            log2_N: self.log2_N,
            ciphertext_allocator: self.ciphertext_allocator.clone(),
            negacyclic_ntt: PhantomData
        }
    }
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> BFVInstantiation for Pow2BFV<A, C> {

    type CiphertextRing = ManagedDoubleRNSRingBase<Pow2CyclotomicNumberRing<C>, A>;
    type PlaintextRing = NumberRingQuotientBase<Pow2CyclotomicNumberRing<C>, Zn>;
    type PlaintextZnRing = ZnBase;

    #[instrument(skip_all)]
    fn number_ring(&self) -> Pow2CyclotomicNumberRing<C> {
        Pow2CyclotomicNumberRing::new_with(2 << self.log2_N)
    }

    #[instrument(skip_all)]
    fn create_plaintext_ring(&self, t: El<BigIntRing>) -> PlaintextRing<Self> {
        NumberRingQuotientBase::new(self.number_ring(), Zn::new(int_cast(t, ZZi64, ZZbig) as u64))
    }

    #[instrument(skip_all)]
    fn create_ciphertext_rings(&self, log2_q: Range<usize>) -> (CiphertextRing<Self>, CiphertextRing<Self>)  {
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;

        let C_rns_base = sample_primes(log2_q.start, log2_q.end, 57, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZi64, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZi64))).unwrap();
        let Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2 + 10, log2_q.end * 2 + 67, 57, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZi64, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZi64))).unwrap();

        let C_rns_base = zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZi64, ZZbig) as u64)).collect::<Vec<_>>(), ZZbig);
        let Cmul_rns_base = zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZi64, ZZbig) as u64)).collect(), ZZbig);

        let C_mul = ManagedDoubleRNSRingBase::new_with(
            number_ring,
            Cmul_rns_base,
            self.ciphertext_allocator.clone()
        );

        let dropped_indices = (0..C_mul.base_ring().len()).filter(|i| C_rns_base.as_iter().all(|Zp| Zp.get_ring() != C_mul.base_ring().at(*i).get_ring())).collect::<Vec<_>>();
        let C = RingValue::from(C_mul.get_ring().drop_rns_factor(&dropped_indices));
        assert!(C.base_ring().get_ring() == C_rns_base.get_ring());
        return (C, C_mul);
    }

    #[instrument(skip_all)]
    fn encode_plain_multiplicant(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>) -> El<CiphertextRing<Self>> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        return C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
    }

    #[instrument(skip_all)]
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>) {
        let mut scale_down = rescale_to_P::<Self>(target, C);
        return (scale_down(&ct.0), scale_down(&ct.1));
    }
}

///
/// Instantiation of BFV over odd, composite cyclotomic rings `Z[X]/(Phi_m(X))`
/// with `m = m1 * m2` and `m2, m2` odd, coprime and squarefree integers. Ciphertexts are represented
/// in double-RNS form. If single-RNS form is instead requires, use [`CompositeSingleRNSBFV`].
/// 
#[derive(Clone, Debug)]
pub struct CompositeBFV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator> {
    pub m1: usize,
    pub m2: usize,
    pub ciphertext_allocator: A
}

impl<A: Allocator + Clone + Send + Sync> Display for CompositeBFV<A> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BFV(m = {} * {})", self.m1, self.m2)
    }
}

impl<A: Allocator + Clone + Send + Sync> BFVInstantiation for CompositeBFV<A> {

    type CiphertextRing = ManagedDoubleRNSRingBase<CompositeCyclotomicNumberRing, A>;
    type PlaintextRing = NumberRingQuotientBase<CompositeCyclotomicNumberRing, Zn>;
    type PlaintextZnRing = ZnBase;

    #[instrument(skip_all)]
    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.m1, self.m2)
    }

    #[instrument(skip_all)]
    fn create_plaintext_ring(&self, t: El<BigIntRing>) -> PlaintextRing<Self> {
        NumberRingQuotientBase::new(self.number_ring(), Zn::new(int_cast(t, ZZi64, ZZbig) as u64))
    }
    
    #[instrument(skip_all)]
    fn create_ciphertext_rings(&self, log2_q: Range<usize>) -> (CiphertextRing<Self>, CiphertextRing<Self>) {
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;

        let C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZi64, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZi64))).unwrap();
        let Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZi64, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZi64))).unwrap();
        
        let C_rns_base = zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZi64, ZZbig) as u64)).collect::<Vec<_>>(), ZZbig);
        let Cmul_rns_base = zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZi64, ZZbig) as u64)).collect(), ZZbig);

        let C_mul = ManagedDoubleRNSRingBase::new_with(
            number_ring,
            Cmul_rns_base,
            self.ciphertext_allocator.clone()
        );

        let dropped_indices = (0..C_mul.base_ring().len()).filter(|i| C_rns_base.as_iter().all(|Zp| Zp.get_ring() != C_mul.base_ring().at(*i).get_ring())).collect::<Vec<_>>();
        let C = RingValue::from(C_mul.get_ring().drop_rns_factor(&dropped_indices));
        assert!(C.base_ring().get_ring() == C_rns_base.get_ring());
        return (C, C_mul);
    }

    #[instrument(skip_all)]
    fn encode_plain_multiplicant(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>) -> El<CiphertextRing<Self>> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        return double_rns_repr::<Self, _, _>(C, &C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c)))));
    }
    
    #[instrument(skip_all)]
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>) {
        let mut scale_down = rescale_to_P::<Self>(target, C);
        return (scale_down(&ct.0), scale_down(&ct.1));
    }
}

///
/// Instantiation of BFV over odd, composite cyclotomic rings `Z[X]/(Phi_m(X))`
/// with `m = m1 m2` and `m2, m2` odd coprime integers. Ciphertexts are represented
/// in single-RNS form. If double-RNS form is instead requires, use [`CompositeBFV`].
/// 
/// This takes a type `C` as last generic argument, which is the type of the convolution
/// algorithm to use to instantiate the ciphertext ring. This has a major impact on 
/// performance.
/// 
#[derive(Clone, Debug)]
pub struct CompositeSingleRNSBFV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator, C: Send + Sync + HERingConvolution<Zn> = DefaultConvolution> {
    pub m1: usize,
    pub m2: usize,
    pub ciphertext_allocator: A,
    pub convolution: PhantomData<C>
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingConvolution<Zn>> Display for CompositeSingleRNSBFV<A, C> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BFV(m = {} * {})", self.m1, self.m2)
    }
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingConvolution<Zn>> BFVInstantiation for CompositeSingleRNSBFV<A, C> {

    type CiphertextRing = SingleRNSRingBase<CompositeCyclotomicNumberRing, A, C>;
    type PlaintextRing = NumberRingQuotientBase<CompositeCyclotomicNumberRing, Zn>;
    type PlaintextZnRing = ZnBase;

    #[instrument(skip_all)]
    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.m1, self.m2)
    }

    #[instrument(skip_all)]
    fn create_plaintext_ring(&self, t: El<BigIntRing>) -> PlaintextRing<Self> {
        NumberRingQuotientBase::new(self.number_ring(), Zn::new(int_cast(t, ZZi64, ZZbig) as u64))
    }

    #[instrument(skip_all)]
    fn create_ciphertext_rings(&self, log2_q: Range<usize>) -> (CiphertextRing<Self>, CiphertextRing<Self>) {
        let number_ring = self.number_ring();
        let required_root_of_unity = 1 << ZZi64.abs_log2_ceil(&(number_ring.m() as i64 * 4)).unwrap();

        let C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZi64, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZi64))).unwrap();
        let Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZi64, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZi64))).unwrap();
        
        let max_log2_n = 1 + ZZi64.abs_log2_ceil(&((self.m1 * self.m2) as i64)).unwrap();
        let C_rns_base = C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZi64, ZZbig) as u64)).collect::<Vec<_>>();
        let Cmul_rns_base = Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZi64, ZZbig) as u64)).collect::<Vec<_>>();

        let C_convolutions = C_rns_base.iter().map(|Zp| C::new(*Zp, max_log2_n)).map(Arc::new).collect::<Vec<_>>();
        let Cmul_convolutions = Cmul_rns_base.iter().map(|Zp| match C_rns_base.iter().enumerate().filter(|(_, C_Zp)| C_Zp.get_ring() == Zp.get_ring()).next() {
            Some((i, _)) => C_convolutions.at(i).clone(),
            None => Arc::new(C::new(*Zp, max_log2_n))
        }).collect();

        let C = SingleRNSRingBase::new_with(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.clone(), ZZbig),
            self.ciphertext_allocator.clone(),
            C_convolutions
        );
        let C_mul = SingleRNSRingBase::new_with(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.clone(), ZZbig),
            self.ciphertext_allocator.clone(),
            Cmul_convolutions
        );
        return (C, C_mul);
    }
    
    #[instrument(skip_all)]
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>) {
        let mut scale_down = rescale_to_P::<Self>(target, C);
        return (scale_down(&ct.0), scale_down(&ct.1));
    }
}

///
/// Forces a ciphertext to be internally stored in small-basis representation.
/// 
/// Use in benchmarks, when you want to control which representation the inputs
/// to the benchmarked code have.
/// 
pub fn small_basis_repr<Params, NumberRing, A>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params>
    where Params: BFVInstantiation<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
        NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    (
        C.get_ring().from_small_basis_repr(C.get_ring().to_small_basis(&ct.0).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el_non_fft(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero_non_fft())), 
        C.get_ring().from_small_basis_repr(C.get_ring().to_small_basis(&ct.1).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el_non_fft(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero_non_fft())), 
    )
}

///
/// Forces a ciphertext to be internally stored in double-RNS representation.
/// 
/// Use in benchmarks, when you want to control which representation the inputs
/// to the benchmarked code have.
/// 
pub fn double_rns_repr<Params, NumberRing, A>(C: &CiphertextRing<Params>, x: &El<CiphertextRing<Params>>) -> El<CiphertextRing<Params>>
    where Params: BFVInstantiation<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
        NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    C.get_ring().from_double_rns_repr(C.get_ring().to_doublerns(x).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero()))
}

fn lift_to_Cmul<'a, Params: ?Sized + BFVInstantiation>(C: &'a CiphertextRing<Params>, C_mul: &'a CiphertextRing<Params>) -> impl use<'a, Params> + for<'b> FnMut(&'b El<CiphertextRing<Params>>) -> El<CiphertextRing<Params>> {
    let lift = AlmostExactSharedBaseConversion::new_with(
        C.base_ring().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
        Vec::new(),
        C_mul.base_ring().as_iter().skip(C.base_ring().len()).map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(),
        Global
    );
    return move |c| perform_rns_op(C_mul.get_ring(), C.get_ring(), c, &lift);
}

fn rescale_to_C<'a, Params: ?Sized + BFVInstantiation>(P: &PlaintextRing<Params>, C: &'a CiphertextRing<Params>, C_mul: &'a CiphertextRing<Params>) -> impl use<'a, Params> + for<'b> FnMut(&'b El<CiphertextRing<Params>>) -> El<CiphertextRing<Params>> {
    assert!(C.get_ring().number_ring() == C_mul.get_ring().number_ring());
    assert_eq!(C.get_ring().small_generating_set_len(), C_mul.get_ring().small_generating_set_len());

    let ZZ = P.base_ring().integer_ring();
    let result: Box<dyn 'a + for<'b> FnMut(&'b El<CiphertextRing<Params>>) -> El<CiphertextRing<Params>>> = if false && ZZ.abs_log2_ceil(P.base_ring().modulus()).unwrap() <= 50 {
        let rescale = AlmostExactRescalingConvert::new_with(
            C_mul.base_ring().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
            vec![ Zn::new(int_cast(ZZ.clone_el(P.base_ring().modulus()), ZZi64, ZZ) as u64) ], 
            (0..C.base_ring().len()).collect(),
            Global
        );
        let result = move |c: &El<CiphertextRing<Params>>| perform_rns_op(C.get_ring(), C_mul.get_ring(), &*c, &rescale);
        Box::new(result)
    } else {
        let log2_extended_modulus = ZZ.abs_log2_ceil(P.base_ring().modulus()).unwrap() + ZZbig.abs_log2_ceil(C_mul.base_ring().modulus()).unwrap();
        let extended_rns_base = extend_sampled_primes(
            &C_mul.base_ring().as_iter().map(|ring| int_cast(*ring.modulus() as i64, ZZbig, ZZi64)).collect::<Vec<_>>(),
            log2_extended_modulus + 10,
            log2_extended_modulus + 67,
            57,
            |bound| prev_prime(ZZbig, bound)
        ).unwrap().into_iter().map(|modulus| Zn::new(int_cast(modulus, ZZi64, ZZbig) as u64)).collect::<Vec<_>>();
        let to_extended = AlmostExactSharedBaseConversion::new_with(
            extended_rns_base[..C_mul.base_ring().len()].iter().cloned().collect::<Vec<_>>(),
            Vec::new(),
            extended_rns_base[C_mul.base_ring().len()..].iter().cloned().collect::<Vec<_>>(),
            Global
        );
        let rescale = AlmostExactRescalingConvert::new_with(
            to_extended.output_rings().to_owned(), 
            Vec::new(), 
            (0..C.base_ring().len()).collect(),
            Global
        );
        let t_mod_extended = extended_rns_base.iter().map(|ring| ring.coerce(ZZ, ZZ.clone_el(P.base_ring().modulus()))).collect::<Vec<_>>();
        let mut tmp = OwnedMatrix::zero(C_mul.base_ring().len(), C_mul.get_ring().small_generating_set_len(), C_mul.base_ring().at(0));
        let mut tmp_ex = OwnedMatrix::zero(extended_rns_base.len(), C_mul.get_ring().small_generating_set_len(), C_mul.base_ring().at(0));
        let result = move |c: &El<CiphertextRing<Params>>| {
            C_mul.get_ring().as_representation_wrt_small_generating_set(c, tmp.data_mut());
            to_extended.apply(tmp.data(), tmp_ex.data_mut());
            for (ring, (row, factor)) in extended_rns_base.iter().zip(tmp_ex.data_mut().row_iter().zip(t_mod_extended.iter())) {
                for x in row {
                    ring.mul_assign_ref(x, factor);
                }
            }
            let mut tmp = tmp.data_mut().restrict_rows(0..C.base_ring().len());
            rescale.apply(tmp_ex.data(), tmp.reborrow());
            return C.get_ring().from_representation_wrt_small_generating_set(tmp.as_const());
        };
        Box::new(result)
    };
    return result;
}

fn rescale_to_P<'a, Params: ?Sized + BFVInstantiation>(P: &'a PlaintextRing<Params>, C: &'a CiphertextRing<Params>) -> impl use<'a, Params> + for<'b> FnMut(&'b El<CiphertextRing<Params>>) -> El<PlaintextRing<Params>>
    where <<PlaintextRing<Params> as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = ZnBase>
{
    assert_eq!(P.rank(), C.rank());
    assert_eq!(P.m(), C.m());

    let mod_switch = AlmostExactRescaling::new_with(
        C.base_ring().as_iter().map(|Zp| *Zp).collect(),
        vec![RingValue::from(*P.base_ring().get_ring())],
        (0..C.base_ring().len()).collect(),
        Global
    );
    let mut tmp = OwnedMatrix::zero(C.base_ring().len(), C.rank(), C.base_ring().at(0));
    let mut tmp_res = OwnedMatrix::zero(1, P.rank(), P.base_ring());
    return move |c| {
        for (mut col, el) in tmp.data_mut().col_iter().zip(C.wrt_canonical_basis(&*c).iter()) {
            let rns_parts = C.base_ring().get_congruence(&el);
            for (i, value) in rns_parts.as_iter().enumerate() {
                *col.at_mut(i) = C.base_ring().at(i).clone_el(value);
            }
        }
        mod_switch.apply(tmp.data(), tmp_res.data_mut());
        return P.from_canonical_basis(tmp_res.data().row_at(0).iter().cloned());
    };
}

impl<NumberRing: HENumberRing> PlaintextCircuit<NumberRingQuotientBase<NumberRing, Zn>> {

    #[instrument(skip_all)]
    pub fn evaluate_bfv<Params>(&self, 
        P: &PlaintextRing<Params>, 
        C: &CiphertextRing<Params>, 
        C_mul: Option<&CiphertextRing<Params>>,
        inputs: &[Ciphertext<Params>], 
        rk: Option<&RelinKey<Params>>, 
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)], 
        key_switches: &mut usize
    ) -> Vec<Ciphertext<Params>> 
        where Params: BFVInstantiation<PlaintextRing = NumberRingQuotientBase<NumberRing, Zn>>
    {
        assert!(!self.has_multiplication_gates() || C_mul.is_some());
        assert_eq!(C_mul.is_some(), rk.is_some());
        let galois_group = C.galois_group();
        let key_switches = RefCell::new(key_switches);
        return self.evaluate_generic(
            inputs,
            DefaultCircuitEvaluator::new(
                |x| match x {
                    Coefficient::Zero => Params::transparent_zero(C),
                    x => Params::hom_add_plain(P, C, &x.clone(P).to_ring_el(P), Params::transparent_zero(C))
                },
                |dst, x, ct| Params::hom_add(C, dst, &Params::hom_mul_plain(P, C, &x.clone(P).to_ring_el(P), Params::clone_ct(C, ct))),
            ).with_mul(|lhs, rhs| {
                **key_switches.borrow_mut() += 1;
                Params::hom_mul(P, C, C_mul.unwrap(), lhs, rhs, rk.unwrap())
            }).with_square(|x| {
                **key_switches.borrow_mut() += 1;
                Params::hom_square(P, C, C_mul.unwrap(), x, rk.unwrap())
            }).with_gal(|x, gs| if gs.len() == 1 {
                **key_switches.borrow_mut() += 1;
                vec![Params::hom_galois(C, x, gs[0], &gks.iter().filter(|(g, _)| galois_group.eq_el(*g, gs[0])).next().unwrap().1)]
            } else {
                **key_switches.borrow_mut() += gs.iter().filter(|g| !galois_group.is_identity(**g)).count();
                Params::hom_galois_many(C, x, gs, gs.as_fn().map_fn(|expected_g| if let Some(gk) = gks.iter().filter(|(g, _)| galois_group.eq_el(*g, *expected_g)).next() {
                    &gk.1
                } else {
                    panic!("Galois key for {} not found", galois_group.underlying_ring().format(&galois_group.to_ring_el(*expected_g)))
                }))
            })
        );
    }
}

impl PlaintextCircuit<StaticRingBase<i64>> {

    #[instrument(skip_all)]
    pub fn evaluate_bfv<Params>(&self, 
        P: &PlaintextRing<Params>, 
        C: &CiphertextRing<Params>, 
        C_mul: Option<&CiphertextRing<Params>>,
        inputs: &[Ciphertext<Params>], 
        rk: Option<&RelinKey<Params>>, 
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)], 
        key_switches: &mut usize
    ) -> Vec<Ciphertext<Params>> 
        where Params: BFVInstantiation
    {
        assert!(!self.has_multiplication_gates() || C_mul.is_some());
        assert_eq!(C_mul.is_some(), rk.is_some());
        const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
        let galois_group = C.galois_group();
        let key_switches = RefCell::new(key_switches);
        return self.evaluate_generic(
            inputs,
            DefaultCircuitEvaluator::new(
                |x| match x {
                    Coefficient::Zero => Params::transparent_zero(C),
                    x => Params::hom_add_plain(P, C, &P.int_hom().map(x.clone(ZZ).to_ring_el(ZZ) as i32), Params::transparent_zero(C))
                },
                |dst, x, ct| Params::hom_add(C, dst, &Params::hom_mul_plain_i64(P, C, x.to_ring_el(ZZ), Params::clone_ct(C, ct))),
            ).with_mul(|lhs, rhs| {
                **key_switches.borrow_mut() += 1;
                Params::hom_mul(P, C, C_mul.unwrap(), lhs, rhs, rk.unwrap())
            }).with_square(|x| {
                **key_switches.borrow_mut() += 1;
                Params::hom_square(P, C, C_mul.unwrap(), x, rk.unwrap())
            }).with_gal(|x, gs| if gs.len() == 1 {
                **key_switches.borrow_mut() += 1;
                vec![Params::hom_galois(C, x, gs[0], &gks.iter().filter(|(g, _)| galois_group.eq_el(*g, gs[0])).next().unwrap().1)]
            } else {
                **key_switches.borrow_mut() += gs.iter().filter(|g| !galois_group.is_identity(**g)).count();
                Params::hom_galois_many(C, x, gs, gs.as_fn().map_fn(|expected_g| &gks.iter().filter(|(g, _)| galois_group.eq_el(*g, *expected_g)).next().unwrap().1))
            })
        );
    }
}

#[cfg(test)]
use tracing_subscriber::prelude::*;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use std::fmt::Debug;
#[cfg(test)]
use crate::{log_time, get_default_ciphertext_allocator};
#[cfg(test)]

#[test]
fn test_pow2_bfv_enc_dec() {
    let mut rng = rand::rng();
    
    let instantiation = Pow2BFV {
        log2_N: 7,
        ciphertext_allocator: get_default_ciphertext_allocator(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    
    let P = instantiation.create_plaintext_ring(int_cast(257, ZZbig, ZZi64));
    let (C, _Cmul) = instantiation.create_ciphertext_rings(500..520);
    let sk = Pow2BFV::gen_sk(&C, &mut rng, None);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BFV::enc_sym(&P, &C, &mut rng, &input, &sk, 3.2);
    let output = Pow2BFV::dec(&P, &C, Pow2BFV::clone_ct(&C, &ctxt), &sk);
    assert_el_eq!(&P, input, output);
}

#[test]
fn test_pow2_bfv_hom_galois() {
    let mut rng = rand::rng();
    
    let instantiation = Pow2BFV {
        log2_N: 7,
        ciphertext_allocator: get_default_ciphertext_allocator(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };

    let P = instantiation.create_plaintext_ring(int_cast(3, ZZbig, ZZi64));
    let (C, _C_mul) = instantiation.create_ciphertext_rings(500..520);
    let sk = Pow2BFV::gen_sk(&C, &mut rng, None);
    let gk = Pow2BFV::gen_gk(&C, &mut rng, &sk, P.galois_group().from_representative(3), &RNSGadgetVectorDigitIndices::select_digits(3, C.base_ring().len()), 3.2);
    
    let input = P.canonical_gen();
    let ctxt = Pow2BFV::enc_sym(&P, &C, &mut rng, &input, &sk, 3.2);
    let result_ctxt = Pow2BFV::hom_galois(&C, ctxt, P.galois_group().from_representative(3), &gk);
    let result = Pow2BFV::dec(&P, &C, result_ctxt, &sk);

    assert_el_eq!(&P, &P.pow(P.canonical_gen(), 3), &result);
}

#[test]
fn test_pow2_bfv_mul() {
    let mut rng = rand::rng();
    
    let instantiation = Pow2BFV {
        log2_N: 10,
        ciphertext_allocator: get_default_ciphertext_allocator(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };

    let P = instantiation.create_plaintext_ring(int_cast(257, ZZbig, ZZi64));
    let (C, C_mul) = instantiation.create_ciphertext_rings(500..520);
    let sk = Pow2BFV::gen_sk(&C, &mut rng, None);
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, &RNSGadgetVectorDigitIndices::select_digits(3, C.base_ring().len()), 3.2);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BFV::enc_sym(&P, &C, &mut rng, &input, &sk, 3.2);
    let result_ctxt = Pow2BFV::hom_mul(&P, &C, &C_mul, Pow2BFV::clone_ct(&C, &ctxt), Pow2BFV::clone_ct(&C, &ctxt), &rk);
    let result = Pow2BFV::dec(&P, &C, result_ctxt, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &result);
}

#[test]
fn test_composite_bfv_mul() {
    let mut rng = rand::rng();
    
    let instantiation = CompositeBFV {
        m1: 17,
        m2: 97,
        ciphertext_allocator: get_default_ciphertext_allocator()
    };

    let P = instantiation.create_plaintext_ring(int_cast(8, ZZbig, ZZi64));
    let (C, C_mul) = instantiation.create_ciphertext_rings(500..520);
    let sk = CompositeBFV::gen_sk(&C, &mut rng, None);
    let rk = CompositeBFV::gen_rk(&C, &mut rng, &sk, &RNSGadgetVectorDigitIndices::select_digits(3, C.base_ring().len()), 3.2);

    let input = P.int_hom().map(2);
    let ctxt = CompositeBFV::enc_sym(&P, &C, &mut rng, &input, &sk, 3.2);
    let result_ctxt = CompositeBFV::hom_mul(&P, &C, &C_mul, CompositeBFV::clone_ct(&C, &ctxt), CompositeBFV::clone_ct(&C, &ctxt), &rk);
    let result = CompositeBFV::dec(&P, &C, result_ctxt, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &result);
}

#[test]
fn test_composite_bfv_hom_galois() {
    let mut rng = rand::rng();
    
    let instantiation = CompositeSingleRNSBFV {
        m1: 7,
        m2: 11,
        ciphertext_allocator: get_default_ciphertext_allocator(),
        convolution: PhantomData::<DefaultConvolution>
    };

    let P = instantiation.create_plaintext_ring(int_cast(3, ZZbig, ZZi64));
    let (C, _C_mul) = instantiation.create_ciphertext_rings(500..520);
    let sk = CompositeSingleRNSBFV::gen_sk(&C, &mut rng, None);
    let gk = CompositeSingleRNSBFV::gen_gk(&C, &mut rng, &sk, P.galois_group().from_representative(3), &RNSGadgetVectorDigitIndices::select_digits(3, C.base_ring().len()), 3.2);
    
    let input = P.canonical_gen();
    let ctxt = CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &input, &sk, 3.2);
    let result_ctxt = CompositeSingleRNSBFV::hom_galois(&C, ctxt, P.galois_group().from_representative(3), &gk);
    let result = CompositeSingleRNSBFV::dec(&P, &C, result_ctxt, &sk);

    assert_el_eq!(&P, &P.pow(P.canonical_gen(), 3), &result);
}

#[test]
fn test_single_rns_composite_bfv_mul() {
    let mut rng = rand::rng();
    
    let instantiation = CompositeSingleRNSBFV {
        m1: 7,
        m2: 11,
        ciphertext_allocator: get_default_ciphertext_allocator(),
        convolution: PhantomData::<DefaultConvolution>
    };
    
    let P = instantiation.create_plaintext_ring(int_cast(3, ZZbig, ZZi64));
    let (C, C_mul) = instantiation.create_ciphertext_rings(500..520);  
    let sk = CompositeSingleRNSBFV::gen_sk(&C, &mut rng, None);
    let rk = CompositeSingleRNSBFV::gen_rk(&C, &mut rng, &sk, &RNSGadgetVectorDigitIndices::select_digits(3, C.base_ring().len()), 3.2);

    let input = P.int_hom().map(2);
    let ctxt = CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &input, &sk, 3.2);
    let result_ctxt = CompositeSingleRNSBFV::hom_mul(&P, &C, &C_mul, CompositeSingleRNSBFV::clone_ct(&C, &ctxt), CompositeSingleRNSBFV::clone_ct(&C, &ctxt), &rk);
    let result = CompositeSingleRNSBFV::dec(&P, &C, result_ctxt, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &result);
}

#[test]
#[ignore]
fn measure_time_pow2_bfv_basic_ops() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();

    let mut rng = rand::rng();
    
    let params = Pow2BFV {
        log2_N: 15,
        ciphertext_allocator: get_default_ciphertext_allocator(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    
    let P = log_time::<_, _, true, _>("CreatePtxtRing", |[]|
        params.create_plaintext_ring(int_cast(257, ZZbig, ZZi64))
    );
    let (C, C_mul) = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_ciphertext_rings(790..800)
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        Pow2BFV::gen_sk(&C, &mut rng, None)
    );

    let m = P.int_hom().map(2);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk, 3.2)
    );

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        Pow2BFV::hom_add_plain(&P, &C, &m, Pow2BFV::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        Pow2BFV::hom_add(&C, Pow2BFV::clone_ct(&C, &ct), &ct)
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        Pow2BFV::hom_mul_plain(&P, &C, &m, Pow2BFV::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        Pow2BFV::gen_rk(&C, &mut rng, &sk, &RNSGadgetVectorDigitIndices::select_digits(3, C.base_ring().len()), 3.2)
    );
    let ct2 = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk, 3.2);
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        Pow2BFV::hom_mul(&P, &C, &C_mul, ct, ct2, &rk)
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));
}

#[test]
#[ignore]
fn measure_time_double_rns_composite_bfv_basic_ops() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();

    let mut rng = rand::rng();
    
    let params = CompositeBFV {
        m1: 127,
        m2: 337,
        ciphertext_allocator: get_default_ciphertext_allocator(),
    };
    
    let P = log_time::<_, _, true, _>("CreatePtxtRing", |[]|
        params.create_plaintext_ring(int_cast(4, ZZbig, ZZi64))
    );
    let (C, C_mul) = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_ciphertext_rings(1090..1100)
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        CompositeBFV::gen_sk(&C, &mut rng, None)
    );
    
    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk, 3.2)
    );
    assert_el_eq!(&P, &P.int_hom().map(3), &CompositeBFV::dec(&P, &C, CompositeBFV::clone_ct(&C, &ct), &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        CompositeBFV::hom_add(&C, CompositeBFV::clone_ct(&C, &ct), &ct)
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        CompositeBFV::hom_add_plain(&P, &C, &m, CompositeBFV::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        CompositeBFV::hom_mul_plain(&P, &C, &m, CompositeBFV::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBFV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        CompositeBFV::gen_rk(&C, &mut rng, &sk, &RNSGadgetVectorDigitIndices::select_digits(3, C.base_ring().len()), 3.2)
    );
    let ct2 = CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk, 3.2);
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        CompositeBFV::hom_mul(&P, &C, &C_mul, ct, ct2, &rk)
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBFV::dec(&P, &C, res, &sk));
}

#[test]
#[ignore]
fn measure_time_single_rns_composite_bfv_basic_ops() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();

    let mut rng = rand::rng();
    
    let params = CompositeSingleRNSBFV {
        m1: 127,
        m2: 337,
        ciphertext_allocator: get_default_ciphertext_allocator(),
        convolution: PhantomData::<DefaultConvolution>
    };

    let P = log_time::<_, _, true, _>("CreatePtxtRing", |[]|
        params.create_plaintext_ring(int_cast(4, ZZbig, ZZi64))
    );
    let (C, C_mul) = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_ciphertext_rings(1090..1100)
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        CompositeSingleRNSBFV::gen_sk(&C, &mut rng, None)
    );

    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk, 3.2)
    );

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        CompositeSingleRNSBFV::hom_add_plain(&P, &C, &m, CompositeSingleRNSBFV::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        CompositeSingleRNSBFV::hom_add(&C, CompositeSingleRNSBFV::clone_ct(&C, &ct), &ct)
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        CompositeSingleRNSBFV::hom_mul_plain(&P, &C, &m, CompositeSingleRNSBFV::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        CompositeSingleRNSBFV::gen_rk(&C, &mut rng, &sk, &RNSGadgetVectorDigitIndices::select_digits(3, C.base_ring().len()), 3.2)
    );
    let ct2 = CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk, 3.2);
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        CompositeSingleRNSBFV::hom_mul(&P, &C, &C_mul, ct, ct2, &rk)
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));
}