use std::ops::Range;

use feanor_math::{homomorphism::Homomorphism, integer::BigIntRing, ring::{El, RingExtension, RingStore, RingValue}, rings::{extension::FreeAlgebra, finite::FiniteRing, matrix::MatrixRing}};
use rand::{CryptoRng, Rng};
use tracing::instrument;

use crate::{NiceZn, bfv::SecretKeyDistribution, ciphertext_ring::{BGFVCiphertextRing, MatrixBGFVCiphertextRing}, gadget_product::{RNSGadgetProductRhsOperand, digits::RNSGadgetVectorDigitIndices}, matrix_ring::ModuleLWEMatrixRing, number_ring::{AbstractNumberRing, NumberRingQuotient}};

pub mod eval;

pub type NumberRing<Params: MatrixBFVInstantiation> = <<Params as MatrixBFVInstantiation>::CiphertextMatrixRing as ModuleLWEMatrixRing>::NumberRing;
pub type PlaintextScalarRing<Params: MatrixBFVInstantiation> = <<Params as MatrixBFVInstantiation>::PlaintextMatrixRing as ModuleLWEMatrixRing>::ScalarRing;
pub type PlaintextMatrixRing<Params: MatrixBFVInstantiation> = RingValue<<Params as MatrixBFVInstantiation>::PlaintextMatrixRing>;
pub type SecretKey<Params: MatrixBFVInstantiation> = El<CiphertextMatrixRing<Params>>;
pub type KeySwitchKey<Params: MatrixBFVInstantiation> = (GadgetProductOperand<Params>, GadgetProductOperand<Params>);
pub type RelinKey<Params: MatrixBFVInstantiation> = KeySwitchKey<Params>;
pub type CiphertextScalarRing<Params: MatrixBFVInstantiation> = RingValue<Params::CiphertextScalarRing>;
pub type CiphertextMatrixRing<Params: MatrixBFVInstantiation> = RingValue<Params::CiphertextMatrixRing>;
pub type Ciphertext<Params: MatrixBFVInstantiation> = (El<CiphertextMatrixRing<Params>>, El<CiphertextMatrixRing<Params>>);
pub type GadgetProductOperand<Params: MatrixBFVInstantiation> = RNSGadgetProductRhsOperand<Params::CiphertextMatrixRing>;

pub trait MatrixBFVInstantiation {

    type NumberRing: AbstractNumberRing;

    ///
    /// Type of the ciphertext scalar ring `R/qR`.
    /// 
    type CiphertextScalarRing: BGFVCiphertextRing<NumberRing = Self::NumberRing> + FiniteRing;

    ///
    /// Ciphertext matrix ring over Z/qZ
    ///
    type CiphertextMatrixRing: MatrixBGFVCiphertextRing<ScalarRing = Self::CiphertextScalarRing, NumberRing = Self::NumberRing> + FiniteRing;

    ///
    /// The base ring Z/tZ of the plaintext matrix ring's scalar ring (polynomial ring)
    ///
    type PlaintextZnRing: NiceZn;

    ///
    /// The plaintext matrix ring's scalar ring (polynomial ring)
    ///
    type PlaintextScalarRing: NumberRingQuotient<BaseRing = RingValue<Self::PlaintextZnRing>, NumberRing = Self::NumberRing>;

    ///
    /// Plaintext matrix ring over Z/tZ
    ///
    type PlaintextMatrixRing: ModuleLWEMatrixRing<ScalarRing = Self::PlaintextScalarRing> + FiniteRing;

    ///
    /// The number ring `R` we work in, i.e. the ciphertext scalar ring is `R/qR` and
    /// the plaintext matrix ring is `R/tR`.
    /// 
    fn number_ring(&self) -> &NumberRing<Self>;

    ///
    /// Creates the ciphertext scalar rings `R/qR` and `R/qq'R`
    /// that are necessary for homomorphic multiplication.
    /// 
    fn create_ciphertext_scalar_rings(&self, log2_q: Range<usize>) -> (CiphertextScalarRing<Self>, CiphertextScalarRing<Self>);

    ///
    /// Creates the ciphertext matrix rings over `R/qR` and `R/qq'R`.
    /// 
    fn create_ciphertext_rings(&self, log2_q: Range<usize>) -> (CiphertextMatrixRing<Self>, CiphertextMatrixRing<Self>);

    ///
    /// Creates the plaintext scalar ring `R/tR` for the given modulus `t`.
    /// 
    fn create_plaintext_scalar_ring(&self, t: El<BigIntRing>) -> PlaintextScalarRing<Self>;

    ///
    /// Creates the plaintext matrix ring over `R/tR`.
    /// 
    fn create_plaintext_ring(&self, t: El<BigIntRing>) -> PlaintextMatrixRing<Self>;

    ///
    /// Generates a secret key, which is either a sparse ternary element of the
    /// ciphertext matrix ring (with hamming weight `hwt`), or a uniform ternary element
    /// of the ciphertext matrix ring (if `hwt == UniformTernary`).
    /// 
    #[instrument(skip_all)]
    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextMatrixRing<Self>, mut rng: R, hwt:SecretKeyDistribution) -> SecretKey<Self> {
        match hwt {
            SecretKeyDistribution::SparseWithHwt(_) => {
                // later: the hamming weight should be meant column by column to match security
                panic!("SparseWithHwt is not (yet) supported for matrix BFV");
            },
            SecretKeyDistribution::UniformTernary => {
                let S = C.get_ring().scalar_ring();
                let d = C.get_ring().dimension();
                C.get_ring().from_scalar_entries((0..d * d).map(|_| {
                    S.from_canonical_basis((0..S.rank()).map(|_| {
                        S.base_ring().int_hom().map((rng.next_u32() % 3) as i32 - 1)
                    }))
                }))
            }
            SecretKeyDistribution::Zero => C.zero(),
            SecretKeyDistribution::Custom(_) => panic!("if you use SecretKeyDistribution::Custom(_), you must generate the secret key yourself")
        }
    }

    #[instrument(skip_all)]
    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextMatrixRing<Self>, mut rng: R, sk: &SecretKey<Self>, noise_sigma: f64) -> Ciphertext<Self> {
        unimplemented!("enc_sym_zero");
    }

    #[instrument(skip_all)]
    fn transparent_zero(C: &CiphertextMatrixRing<Self>) -> Ciphertext<Self> {
        (C.zero(), C.zero())
    }

    #[instrument(skip_all)]
    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, rng: R, m: &El<PlaintextMatrixRing<Self>>, sk: &SecretKey<Self>, noise_sigma: f64) -> Ciphertext<Self> {
        unimplemented!("enc_sym");
    }

    #[instrument(skip_all)]
    fn enc_sk(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>) -> Ciphertext<Self> { 
        unimplemented!("enc_sk");
    }

    #[instrument(skip_all)]
    fn remove_noise(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, c: &mut El<CiphertextMatrixRing<Self>>) -> El<PlaintextMatrixRing<Self>> {
        unimplemented!("remove_noise");
    }

    #[instrument(skip_all)]
    fn dec(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, ct: Ciphertext<Self>, sk: &SecretKey<Self>) -> El<PlaintextMatrixRing<Self>> {
        unimplemented!("dec");
    }

    #[instrument(skip_all)]
    fn dec_println(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        unimplemented!("dec_println");
    }

    #[instrument(skip_all)]
    fn hom_add(C: &CiphertextMatrixRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_add");
    }

    #[instrument(skip_all)]
    fn hom_sub(C: &CiphertextMatrixRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_sub");
    }

    #[instrument(skip_all)]
    fn clone_ct(C: &CiphertextMatrixRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
        (C.clone_el(&ct.0), C.clone_el(&ct.1))
    }

    #[instrument(skip_all)]
    fn hom_add_plain(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<PlaintextMatrixRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_add_plain");
    }

    #[instrument(skip_all)]
    fn hom_mul_plain(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<PlaintextMatrixRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_mul_plain");
        //Self::hom_mul_plain_encoded(P, C, &Self::encode_plain_multiplicant(P, C, m), ct)
    }

    #[instrument(skip_all)]
    fn encode_plain_multiplicant(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<PlaintextMatrixRing<Self>>) -> El<CiphertextMatrixRing<Self>> {
        unimplemented!("encode_plain_multiplicant");
        // let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        // return C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
    }

    #[instrument(skip_all)]
    fn hom_mul_plain_encoded(_P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<CiphertextMatrixRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_mul_plain_encoded");
        // (C.mul_ref_snd(ct.0, m), C.mul_ref_snd(ct.1, m))
    }

    #[instrument(skip_all)]
    fn hom_mul_plain_int(_P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<BigIntRing>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_mul_plain_int");
        // let hom = C.inclusion().compose(C.base_ring().can_hom(&ZZbig).unwrap());
        // (hom.mul_ref_snd_map(ct.0, m), hom.mul_ref_snd_map(ct.1, m))
    }

    #[instrument(skip_all)]
    fn hom_fma_plain_int(_P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, dst: Ciphertext<Self>, lhs: &El<BigIntRing>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_fma_plain_int");
        // let hom = C.inclusion().compose(C.base_ring().can_hom(&ZZbig).unwrap());
        // (hom.fma_map(&rhs.0, lhs, dst.0), hom.fma_map(&rhs.1, lhs, dst.1))
    }

    #[instrument(skip_all)]
    fn noise_budget(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) -> usize {
        unimplemented!("noise_budget");
    }

    #[instrument(skip_all)]
    fn gen_rk<R: Rng + CryptoRng>(C: &CiphertextMatrixRing<Self>, rng: R, sk: &SecretKey<Self>, digits: &RNSGadgetVectorDigitIndices, noise_sigma: f64) -> RelinKey<Self> {
        unimplemented!("gen_rk");
        // Self::gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk, digits, noise_sigma)
    }

    #[instrument(skip_all)]
    fn hom_mul(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, C_mul: &CiphertextMatrixRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_mul");
    }

    #[instrument(skip_all)]
    fn hom_square(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, C_mul: &CiphertextMatrixRing<Self>, val: Ciphertext<Self>, rk: &RelinKey<Self>) -> Ciphertext<Self> {
        unimplemented!("hom_square");
    }

    #[instrument(skip_all)]
    fn gen_switch_key<R: Rng + CryptoRng>(C: &CiphertextMatrixRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: &RNSGadgetVectorDigitIndices, noise_sigma: f64) -> KeySwitchKey<Self> {
        unimplemented!("gen_switch_key");
    }

    #[instrument(skip_all)]
    fn key_switch(C: &CiphertextMatrixRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<Self>) -> Ciphertext<Self> {
        unimplemented!("key_switch");
    }

    #[instrument(skip_all)]
    fn mod_switch_to_plaintext(target: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, ct: Ciphertext<Self>) -> (El<PlaintextMatrixRing<Self>>, El<PlaintextMatrixRing<Self>>) {
        unimplemented!("mod_switch_to_plaintext");
    }

    #[instrument(skip_all)]
    fn mod_switch_ct(_P: &PlaintextMatrixRing<Self>, Cnew: &CiphertextMatrixRing<Self>, Cold: &CiphertextMatrixRing<Self>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        unimplemented!("mod_switch_ct");
    }

    #[instrument(skip_all)]
    fn mod_switch_sk(_P: &PlaintextMatrixRing<Self>, Cnew: &CiphertextMatrixRing<Self>, Cold: &CiphertextMatrixRing<Self>, sk: &SecretKey<Self>) -> SecretKey<Self> {
        unimplemented!("mod_switch_sk");
    }

    fn lift_to_Cmul<'a>(C: &'a CiphertextMatrixRing<Self>, C_mul: &'a CiphertextMatrixRing<Self>) -> Box<dyn 'a + for<'b> FnMut(&'b El<CiphertextMatrixRing<Self>>) -> El<CiphertextMatrixRing<Self>>> {
        unimplemented!("lift_to_Cmul");
    }

    fn rescale_to_C<'a>(P: &PlaintextMatrixRing<Self>, C: &'a CiphertextMatrixRing<Self>, C_mul: &'a CiphertextMatrixRing<Self>) -> Box<dyn 'a + for<'b> FnMut(&'b El<CiphertextMatrixRing<Self>>) -> El<CiphertextMatrixRing<Self>>> {
        unimplemented!("rescale_to_C");
    }
}