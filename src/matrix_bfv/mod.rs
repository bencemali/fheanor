use std::ops::Range;

use feanor_math::{integer::{int_cast, BigIntRing}, ring::{El, RingExtension, RingExtensionStore, RingStore, RingValue}, rings::{finite::FiniteRing, matrix::MatrixRing, zn::zn_64::Zn}};
use rand::{CryptoRng, Rng};
use tracing::instrument;

use crate::{bfv::SecretKeyDistribution, ciphertext_ring::MatrixBGFVCiphertextRing, gadget_product::{digits::RNSGadgetVectorDigitIndices, RNSGadgetProductRhsOperand}, matrix_ring::LWEMatrixRing, NiceZn, ZZbig};

pub type PlaintextMatrixRing<Params: MatrixBrakerskiInstantiation> = RingValue<<Params as MatrixBrakerskiInstantiation>::PlaintextMatrixRing>;
pub type SecretKey<Params: MatrixBrakerskiInstantiation> = El<CiphertextMatrixRing<Params>>;
pub type KeySwitchKey<Params: MatrixBrakerskiInstantiation> = (GadgetProductOperand<Params>, GadgetProductOperand<Params>);
pub type RelinKey<Params: MatrixBrakerskiInstantiation> = KeySwitchKey<Params>;
pub type CiphertextMatrixRing<Params: MatrixBrakerskiInstantiation> = RingValue<Params::CiphertextMatrixRing>;
pub type Ciphertext<Params: MatrixBrakerskiInstantiation> = (El<CiphertextMatrixRing<Params>>, El<CiphertextMatrixRing<Params>>);
pub type GadgetProductOperand<Params: MatrixBrakerskiInstantiation> = RNSGadgetProductRhsOperand<Params::CiphertextMatrixRing>;

pub trait MatrixBrakerskiInstantiation {

    ///
    /// The base ring Z/qZ of the ciphertext matrix ring
    ///
    type CiphertextScalarRing: NiceZn;

    ///
    /// Ciphertext matrix ring over Z/qZ
    ///
    type CiphertextMatrixRing: MatrixBGFVCiphertextRing + FiniteRing;

    ///
    /// The base ring Z/tZ of the plaintext matrix ring
    ///
    type PlaintextScalarRing: NiceZn;

    ///
    /// Plaintext matrix ring over Z/tZ
    ///
    type PlaintextMatrixRing: LWEMatrixRing + FiniteRing;

    fn create_ciphertext_ring(&self, log2_q: Range<usize>) -> CiphertextMatrixRing<Self>;

    fn create_plaintext_ring(&self, t: El<BigIntRing>) -> PlaintextMatrixRing<Self>;

    #[instrument(skip_all)]
    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextMatrixRing<Self>, mut rng: R, hwt:SecretKeyDistribution) -> SecretKey<Self> {
        unimplemented!("gen_sk");
    }

    #[instrument(skip_all)]
    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextMatrixRing<Self>, mut rng: R, sk: &SecretKey<Self>, noise_sigma: f64) -> Ciphertext<Self> {
        unimplemented!("enc_sym_zero");
    }

    #[instrument(skip_all)]
    fn transparent_zero(C: &CiphertextMatrixRing<Self>) -> Ciphertext<Self> {
        (C.zero(), C.zero())
    }

    // #[instrument(skip_all)]
    // fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, rng: R, m: &El<PlaintextMatrixRing<Self>>, sk: &SecretKey<Self>, noise_sigma: f64) -> Ciphertext<Self> {
    //     Self::hom_add_plain(P, C, m, Self::enc_sym_zero(C, rng, sk, noise_sigma))
    // }

    // #[instrument(skip_all)]
    // fn enc_sk(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>) -> Ciphertext<Self> { }

    #[instrument(skip_all)]
    fn remove_noise(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, c: &mut El<CiphertextMatrixRing<Self>>) -> El<PlaintextMatrixRing<Self>> {
        let matrix_ring = RingValue::<Self::CiphertextMatrixRing>::into(C.clone());
        let data = matrix_ring.to_elements_mut(c);
        for i in 0..matrix_ring.dimension() {
            for j in 0..matrix_ring.dimension() {
                let entry = data.at_mut(i, j);
                let lifted = C.base_ring().smallest_positive_lift(entry);
                let Delta = ZZbig.rounded_div(
                    ZZbig.clone_el(C.base_ring().modulus()),
                    &int_cast(ZZ.clone_el(P.base_ring().modulus()), ZZbig, ZZ)
                );
                let modulo = P.base_ring().can_hom(&ZZbig).unwrap();
                *entry = modulo.map(ZZbig.rounded_div(lifted, &Delta));
            }
        }
        return data.make_owned();
    }

    // #[instrument(skip_all)]
    // fn dec(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, ct: Ciphertext<Self>, sk: &SecretKey<Self>) -> El<PlaintextMatrixRing<Self>> {
    //     unimplemented!("dec");
    // }

    // #[instrument(skip_all)]
    // fn dec_println(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
    //     unimplemented!("dec_println");
    // }

    // #[instrument(skip_all)]
    // fn hom_add(C: &CiphertextMatrixRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_add");
    // }

    // #[instrument(skip_all)]
    // fn hom_sub(C: &CiphertextMatrixRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_sub");
    // }

    // #[instrument(skip_all)]
    // fn clone_ct(C: &CiphertextMatrixRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
    //     (C.clone_el(&ct.0), C.clone_el(&ct.1))
    // }

    // #[instrument(skip_all)]
    // fn hom_add_plain(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<PlaintextMatrixRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_add_plain");
    // }

    // #[instrument(skip_all)]
    // fn hom_mul_plain(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<PlaintextMatrixRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_mul_plain");
    //     //Self::hom_mul_plain_encoded(P, C, &Self::encode_plain_multiplicant(P, C, m), ct)
    // }

    // #[instrument(skip_all)]
    // fn encode_plain_multiplicant(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<PlaintextMatrixRing<Self>>) -> El<CiphertextMatrixRing<Self>> {
    //     unimplemented!("encode_plain_multiplicant");
    //     // let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
    //     // return C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
    // }

    // #[instrument(skip_all)]
    // fn hom_mul_plain_encoded(_P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<CiphertextMatrixRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_mul_plain_encoded");
    //     // (C.mul_ref_snd(ct.0, m), C.mul_ref_snd(ct.1, m))
    // }

    // #[instrument(skip_all)]
    // fn hom_mul_plain_int(_P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, m: &El<BigIntRing>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_mul_plain_int");
    //     // let hom = C.inclusion().compose(C.base_ring().can_hom(&ZZbig).unwrap());
    //     // (hom.mul_ref_snd_map(ct.0, m), hom.mul_ref_snd_map(ct.1, m))
    // }

    // #[instrument(skip_all)]
    // fn hom_fma_plain_int(_P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, dst: Ciphertext<Self>, lhs: &El<BigIntRing>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_fma_plain_int");
    //     // let hom = C.inclusion().compose(C.base_ring().can_hom(&ZZbig).unwrap());
    //     // (hom.fma_map(&rhs.0, lhs, dst.0), hom.fma_map(&rhs.1, lhs, dst.1))
    // }

    // #[instrument(skip_all)]
    // fn noise_budget(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) -> usize {
    //     unimplemented!("noise_budget");
    // }

    // #[instrument(skip_all)]
    // fn gen_rk<R: Rng + CryptoRng>(C: &CiphertextMatrixRing<Self>, rng: R, sk: &SecretKey<Self>, digits: &RNSGadgetVectorDigitIndices, noise_sigma: f64) -> RelinKey<Self> {
    //     unimplemented!("gen_rk");
    //     // Self::gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk, digits, noise_sigma)
    // }

    // #[instrument(skip_all)]
    // fn hom_mul(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, C_mul: &CiphertextMatrixRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_mul");
    // }

    // #[instrument(skip_all)]
    // fn hom_square(P: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, C_mul: &CiphertextMatrixRing<Self>, val: Ciphertext<Self>, rk: &RelinKey<Self>) -> Ciphertext<Self> {
    //     unimplemented!("hom_square");
    // }

    // #[instrument(skip_all)]
    // fn gen_switch_key<R: Rng + CryptoRng>(C: &CiphertextMatrixRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: &RNSGadgetVectorDigitIndices, noise_sigma: f64) -> KeySwitchKey<Self> {
    //     unimplemented!("gen_switch_key");
    // }

    // #[instrument(skip_all)]
    // fn key_switch(C: &CiphertextMatrixRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<Self>) -> Ciphertext<Self> {
    //     unimplemented!("key_switch");
    // }

    // #[instrument(skip_all)]
    // fn mod_switch_to_plaintext(target: &PlaintextMatrixRing<Self>, C: &CiphertextMatrixRing<Self>, ct: Ciphertext<Self>) -> (El<PlaintextMatrixRing<Self>>, El<PlaintextMatrixRing<Self>>) {
    //     unimplemented!("mod_switch_to_plaintext");
    // }

    // #[instrument(skip_all)]
    // fn mod_switch_ct(_P: &PlaintextMatrixRing<Self>, Cnew: &CiphertextMatrixRing<Self>, Cold: &CiphertextMatrixRing<Self>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
    //     unimplemented!("mod_switch_ct");
    // }

    // #[instrument(skip_all)]
    // fn mod_switch_sk(_P: &PlaintextMatrixRing<Self>, Cnew: &CiphertextMatrixRing<Self>, Cold: &CiphertextMatrixRing<Self>, sk: &SecretKey<Self>) -> SecretKey<Self> {
    //     unimplemented!("mod_switch_sk");
    // }

    // fn lift_to_Cmul<'a>(C: &'a CiphertextMatrixRing<Self>, C_mul: &'a CiphertextMatrixRing<Self>) -> Box<dyn 'a + for<'b> FnMut(&'b El<CiphertextMatrixRing<Self>>) -> El<CiphertextMatrixRing<Self>>> {
    //     unimplemented!("lift_to_Cmul");
    // }

    // fn rescale_to_C<'a>(P: &PlaintextMatrixRing<Self>, C: &'a CiphertextMatrixRing<Self>, C_mul: &'a CiphertextMatrixRing<Self>) -> Box<dyn 'a + for<'b> FnMut(&'b El<CiphertextMatrixRing<Self>>) -> El<CiphertextMatrixRing<Self>>> {
    //     unimplemented!("rescale_to_C");
    // }
}