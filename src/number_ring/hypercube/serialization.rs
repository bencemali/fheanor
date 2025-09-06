use std::marker::PhantomData;

use feanor_math::homomorphism::*;
use feanor_math::ring::*;
use feanor_math::rings::local::AsLocalPIR;
use feanor_math::rings::poly::PolyRing;
use feanor_math::rings::zn::ZnReductionMap;
use feanor_math::serialization::*;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::poly::dense_poly::*;
// use feanor_serde::dependent_tuple::DeserializeSeedDependentTuple;
use feanor_serde::newtype_struct::DeserializeSeedNewtypeStruct;
use feanor_serde::newtype_struct::SerializableNewtypeStruct;
use feanor_serde::seq::DeserializeSeedSeq;
use feanor_serde::seq::SerializableSeq;
use serde::de::DeserializeSeed;
use serde::de::IgnoredAny;
use serde::de::SeqAccess;
use serde::de::{Visitor, Error};
use serde::Deserializer;
use serde::{Deserialize, Serialize};
use feanor_serde::impl_deserialize_seed_for_dependent_struct;

use crate::cyclotomic::*;
use crate::{NiceZn, ZZi64};

use super::isomorphism::{BaseRing, DecoratedBaseRingBase, HypercubeIsomorphism};
use super::structure::{HypercubeStructure, HypercubeTypeData};

impl Serialize for HypercubeStructure {

    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        #[derive(Serialize)]
        #[serde(rename = "HypercubeStructureData")]
        struct SerializableHypercubeStructureData<'a, G: Serialize> {
            p: SerializableCyclotomicGaloisGroupEl<'a>,
            d: usize,
            ls: &'a [usize],
            gs: G,
            choice: &'a HypercubeTypeData
        }

        SerializableNewtypeStruct::new("HypercubeStructure", (&self.galois_group, SerializableHypercubeStructureData {
            choice: &self.choice,
            d: self.d,
            p: SerializableCyclotomicGaloisGroupEl::new(&self.galois_group, self.p.clone()),
            ls: &self.ls,
            gs: SerializableSeq::new_with_len(self.gs.iter().map(|g| SerializableCyclotomicGaloisGroupEl::new(&self.galois_group, g.clone())), self.gs.len())
        })).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for HypercubeStructure {

    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        struct DeserializeSeedHypercubeStructureData {
            galois_group: CyclotomicGaloisGroup
        }

        fn derive_single_galois_group_deserializer<'a>(deserializer: &'a DeserializeSeedHypercubeStructureData) -> DeserializeSeedCyclotomicGaloisGroupEl<'a> {
            DeserializeSeedCyclotomicGaloisGroupEl::new(&deserializer.galois_group)
        }

        fn derive_multiple_galois_group_deserializer<'de, 'a>(deserializer: &'a DeserializeSeedHypercubeStructureData) -> impl use<'a, 'de> + DeserializeSeed<'de, Value = Vec<CyclotomicGaloisGroupEl>> {
            DeserializeSeedSeq::new(
                std::iter::repeat(DeserializeSeedCyclotomicGaloisGroupEl::new(&deserializer.galois_group)),
                Vec::new(),
                |mut current, next| { current.push(next); current }
            )
        }

        impl_deserialize_seed_for_dependent_struct!{
            pub struct HypercubeStructureData<'de> using DeserializeSeedHypercubeStructureData {
                p: CyclotomicGaloisGroupEl: derive_single_galois_group_deserializer,
                d: usize: |_| PhantomData,
                ls: Vec<usize>: |_| PhantomData,
                gs: Vec<CyclotomicGaloisGroupEl>: derive_multiple_galois_group_deserializer,
                choice: HypercubeTypeData: |_| PhantomData
            }
        }

        let mut deserialized_galois_group = None;
        Ok(DeserializeSeedNewtypeStruct::new("HypercubeStructure", DeserializeSeedDependentTuple::new(
            PhantomData::<CyclotomicGaloisGroup>,
            |galois_group| {
                deserialized_galois_group = Some(galois_group.clone());
                DeserializeSeedHypercubeStructureData { galois_group }
            }
        )).deserialize(deserializer).map(|data| {
            let mut result = HypercubeStructure::new(deserialized_galois_group.take().unwrap(), data.p, data.d, data.ls, data.gs);
            result.choice = data.choice;
            return result;
        }).unwrap())
    }
}

pub struct DeserializeSeedDependentTuple<'de, T0, F, T1>
    where T0: DeserializeSeed<'de>,
        T1: DeserializeSeed<'de>,
        F: FnOnce(T0::Value) -> T1
{
    deserializer: PhantomData<&'de ()>,
    first: T0,
    derive_second: F
}

impl<'de, T0, F, T1> DeserializeSeedDependentTuple<'de, T0, F, T1>
    where T0: DeserializeSeed<'de>,
        T1: DeserializeSeed<'de>,
        F: FnOnce(T0::Value) -> T1
{
    pub fn new(first: T0, derive_second: F) -> Self {
        Self {
            deserializer: PhantomData,
            first: first,
            derive_second: derive_second
        }
    }
}

impl<'de, T0, F, T1> DeserializeSeed<'de> for DeserializeSeedDependentTuple<'de, T0, F, T1>
    where T0: DeserializeSeed<'de>,
        T1: DeserializeSeed<'de>,
        F: FnOnce(T0::Value) -> T1
{
    type Value = T1::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where D: Deserializer<'de>
    {
        pub struct ResultVisitor<'de, T0, F, T1>
            where T0: DeserializeSeed<'de>,
                T1: DeserializeSeed<'de>,
                F: FnOnce(T0::Value) -> T1
        {
            deserializer: PhantomData<&'de ()>,
            first: T0,
            derive_second: F
        }

        impl<'de, T0, F, T1> Visitor<'de> for ResultVisitor<'de, T0, F, T1>
            where T0: DeserializeSeed<'de>,
                T1: DeserializeSeed<'de>,
                F: FnOnce(T0::Value) -> T1
        {
            type Value = T1::Value;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a tuple with 2 elements")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where A: SeqAccess<'de>
            {
                if let Some(first) = seq.next_element_seed(self.first).unwrap() {
                    if let Some(second) = seq.next_element_seed((self.derive_second)(first)).unwrap() {
                        if let Some(_) = seq.next_element::<IgnoredAny>().unwrap() {
                            return Err(<A::Error as Error>::invalid_length(3, &"a tuple with 2 elements"));
                        } else {
                            return Ok(second);
                        }
                    } else {
                        return Err(<A::Error as Error>::invalid_length(1, &"a tuple with 2 elements"));
                    }
                } else {
                    return Err(<A::Error as Error>::invalid_length(0, &"a tuple with 2 elements"));
                }
            }
        }

        return deserializer.deserialize_tuple(2, ResultVisitor {
            deserializer: PhantomData,
            first: self.first,
            derive_second: self.derive_second
        });
    }
}

///
/// Wrapper around a reference to a [`HypercubeIsomorphism`] that
/// can be used for serialization, without including the ring.
/// 
/// This can be deserialized using [`DeserializeSeedHypercubeIsomorphismWithoutRing`]
/// if the ring is provided during deserialization time.
/// 
pub struct SerializableHypercubeIsomorphismWithoutRing<'a, R>
    where R: RingStore,
        R::Type: CyclotomicQuotient,
        BaseRing<R>: NiceZn + SerializableElementRing,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    hypercube_isomorphism: &'a HypercubeIsomorphism<R>
}

impl<'a, R> SerializableHypercubeIsomorphismWithoutRing<'a, R>
    where R: RingStore,
        R::Type: CyclotomicQuotient,
        BaseRing<R>: NiceZn + SerializableElementRing,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    pub fn new(hypercube_isomorphism: &'a HypercubeIsomorphism<R>) -> Self {
        Self { hypercube_isomorphism }
    }
}

impl<'a, R> Serialize for SerializableHypercubeIsomorphismWithoutRing<'a, R>
    where R: RingStore,
        R::Type: CyclotomicQuotient,
        BaseRing<R>: NiceZn + SerializableElementRing,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        #[derive(Serialize)]
        #[serde(rename = "HypercubeIsomorphismData", bound = "")]
        struct SerializableHypercubeIsomorphismData<'a, R>
            where R: RingStore,
                R::Type: PolyRing + SerializableElementRing
        {
            p: i64,
            e: usize,
            m: usize,
            hypercube_structure: &'a HypercubeStructure,
            slot_ring_moduli: Vec<SerializeOwnedWithRing<R>>
        }

        let decorated_base_ring: RingValue<DecoratedBaseRingBase<R>> = AsLocalPIR::from_zn(RingValue::from(self.hypercube_isomorphism.ring().base_ring().get_ring().clone())).unwrap();
        let ZpeX = DensePolyRing::new(decorated_base_ring, "X");
        let hom = ZnReductionMap::new(self.hypercube_isomorphism.slot_ring().base_ring(), ZpeX.base_ring()).unwrap();
        SerializableHypercubeIsomorphismData {
            p: self.hypercube_isomorphism.p(),
            e: self.hypercube_isomorphism.e(),
            m: self.hypercube_isomorphism.hypercube().m(),
            hypercube_structure: self.hypercube_isomorphism.hypercube(),
            slot_ring_moduli: (0..self.hypercube_isomorphism.slot_count()).map(|i| 
                SerializeOwnedWithRing::new(self.hypercube_isomorphism.slot_ring_at(i).generating_poly(&ZpeX, &hom), &ZpeX)
            ).collect()
        }.serialize(serializer)
    }
}

struct DeserializeSeedHypercubeIsomorphismData<R>
    where R: RingStore,
        R::Type: PolyRing + SerializableElementRing
{
    poly_ring: R
}

fn derive_multiple_poly_deserializer<'de, 'a, R>(deserializer: &'a DeserializeSeedHypercubeIsomorphismData<R>) -> impl use <'a, 'de, R> + DeserializeSeed<'de, Value = Vec<El<R>>>
    where R: RingStore,
        R::Type: PolyRing + SerializableElementRing
{
    DeserializeSeedSeq::new(
        std::iter::repeat(DeserializeWithRing::new(&deserializer.poly_ring)),
        Vec::new(),
        |mut current, next| { current.push(next); current }
    )
}

impl_deserialize_seed_for_dependent_struct!{
    <{'de, R}> pub struct HypercubeIsomorphismData<{'de, R}> using DeserializeSeedHypercubeIsomorphismData<R> {
        p: i64: |_| PhantomData,
        e: usize: |_| PhantomData,
        m: usize: |_| PhantomData,
        hypercube_structure: HypercubeStructure: |_| PhantomData,
        slot_ring_moduli: Vec<El<R>>: derive_multiple_poly_deserializer
    } where R: RingStore, R::Type: PolyRing + SerializableElementRing
}

///
/// A [`DeserializeSeed`] to deserialize a [`HypercubeIsomorphism`]
/// that has been serialized without the ring. Hence, for deserialization,
/// it is necessary that the ring is provided again. Therefore, we must
/// use a [`DeserializeSeed`] wrapping the ring, i.e. this struct.
/// 
pub struct DeserializeSeedHypercubeIsomorphismWithoutRing<R>
    where R: RingStore,
        R::Type: CyclotomicQuotient,
        BaseRing<R>: NiceZn + SerializableElementRing,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    ring: R
}

impl<R> DeserializeSeedHypercubeIsomorphismWithoutRing<R>
    where R: RingStore,
        R::Type: CyclotomicQuotient,
        BaseRing<R>: NiceZn + SerializableElementRing,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    pub fn new(ring: R) -> Self {
        Self { ring }
    }
}

impl<'de, R> DeserializeSeed<'de> for DeserializeSeedHypercubeIsomorphismWithoutRing<R>
    where R: RingStore,
        R::Type: CyclotomicQuotient,
        BaseRing<R>: NiceZn + SerializableElementRing,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    type Value = HypercubeIsomorphism<R>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where D: serde::Deserializer<'de>
    {
        let decorated_base_ring: RingValue<DecoratedBaseRingBase<R>> = AsLocalPIR::from_zn(RingValue::from(self.ring.base_ring().get_ring().clone())).unwrap();
        let ZpeX = DensePolyRing::new(decorated_base_ring, "X");
        let deserialized = DeserializeSeedHypercubeIsomorphismData { poly_ring: &ZpeX }.deserialize(deserializer)?;
        assert_eq!(self.ring.m(), deserialized.m, "ring mismatch");
        assert_eq!(self.ring.characteristic(ZZi64).unwrap(), ZZi64.pow(deserialized.p, deserialized.e), "ring mismatch");
        let hypercube_structure = deserialized.hypercube_structure;
        let slot_ring_moduli = deserialized.slot_ring_moduli;
        let result = HypercubeIsomorphism::create::<false>(
            self.ring,
            hypercube_structure,
            ZpeX,
            slot_ring_moduli
        );
        return Ok(result);
    }
}

impl<R> Serialize for HypercubeIsomorphism<R>
    where R: RingStore + Serialize,
        R::Type: CyclotomicQuotient,
        BaseRing<R>: NiceZn + SerializableElementRing,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        SerializableNewtypeStruct::new("HypercubeIsomorphism", (self.ring(), SerializableHypercubeIsomorphismWithoutRing::new(self))).serialize(serializer)
    }
}

impl<'de, R> Deserialize<'de> for HypercubeIsomorphism<R>
    where R: RingStore + Deserialize<'de>,
        R::Type: CyclotomicQuotient,
        BaseRing<R>: NiceZn + SerializableElementRing,
        DecoratedBaseRingBase<R>: CanIsoFromTo<BaseRing<R>>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("HypercubeIsomorphism", DeserializeSeedDependentTuple::new(
            PhantomData::<R>,
            |ring| DeserializeSeedHypercubeIsomorphismWithoutRing::new(ring)
        )).deserialize(deserializer)
    }
}
