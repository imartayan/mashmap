use ahash::RandomState;
use hashbrown::raw::{Bucket, RawTable};
use hashbrown::TryReserveError;
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};

#[cfg(not(feature = "nightly"))]
use core::convert::identity as likely;
#[cfg(feature = "nightly")]
use core::intrinsics::likely;

#[inline]
fn equivalent_key<Q, K, V>(k: &Q) -> impl Fn(&(K, V)) -> bool + '_
where
    K: Borrow<Q>,
    Q: ?Sized + Hash + Eq,
{
    move |x| k == x.0.borrow()
}

#[inline]
fn make_hash<T, S>(hash_builder: &S, value: &T) -> u64
where
    T: ?Sized + Hash,
    S: BuildHasher,
{
    hash_builder.hash_one(value)
}

#[inline]
fn make_hasher<K, V, S>(hash_builder: &S) -> impl Fn(&(K, V)) -> u64 + '_
where
    K: Hash,
    S: BuildHasher,
{
    move |val| make_hash(hash_builder, &val.0)
}

/// A flat HashMap that supports multiple entries per key.
///
/// # Examples
///
/// ```
/// use mashmap::MashMap;
///
/// let mut map = MashMap::new();
/// map.insert(1, 1);
/// map.insert(1, 2);
/// map.insert(2, 3);
///
/// assert_eq!(map.len(), 3);
/// ```
#[derive(Clone)]
pub struct MashMap<K, V, S = RandomState> {
    hash_builder: S,
    pub(crate) table: RawTable<(K, V)>,
}

impl<K, V> MashMap<K, V, RandomState> {
    /// Creates an empty `MashMap` with a capacity of 0,
    /// so it will not allocate until it is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use mashmap::MashMap;
    ///
    /// let mut map: MashMap<&str, i32> = MashMap::new();
    ///
    /// assert_eq!(map.capacity(), 0);
    /// ```
    pub fn new() -> Self {
        Self::with_hasher(RandomState::default())
    }

    /// Creates an empty `MashMap` with at least the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::default())
    }
}

impl<K, V> Default for MashMap<K, V, RandomState> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> MashMap<K, V, S> {
    /// Creates an empty `MashMap` with default capacity which will use the given hash builder to hash keys.
    pub const fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            table: RawTable::new(),
        }
    }

    /// Creates an empty `MashMap` with at least the specified capacity, using the given hash builder to hash keys.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            hash_builder,
            table: RawTable::with_capacity(capacity),
        }
    }

    /// Returns the number of elements the map can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Returns a reference to the map’s [`BuildHasher`].
    #[inline]
    pub const fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Returns the number of elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns `true` if the map contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.table.clear();
    }

    /// Clears the map, returning all key-value pairs as an iterator.
    #[inline]
    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        self.table.drain()
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        unsafe {
            self.table
                .iter()
                .filter(|bucket| {
                    let &mut (ref key, ref mut value) = bucket.as_mut();
                    !f(key, value)
                })
                .for_each(|bucket| {
                    self.table.erase(bucket);
                })
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        unsafe { self.table.iter().map(|bucket| bucket.as_ref()) }
    }

    #[inline]
    pub fn iter_mut(&self) -> impl Iterator<Item = &mut (K, V)> {
        unsafe { self.table.iter().map(|bucket| bucket.as_mut()) }
    }
}

impl<K, V, S> MashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Reserves capacity for at least additional more elements to be inserted in the `MashMap`.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.table
            .reserve(additional, make_hasher(&self.hash_builder));
    }

    /// Tries to reserve capacity for at least additional more elements to be inserted in the `MashMap`.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.table
            .try_reserve(additional, make_hasher(&self.hash_builder))
    }

    /// Shrinks the capacity of the map as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.table.shrink_to(0, make_hasher(&self.hash_builder));
    }

    /// Shrinks the capacity of the map with a lower limit.
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.table
            .shrink_to(min_capacity, make_hasher(&self.hash_builder));
    }

    /// Inserts a key-value pair into the map.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) {
        let hash = make_hash(&self.hash_builder, &key);
        self.table
            .insert(hash, (key, value), make_hasher(&self.hash_builder));
    }

    #[inline]
    fn get_iter_buckets<'a, Q>(&'a self, key: &'a Q) -> impl Iterator<Item = Bucket<(K, V)>> + 'a
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        let hash = make_hash(&self.hash_builder, key);
        unsafe {
            self.table
                .iter_hash(hash)
                .filter(move |bucket| likely(bucket.as_ref().0.borrow() == key))
        }
    }

    #[inline]
    pub fn get_iter<'a, Q>(&'a self, key: &'a Q) -> impl Iterator<Item = &V> + 'a
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.get_iter_buckets(key)
            .map(|bucket| unsafe { &bucket.as_ref().1 })
    }

    #[inline]
    pub fn get_mut_iter<'a, Q>(&'a self, key: &'a Q) -> impl Iterator<Item = &mut V> + 'a
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.get_iter_buckets(key)
            .map(|bucket| unsafe { &mut bucket.as_mut().1 })
    }

    /// Returns `true` if the map contains at least a single value for the specified key.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        let hash = make_hash(&self.hash_builder, key);
        self.table.find(hash, equivalent_key(key)).is_some()
    }

    #[inline]
    pub fn get_one<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        let hash = make_hash(&self.hash_builder, key);
        self.table
            .find(hash, equivalent_key(key))
            .map(|bucket| unsafe { bucket.as_ref().1.borrow() })
    }

    #[inline]
    pub fn get_one_mut<Q>(&self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        let hash = make_hash(&self.hash_builder, key);
        self.table
            .find(hash, equivalent_key(key))
            .map(|bucket| unsafe { &mut bucket.as_mut().1 })
    }

    /// Removes an arbitrary value with the given key from the map, returning the removed value if there was a value at the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use mashmap::MashMap;
    ///
    /// let mut map = MashMap::new();
    /// map.insert(1, 1);
    /// map.insert(1, 2);
    ///
    /// assert!(map.remove_one(&1).is_some()); // Could be either Some(1) or Some(2).
    /// assert!(map.remove_one(&1).is_some()); // Could be either Some(1) or Some(2), depending on the previous remove_one.
    /// assert!(map.remove_one(&1).is_none());
    /// ```
    #[inline]
    pub fn remove_one<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        let hash = make_hash(&self.hash_builder, key);
        self.table
            .remove_entry(hash, equivalent_key(key))
            .map(|(_, value)| value)
    }

    pub fn remove_all<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        let hash = make_hash(&self.hash_builder, key);
        unsafe {
            self.table
                .iter_hash(hash)
                .filter(|bucket| likely(bucket.as_ref().0.borrow() == key))
                .for_each(|bucket| {
                    self.table.remove(bucket);
                })
        }
    }

    pub fn drain_key<'a, Q>(&'a mut self, key: &'a Q) -> impl Iterator<Item = V> + 'a
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        let hash = make_hash(&self.hash_builder, key);
        unsafe {
            self.table
                .iter_hash(hash)
                .filter(move |bucket| likely(bucket.as_ref().0.borrow() == key))
                .map(|bucket| (self.table.remove(bucket).0).1)
        }
    }
}

impl<K, V, S> Extend<(K, V)> for MashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        iter.for_each(move |(k, v)| {
            self.insert(k, v);
        });
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_map() {
        let mut map = MashMap::<usize, usize>::new();
        map.insert(1, 10);
        map.insert(1, 11);
        map.insert(1, 12);
        map.insert(2, 20);
        map.insert(2, 21);

        // iterate over mutable values associated with key `1`
        // and increment them
        for v in map.get_mut_iter(&1) {
            *v += 1;
        }

        // collect the values associated with keys `1` and `2`
        // note that the order may differ from the insertion order
        let mut values_1: Vec<_> = map.get_iter(&1).copied().collect();
        let mut values_2: Vec<_> = map.get_iter(&2).copied().collect();
        values_1.sort_unstable();
        values_2.sort_unstable();

        assert_eq!(values_1, vec![11, 12, 13]);
        assert_eq!(values_2, vec![20, 21]);
    }
}
