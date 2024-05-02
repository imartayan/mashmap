# MashMap

A flat HashMap that supports multiple entries per key.

This is an adaptation of Rust's standard `HashMap` (using [hashbrown](https://github.com/rust-lang/hashbrown)'s [`RawTable`](https://docs.rs/hashbrown/latest/hashbrown/raw/struct.RawTable.html)) to support multiple entries with the same key.
While a common approach is to use a `HashMap<K, Vec<V>>` to store the entries corresponding to a key in a `Vec`, `MashMap` keeps a flat layout and stores all entries in the same table using probing to select the slots.
This approach avoids the memory indirection caused by vector pointers, and reduces memory overhead since it avoids storing the pointer + length + capacity of a vector for each key.

## Example usage

```rs
use mashmap::MashMap;

let mut map = MashMap::<usize, usize>::new();
map.insert(1, 10);
map.insert(1, 11);
map.insert(1, 12);
map.insert(2, 20);
map.insert(2, 21);

// iterate over the values with key `1` with mutable references and increment them
for v in map.get_mut_iter(&1) {
    *v += 1;
}

// collect the values with keys `1` and `2`
// note that the order may differ from the insertion order
let mut values_1: Vec<_> = map.get_iter(&1).copied().collect();
let mut values_2: Vec<_> = map.get_iter(&2).copied().collect();
values_1.sort_unstable();
values_2.sort_unstable();

assert_eq!(values_1, vec![11, 12, 13]);
assert_eq!(values_2, vec![20, 21]);
```

## Acknowledgement

This crate was inspired by the [flat-multimap](https://crates.io/crates/flat-multimap) crate which does not have a public repository anymore.
