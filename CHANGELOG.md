# Changelog

## 0.2.2

- Add `drain_key_if` and `remove_key_if` to remove values satisfying a certain predicate, thanks to @shanecelis.
- Update `drain_key` to exhausted the iterator when dropped.

## 0.2.1

- Fix `iter_group_by_key` and `iter_mut_group_by_key` to actually group by key and not by hash.

## 0.2.0

- Add `iter_group_by_key` and `iter_mut_group_by_key` to iterate over values while grouping them by key.
