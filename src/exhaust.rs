pub(crate) struct ExhaustIter<I>
where
    I: Iterator,
{
    inner: I,
}

impl<I> ExhaustIter<I>
where
    I: Iterator,
{
    pub fn new(inner: I) -> Self {
        Self { inner }
    }
}

impl<I> Iterator for ExhaustIter<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<I> Drop for ExhaustIter<I>
where
    I: Iterator,
{
    fn drop(&mut self) {
        // Exhaust the remaining elements
        for _ in &mut self.inner {}
    }
}
