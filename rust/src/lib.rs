//! Shared-memory transport under pipe's queues: `RingQueue` (ring.rs) moves
//! pickled items between stage processes; `Store` (store.rs) holds large
//! array payloads once so later hops pass a handle instead of the bytes.

use pyo3::prelude::*;

mod ring;
mod store;
mod sys;

#[pymodule]
fn _rustq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ring::RingQueue>()?;
    m.add_class::<store::Store>()?;
    m.add_class::<store::Block>()?;
    Ok(())
}
