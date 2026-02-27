//! # Arena
//!
//! A small, extremely fast, lock-free thread-safe arena bump allocator that can hand out multiple
//! mutable elements, structs, slices, or read-only &strs from a single pre-allocated block.
//!
//! ## What is an Arena?
//!
//! An arena allocator is a memory allocation strategy that pre-allocates a large block
//! of memory once, and can then hand out sub-allocations from that block sequentially.
//! Arenas are much faster than normal memory allocations because:
//!
//!   - **Bulk Allocation**: The entire arena is allocated all at once
//!   - **No Fragmentation**: Memory is allocated sequentially
//!   - **Fast Bookkeeping**: No complex tracking/reallocations
//!   - **Simplified Deallocation**: The entire arena is freed simultaneously
//!
//! `Arena` is special in that, because of its design, it is not subject
//! to the same use after free or overlapped memory bugs that are possible with
//! some other bump allocators.
//!
//! ## Quick Start
//!
//! Add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! arena = "0.1"
//! ```
//!
//! ```rust
//!   use arena::Arena;
//!   fn main() {
//!     let mut arena = Arena::new(1024).expect("Failed to allocate memory");
//!     let num = arena.alloc(42u32);
//!     let stnum = format!("Num: {}", *num);
//!     let s = arena.alloc_str(&stnum);
//!     println!("{}  {:?}", *num, s);
//!   }
//! ```
//!
//! # The API
//!
//! ## new(byte_capacity)
//! Allocate a new Arena with a specified capacity.
//!
//! ## alloc(item) or try_alloc(item)
//! Allocate an element or structure in the Arena
//!
//! ## alloc_slice(init_value, items) or try_alloc_slice(init_value, items)
//! Allocate a slice vector of elements
//!
//! ## alloc_str(&str) or try_alloc_str(&str)
//! Store a &str in the Arena
//!
//! ## reset()
//! Reset the arena. This requires that all allocations are vacated, and
//! re-initializes the Arena to it's brand new state.
//!
//! ## Tradeoffs
//!
//!   - The entire arena will be dropped in a single operation. Individual Drop
//!     operations will not be performed on the Arena's contents. This then will
//!     leak any memory separately allocated as with Strings and Vecs.
//!
//!   - **No item Reclamation**: Any unused allocations are stuck until
//!     the whole arena is dropped or reset().
//!
//!   - Individual Items are NOT DROPPED
//!     You need to individually Box<T> any items, (Strings, Vecs, Fat Pointers,
//!     file handles, etc) to avoid leaking memory.
//!
//!   - **Fixed Size**: The arena has a set fixed size that doesn't grow.
//!
//! # MIRI to the Rescue
//!
//! In some limited testing, MIRI successfully detected the most common forms of
//! memory leaks. Please test your code with Miri.
//!
//! ```ignore
//!   cargo +nightly miri run
//! ```
//!
//!
//! # Use Cases:
//!
//!  - **Long-lived Data**: Perform one alloc from the system, and break that into
//!    all the allocations your need for the life of your program
//!
//!  - **Short-lived Processing**: Temporary allocations for a process... encoding,
//!    parsing, compiling, translation, etc. When the function returns, all the
//!    memory is returned in a single free.
//!
//!  - **Saving Space**: Many system allocation schemes allocate more memory than necessary
//!    so freed memory can be more efficiently managed for reallocation. Arena fills
//!    every byte it can excepting alignment requirements.
//!
//!
//! # Design Choices
//!
//! There are hundreds of possible improvements...  A lot of them very
//! useful...
//!
//!  - Chunk Size, Statistics, Diagnostics, Memory Trimming, Snapshots - See arena-b
//!  - Generation Counter and Key reservation - See atomic-arena
//!  - Growable - See blink-alloc
//!  - Memory Paging and Arena Growth - See arena-allocator
//!  - Memory Reclamation from Individual Items - See drop-arena
//!  - Scoped Allocator, so you can restore memory in stages - See bump-scope
//!  - Memory Pools - See shared-arena
//!  - Boxed Allocations or Collections so you CAN use an arena with strings
//!       and vecs. See Rodeo and Bumpalo
//!  - Memory Layout Control, Rewinding, Thread-Local memory lakes, etc (See lake)
//!  - Detect Use after free - See arena-allocator
//!
//! The goal of this was Simple, Fast, and Multi-threaded, to the degree possible.
//!
//! # Where NOT to use Arenas:
//!
//! ❌ - Don't do this:
//! ```ignore
//!      let v = arena.try_alloc("Hello".to_string())?;    <== Still allocates from the heap
//! ```
//!
//! ✅ - Do this instead:
//! ```ignore
//!      let v = arena.try_alloc("Hello")?;   <==  Arena based READ ONLY str
//! ```
//!
//! ❌ - Don't do this:  
//! ```ignore
//!      let v = arena.try_alloc(vec![42u32; 10])?;  <== Allocates data on the heap
//! ```
//!
//! In both cases of the Don't do this, a fat pointer will be stored in the arena,
//! and memory for the data or string will be allocated and LEAKED on the heap.
//!
//! ## License
//! MIT
//!
//! ## Credits
//! Reverse allocations inspired by:
//!   https://fitzgen.com/2019/11/01/always-bump-downwards.html
//!
//! ## Contributions
//!
//! All contributions intentionally submitted for inclusion in this work by you will
//! be TODO

// Note, Miri has trouble with the sysconf(_SC_CLK_TCK), so Miri testing skipped the
// test_large_allocation that uses sysconf.

use std::alloc::{Layout, dealloc};
use std::marker::PhantomData;
use std::mem;
use std::num::NonZero;
use std::ptr::{copy_nonoverlapping, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Arena
///
/// A small, fast and light basic arena allocator that can hand out multiple
/// mutable slices of different types.
///
/// This crate is optimized for performance-critical applications removing some
/// of Rust’s memory safety guarantees, and safety, including slice initialization
/// and individual items being dropped.
///
/// This crate completely removes the memory reuse protections, and use after free
/// detection, and panics if anything that shouldn't go wrong does, in fact, go wrong.
///
///  ALLOCED ITEMS DON'T DROP
///
///  NO USE AFTER FREE DETECTION OR PROVENTION
///
/// Great care is required! Proper understanding of its behavior and limitations
/// is mandatory to prevent memory leaks, or UB undefined behavior.
///
/// Specifically, any element types that reserver memory or resources, file handles,
/// vecs, and strings will leak memory because Drop is NOT EXECUTED when arena
/// drops or clears.
///
mod error;
pub use self::error::{Error, Result};

// Inspiration: https://github.com/emoon/arena-allocator/blob/main/src/lib.rs

/// A lightweight arena allocator for fast allocation of multiple mutable slices.
///
/// # Examples
///
/// ```
/// use arena::*;
///
/// fn main() -> Result<()> {
///     let arena = Arena::new(1024)?;
///     let slice = arena.try_alloc_slice(0u32, 4)?;
///     for i in 0..4 {
///         slice[i] = i as u32;
///     }
///     println!("{:?}", slice);
///     Ok(())
/// }
/// ```
pub struct Arena<'a> {
    buf: NonNull<u8>,
    end_byte_idx: AtomicUsize, // Allows for interior mutability without RefCells, Arcs, etc.
    layout: Layout,            //    byte_capacity: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Arena<'a> {
    /// Creates a new `Arena` with the specified byte capacity.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - byte_capacity is 0
    /// - Memory allocation fails
    /// - Layout creation fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use arena::*;
    ///
    /// fn main() -> Result<()> {
    ///     let arena = Arena::new(1024)?;
    ///
    ///     assert!(matches!( Arena::new(usize::MAX), Err(Error::Layout(_)) ));
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn new(byte_capacity: usize) -> Result<Self> {
        assert!(byte_capacity > 0, "Capacity must be greater than 0");

        let layout = Layout::from_size_align(byte_capacity, mem::align_of::<u8>())?;
        let buf = unsafe {
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                return Err(Error::OutOfMemory);
            }
            ptr as *mut u8
        };
        Ok(Self {
            buf: NonNull::new(buf).ok_or(Error::PointerUnderflow)?,
            end_byte_idx: AtomicUsize::new(byte_capacity),
            layout,
            _marker: PhantomData,
        })
    }

    /// Allocates space for a single element and returns a mutable reference to it.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The arena has enough remaining memory
    /// - The reference is not used after the arena is reset or dropped
    /// - The type T is Copy (enforced by the trait bound)
    ///
    /// # Examples
    ///
    /// ```
    /// use arena::*;
    ///
    /// fn main() -> Result<()> {
    ///     let arena = Arena::new(1024)?;
    ///     let value = arena.try_alloc(42u32)?;
    ///     let num = arena.try_alloc(24u32)?;
    ///     *num = 100;
    ///     println!("{}", *num);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn alloc<T>(&self, val: T) -> &mut T {
        self.try_alloc(val)
            .unwrap_or_else(|e| panic!("Arena Failed: {}", e))
    }

    pub fn try_alloc<T>(&self, val: T) -> Result<&mut T> {
        let sizet = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        debug_assert!(sizet > 0, "Can't alloc 0 bytes");
        debug_assert!(align.is_power_of_two(), "Alignment must be a power of two");

        unsafe {
            loop {
                let end_byte_idx = self.end_byte_idx.load(Ordering::Relaxed);
                let ptr_num = (self.buf.as_ptr().add(end_byte_idx as usize) as usize)
                    .checked_sub(sizet)
                    .ok_or(Error::PointerUnderflow)?;

                //let ptr = (ptr as usize & !(align - 1)) as *mut u8;  // Miri-sad-version
                let ptr = self
                    .buf
                    .with_addr(NonZero::new(ptr_num & !(align - 1)).ok_or(Error::PointerUnderflow)?)
                    .as_ptr() as *mut u8;

                if (ptr as usize) < self.buf.as_ptr() as usize {
                    return Err(Error::OutOfMemory);
                }
                let new_end_byte_idx =
                    (ptr as usize).saturating_sub(self.buf.as_ptr() as usize) as usize;

                if let Ok(_) = self.end_byte_idx.compare_exchange_weak(
                    end_byte_idx,     // Expected value
                    new_end_byte_idx, // New value
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    std::ptr::write(ptr as *mut T, val);
                    return Ok(&mut *(ptr as *mut T));
                }
            }
        }
    }

    /// Allocates space for a slice and returns a mutable slice reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The arena has enough remaining memory
    /// - The reference is not used after the arena is reset or dropped
    /// - The type T is Copy (enforced by the trait bound)
    ///
    /// # Example
    ///
    /// ```
    /// use arena::*;
    ///
    /// fn main() -> Result<()> {
    ///     let arena = Arena::new(1024)?;
    ///         let slice = arena.try_alloc_slice(0u32, 4)?;
    ///         for i in 0..4 {
    ///             slice[i] = i as u32;
    ///         }
    ///         println!("{:?}", slice);
    ///     Ok(())
    /// }
    /// ```
    // My preference was for MaybeUninit<T>, but handling that is so onerous
    // it was easier to just initialize the memory.
    //
    // Attempted to enter the world of strict_provenance
    // std::ptr::with_exposed_provenance_mut(), etc.
    #[inline]
    pub fn alloc_slice<T>(&self, initial_value: T, len: usize) -> &mut [T] {
        self.try_alloc_slice(initial_value, len)
            .unwrap_or_else(|e| panic!("Arena Failed: {}", e))
    }

    pub fn try_alloc_slice<T>(&self, initial_value: T, len: usize) -> Result<&mut [T]> {
        let sizet = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        debug_assert!(sizet > 0, "Can't alloc 0 bytes");
        debug_assert!(align.is_power_of_two(), "Alignment must be a power of two");

        // This performs a compare and exchange loop on atomicUsize for the end_byte_idx value...
        // Making this algorithm safe for multi-thread apps
        unsafe {
            loop {
                let end_byte_idx = self.end_byte_idx.load(Ordering::Relaxed);
                let ptr_num = (self.buf.as_ptr().add(end_byte_idx) as usize)
                    .checked_sub(len * sizet)
                    .ok_or(Error::PointerUnderflow)?;

                //let ptr = (ptr as usize & !(align - 1)) as *mut u8;  // Miri-sad-version
                let ptr = self
                    .buf
                    .with_addr(NonZero::new(ptr_num & !(align - 1)).ok_or(Error::PointerUnderflow)?)
                    .as_ptr() as *mut u8;

                if (ptr as *mut u8 as usize) < self.buf.as_ptr() as usize {
                    return Err(Error::OutOfMemory);
                }
                let new_end_byte_idx =
                    (ptr as usize).saturating_sub(self.buf.as_ptr() as usize) as usize;

                if let Ok(_) = self.end_byte_idx.compare_exchange_weak(
                    end_byte_idx,     // Expected value
                    new_end_byte_idx, // New value
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    // Initialize New Slice
                    if sizet == 1 {
                        // Bytes are VERY FAST to initialize
                        let byte_ptr = &initial_value as *const T as *const u8;
                        std::ptr::write_bytes(ptr, *byte_ptr, len * sizet);
                    } else if is_all_zeros(&initial_value) {
                        // Zeroed Memory
                        std::ptr::write_bytes(ptr, 0, len * sizet);
                    } else {
                        // Not so fast!!!
                        let initial_value_ptr = &initial_value as *const T as *const u8;
                        for i in 0..len {
                            copy_nonoverlapping(
                                initial_value_ptr,
                                (ptr as *mut u8).add(i * sizet),
                                sizet,
                            );
                        }
                    }
                    return Ok(std::slice::from_raw_parts_mut(ptr as *mut T, len));
                }
            }
        }
    }

    /// Allocates space for a str and returns a mutable reference to it.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The arena has enough remaining memory
    /// - The reference is not used after the arena is reset or dropped
    /// - The type T is Copy (enforced by the trait bound)
    ///
    /// # Example
    ///
    /// ```
    /// use arena::*;
    ///
    /// fn main() -> Result<()> {
    ///     let arena = Arena::new(1024)?;
    ///         let num = arena.try_alloc(42u32)?;
    ///         *num = 100;
    ///         let stnum = format!("Num: {}", *num);
    ///         let s = arena.try_alloc_str(&stnum)?;
    ///         println!("{}  {:?}", *num, s);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn alloc_str(&self, st: &str) -> &str {
        self.try_alloc_str(st)
            .unwrap_or_else(|e| panic!("Arena Failed: {}", e))
    }

    pub fn try_alloc_str(&self, st: &str) -> Result<&str> {
        let sizet = st.len();
        let align = std::mem::align_of::<u8>();
        if sizet == 0 {
            return Ok::<&str, Error>("");
        }
        debug_assert!(align.is_power_of_two(), "Alignment must be a power of two");

        unsafe {
            loop {
                let end_byte_idx = self.end_byte_idx.load(Ordering::Relaxed);
                let ptr_num = (self.buf.as_ptr().add(end_byte_idx as usize) as usize)
                    .checked_sub(sizet)
                    .ok_or(Error::PointerUnderflow)?;

                //let ptr = (ptr as usize & !(align - 1)) as *mut u8;  // Miri-sad-version
                let ptr = self
                    .buf
                    .with_addr(NonZero::new(ptr_num & !(align - 1)).ok_or(Error::PointerUnderflow)?)
                    .as_ptr() as *mut u8;

                if (ptr as usize) < self.buf.as_ptr() as usize {
                    return Err(Error::OutOfMemory);
                }
                let new_end_byte_idx =
                    (ptr as usize).saturating_sub(self.buf.as_ptr() as usize) as usize;

                if let Ok(_) = self.end_byte_idx.compare_exchange_weak(
                    end_byte_idx,     // Expected value
                    new_end_byte_idx, // New value
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    copy_nonoverlapping(st.as_ptr(), ptr, sizet);
                    // Unchecked is Ok since the bytes came from a valid str
                    return Ok(std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                        ptr, sizet,
                    )));
                }
            }
        }
    }

    /// Returns the number of bytes remaining in the arena.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arena::*;
    ///
    /// fn main() -> Result<()> {
    ///     let arena = Arena::new(1024)?;
    ///     assert_eq!(arena.remaining(), 1024);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn remaining(&self) -> usize {
        self.end_byte_idx.load(Ordering::Relaxed)
    }

    /// Resets then arena, making all previously allocated memory available again.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arena::*;
    ///
    /// fn main() -> Result<()> {
    ///     let mut arena = Arena::new(1024)?;
    ///     unsafe {
    ///         let slice = arena.try_alloc_slice(1u8, 100)?;
    ///     assert_eq!(arena.remaining(), 924);
    ///     arena.reset();
    ///     assert_eq!(arena.remaining(), 1024);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn reset(&mut self) {
        loop {
            let end_byte_idx = self.end_byte_idx.load(Ordering::Relaxed);

            if let Ok(_) = self.end_byte_idx.compare_exchange_weak(
                end_byte_idx,       // Expected value
                self.layout.size(), // New value
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                return ();
            }
        }
    }
}

impl Drop for Arena<'_> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            dealloc(self.buf.as_ptr(), self.layout);
        }
    }
}

unsafe impl Send for Arena<'_> {}
unsafe impl Sync for Arena<'_> {}

/// This function detects if the passed in value is comprised of all
/// zeros, which is much faster to initialize.
#[inline]
fn is_all_zeros<T>(value: &T) -> bool {
    let num_bytes = std::mem::size_of::<T>();
    unsafe {
        let ptr = value as *const T as *const u8;
        for i in 0..num_bytes {
            if *ptr.add(i) != 0 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod test {
    //type Result<T> = core::result::Result<T, Error>; // It's really cool... These OVERWRITE these types for testing
    //type Error = Box<dyn std::error::Error>;

    use super::*;
    use sysinfo::{Pid, System};

    #[test]
    fn test_try_alignment() -> Result<()> {
        let arena = Arena::new(1024)?;

        let _a = arena.alloc_slice(0i8, 1); // 1-byte aligned
        let _a = arena.alloc_slice(0i8, 1); // 1-byte aligned
        let a = arena.alloc_slice(0i8, 1); // 1-byte aligned
        assert_eq!(a.as_ptr() as usize % 1, 0);

        let b = arena.alloc(0u16); // 2-byte aligned
        assert_eq!(b as *const u16 as usize % 2, 0);

        let c = arena.alloc_slice(0i32, 1); // 4-byte aligned
        assert_eq!(c.as_ptr() as usize % 4, 0);

        let d = arena.alloc_slice(0i64, 1); // 8-byte aligned
        assert_eq!(d.as_ptr() as usize % 8, 0);

        // try_
        let a = arena.try_alloc_slice(0i8, 1)?; // 1-byte aligned
        assert_eq!(a.as_ptr() as usize % 1, 0);

        let b = arena.try_alloc(0u16)?; // 2-byte aligned
        assert_eq!(b as *const u16 as usize % 2, 0);

        let c = arena.try_alloc_slice(0i32, 1)?; // 4-byte aligned
        assert_eq!(c.as_ptr() as usize % 4, 0);

        let d = arena.try_alloc_slice(0i64, 1)?; // 8-byte aligned
        assert_eq!(d.as_ptr() as usize % 8, 0);
        Ok(())
    }

    #[test]
    fn test_try_arena() -> Result<()> {
        let mut arena = Arena::new(1024)?;

        assert_eq!(arena.remaining(), 1024, "Arena should report 1024 bytes");

        let u8_ptr: &mut u8 = arena.alloc(41u8);
        assert_eq!(arena.remaining(), 1023, "Arena should report 1023 bytes");

        let u32_ptr: &mut u32 = arena.alloc(42u32);
        assert_eq!(
            u32_ptr as *mut u32 as usize % 4,
            0,
            "Pointer should be aligned"
        );

        *u32_ptr += *u8_ptr as u32;

        let u8_ptr: &mut [u8] = arena.alloc_slice(43u8, 5);
        assert_eq!(u8_ptr, vec![43u8, 43, 43, 43, 43]);

        arena.reset();

        assert_eq!(arena.remaining(), 1024, "Arena should report 1024 bytes");

        let _u8_ptr: &mut u8 = arena.alloc(44u8);
        assert_eq!(arena.remaining(), 1023, "Arena should report 1023 bytes");

        let u64_ptr = arena.alloc_slice(0u64, 4);
        assert_eq!(
            u64_ptr.as_ptr() as usize % 8,
            0,
            "u64 pointer should be 8-byte aligned"
        );

        let _u8_ptr: &mut u8 = arena.alloc(46u8);

        let u128_ptr = arena.alloc_slice(0u128, 5);
        assert_eq!(
            u128_ptr.as_ptr() as usize % 16,
            0,
            "u128 pointer should be 8-byte aligned"
        );

        // try_ testing:
        arena.reset();
        let u8_ptr: &mut u8 = arena.try_alloc(41u8)?;
        assert_eq!(arena.remaining(), 1023, "Arena should report 1023 bytes");

        let u32_ptr: &mut u32 = arena.try_alloc(42u32)?;
        assert_eq!(
            u32_ptr as *mut u32 as usize % 4,
            0,
            "Pointer should be aligned"
        );

        *u32_ptr += *u8_ptr as u32;

        let u8_ptr: &mut [u8] = arena.try_alloc_slice(43u8, 5)?;
        assert_eq!(u8_ptr, vec![43u8, 43, 43, 43, 43]);

        arena.reset();

        assert_eq!(arena.remaining(), 1024, "Arena should report 1024 bytes");

        let _u8_ptr: &mut u8 = arena.try_alloc(44u8)?;
        assert_eq!(arena.remaining(), 1023, "Arena should report 1023 bytes");

        let st = arena.try_alloc_str("Test")?;
        assert_eq!(arena.remaining(), 1019, "Arena should report 1023 bytes");
        assert_eq!(st, "Test");

        let st = arena.try_alloc_str("")?;
        assert_eq!(arena.remaining(), 1019, "Arena should report 1023 bytes");
        assert_eq!(st, "");

        let u64_ptr = arena.try_alloc_slice(0u64, 4)?;
        assert_eq!(
            u64_ptr.as_ptr() as usize % 8,
            0,
            "u64 pointer should be 8-byte aligned"
        );

        let _u8_ptr: &mut u8 = arena.try_alloc(46u8)?;

        let u128_ptr = arena.try_alloc_slice(0u128, 5)?;
        assert_eq!(
            u128_ptr.as_ptr() as usize % 16,
            0,
            "u128 pointer should be 8-byte aligned"
        );

        Ok(())
    }

    #[test]
    #[should_panic(expected = "Layout Error: invalid parameters to Layout::from_size_align")]
    fn test_failed_to_allocate_panic() {
        let _arena = Arena::new(usize::MAX).unwrap_or_else(|e| panic!("Arena Failed: {}", e));
    }

    #[test]
    fn test_try_failed_to_allocate() -> Result<()> {
        assert!(matches!(Arena::new(usize::MAX), Err(Error::Layout(_))));
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Arena Failed: Out of Memory")]
    fn test_out_of_memory_panic() {
        let arena = Arena::new(1024).unwrap_or_else(|e| panic!("Should work Arena Failed: {}", e));
        let _large_slice: &mut [u64] = arena.alloc_slice(0u64, 150);
    }

    #[test]
    fn test_try_out_of_memory() -> Result<()> {
        let arena = Arena::new(1024)?;
        // Check that alloc_slice returns Err(Error::OutOfMemory)
        assert!(matches!(
            arena.try_alloc_slice(0u64, 150),
            Err(Error::OutOfMemory)
        ));
        Ok(())
    }

    fn format_number(n: u64) -> String {
        let s = n.to_string();
        let mut result = String::new();
        let mut count = 0;

        for c in s.chars().rev() {
            if count == 3 {
                result.push(',');
                count = 0;
            }
            result.push(c);
            count += 1;
        }

        result.chars().rev().collect()
    }

    fn get_system_available_memory() -> u64 {
        let sys = System::new_all();
        sys.available_memory()
    }

    fn get_process_memory_usage() -> u64 {
        let sys = System::new_all();
        let pid = Pid::from(std::process::id() as usize); // Convert to sysinfo's Pid type
        sys.process(pid)
            .map(|process: &sysinfo::Process| process.memory())
            .unwrap_or(0)
    }

    fn test_lg_alloc(size: usize) -> Result<()> {
        let arena = Arena::new(size)?;
        let j = arena.try_alloc_slice(0u8, size)?;
        j.fill(15u8);
        Ok(())
    }

    #[cfg_attr(miri, cfg(miri_skip))]
    #[test]
    fn test_try_large_allocation() -> Result<()> {
        const TARGET_SIZE: usize = 2 * 1024 * 1024 * 1024; // 2GB
        const NUM_ALLOCS: usize = 1000;
        const ALLOC_SIZE: usize = TARGET_SIZE / NUM_ALLOCS;

        //let arena = Arena::new(TARGET_SIZE);
        let start_sys = get_system_available_memory();
        let start_proc = get_process_memory_usage();

        println!(
            "Available memory: {} bytes",
            format_number(get_system_available_memory())
        );
        println!(
            "Process memory usage: {} bytes",
            format_number(get_process_memory_usage())
        );

        for _ in 0..NUM_ALLOCS {
            test_lg_alloc(ALLOC_SIZE).unwrap();
        }

        let used_sys = start_sys.saturating_sub(get_system_available_memory());
        let used_proc = get_process_memory_usage().saturating_sub(start_proc);

        println!(
            "Available memory: {} bytes, used: {}",
            format_number(get_system_available_memory()),
            format_number(used_sys)
        );
        println!(
            "Process memory usage: {} bytes, Additional proc mem used: {}",
            format_number(get_process_memory_usage()),
            format_number(used_proc)
        );

        assert!(used_sys < 50_000_000); // Arbitrary 50 MB Limit
        assert!(used_proc < 10_000_000); // Arbitrary 10 MB Limit
        Ok(())
    }
}
