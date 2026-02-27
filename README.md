# Arena
A small, super fast, lock-free arena bump allocator that supports a variety of types, structs, slices, or read-only &amp;strs.

## What is an Arena?

An arena allocator is a memory allocation strategy that pre-allocates a large block
of memory and then hands out sub-allocations from that block sequentially.
Arenas are much faster than normal memory allocations because:

  - **Bulk Allocation**: The entire arena is allocated once
  - **No Fragmentation**: Memory is allocated sequentially
  - **Fast Bookkeeping**: No complex tracking/reallocations
  - **Simplified Deallocation**: The entire arena is freed at once

Arena is special in that, because of its design, it is not subject
to the same use after free or overlapped memory bugs that are possible with
some other bump allocators.

## Tradeoffs

  - The entire arena will be dropped in a single operation. Individual Drop
    operations will not be performed on the Arena's contents. This then will
    leak any memory separately allocated as with Strings and Vecs.

  - **No item Reclamation**: Any unused allocations are stuck until
    the whole arena is dropped or reset().

  - Individual Items are NOT DROPPED
    You need to individually Box<T> any items, (Strings, Vecs, Fat Pointers,
    file handles, etc) to avoid leaking memory.

  - **Fixed Size**: The arena has a set fixed size that doesn't grow.

# MIRI to the Rescue

In some limited testing, MIRI successfully detected the most common forms of
memory leaks. Please test your code with Miri.

```ignore
  cargo +nightly miri run
```


# Use Cases:

 - **Long-lived Data**: Perform one alloc from the system, and break that into
   all the allocations your need for the life of your program

 - **Short-lived Processing**: Temporary allocations for a process... encoding,
   parsing, compiling, translation, etc. When the function returns, all the
   memory is returned in a single free.

 - **Saving Space**: Many system allocation schemes allocate more memory than necessary
   so freed memory can be more efficiently managed for reallocation. Arena fills
   every byte it can excepting alignment requirements.


## Quick Start
```rust
  use arena::Arena;
  fn main() {
    let mut arena = Arena::new(1024).expect("Failed to allocate memory");
    let num = arena.try_alloc::<u32>(42).expect("Arena Out of Memory") ;
    let stnum = format!("Num: {}", *num);
    let s = arena.try_alloc_str(&stnum).expect("Arena Out of Memory");
    println!("{}  {:?}", *num, s);
  }
```

# Design Choices

There are hundreds of possible improvements...  A lot of them very
useful...

 - Chunk Size, Statistics, Diagnostics, Memory Trimming, Snapshots - See arena-b
 - Generation Counter and Key reservation - See atomic-arena
 - Growable - See blink-alloc
 - Memory Paging and Arena Growth - See arena-allocator
 - Memory Reclamation from Individual Items - See drop-arena
 - Scoped Allocator, so you can restore memory in stages - See bump-scope
 - Memory Pools - See shared-arena
 - Boxed Allocations or Collections so you CAN use an arena with strings
      and vecs. See Rodeo and Bumpalo
 - Memory Layout Control, Rewinding, Thread-Local memory lakes, etc (See lake)
 - Detect Use after free - See arena-allocator

The goal of this was Simple, Fast, and Multi-threaded, to the degree possible.

# Where NOT to use Arenas:

❌ - Don't do this:
```ignore
     let v = arena.try_alloc("Hello".to_string())?;    <== Still allocates from the heap
```

✅ - Do this instead:
```ignore
     let v = arena.try_alloc("Hello")?;   <==  Arena based READ ONLY str
```

❌ - Don't do this:  
```ignore
     let v = arena.try_alloc(vec![42u32; 10])?;  <== Allocates data on the heap
```

In both cases of the Don't do this, a fat pointer will be stored in the arena,
and memory for the data or string will be allocated and LEAKED on the heap.

## License
MIT

## Credits
Reverse allocations inspired by:
  https://fitzgen.com/2019/11/01/always-bump-downwards.html

