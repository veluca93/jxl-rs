# Implementation details of 2d-IDCT and reinterpreting-DCT

First of all, note that we only need to generate Nx2N, NxN, 2NxN transforms for
most sizes. The one exception are 8x32/32x8 IDCTs and 1x4/4x1
reinterpreting-DCTs.

The 4x1/1x4 reinterpreting-DCTs are very small and don't need special
considerations.

Large transforms use the same implementation strategy, but avoid increasing code
size by using size-generic code.

## Code generation
Code is generated with python scripts. The following bash snippet
generates the relevant files:

```bash
for i in 2 4 8 16 32
do
    python3 gen_idct.py $i > src/idct$i.rs
done
for i in 2 4 8 16 32
do
    python3 gen_reinterpreting_dct.py $i > src/reinterpreting_dct$i.rs
done
python3 gen_idct2d.py > src/idct2d.rs
python3 gen_reinterpreting_dct2d.py > src/reinterpreting_dct2d.rs
cargo +nightly fmt
```

## SIMD type selection

For each 1D column transform, we select the widest SIMD vector size that
divides the number of columns of that pass, allowing rectangular transforms to
use wider vectors along their longer dimension.

## DCT/IDCT Implementation
Both the DCT and the IDCT use the same recursive algorithm used in libjxl to
compute a vector worth of DCTs/IDCTs.

## 2d transforms and matrix transposes
`src/transpose.rs` provides size-generic, width-tiered rectangular `(N, M)`
(`transpose`, `transpose_into_strided`) and element-generic in-place square
(`transpose_in_place`, `transpose_in_place_strided`) matrix transposes
(`16x16` on 512-bit vectors; `8x8` on 256-bit vectors; `4x4` on 128-bit
vectors for `f32`, plus arbitrary lane widths for generic element types; scalar
fallback for emulated descriptors, downgrading to narrower hardware descriptors
for smaller tiles and remainders):

- **Square (`N x N`) IDCT**: 1D column IDCTs, followed by in-place tiled
  transpose (transposing diagonal tiles and transposing+swapping off-diagonal
  tile pairs directly in registers), followed by 1D column IDCTs.
- **Wide (`ROWS < COLS`) IDCT**: rectangular `(ROWS, COLS)` transpose into
  temporary storage, 1D column IDCT of length `COLS`, rectangular `(COLS, ROWS)`
  transpose back into `data`, and 1D column IDCT of length `ROWS`.
- **Tall (`ROWS > COLS`) IDCT** (input stored as `COLS x ROWS`): 1D column IDCT
  of length `COLS` in place, rectangular `(COLS, ROWS)` transpose into temporary
  storage, and 1D column IDCT of length `ROWS` writing directly to `data`.
  *(Note: large `>= 64` rectangular IDCTs in `src/idct_large.rs` transpose
  `K x K` strip tiles in place via `do_idct_rowblock` / `do_idct_trh` rather
  than allocating a `32–128 KiB` auxiliary transpose buffer.)*
- **Square and tall (`ROWS >= COLS`) reinterpreting DCT**: 1D column DCT of
  length `ROWS` in place, strided `(ROWS, COLS)` transpose directly into
  `output`, and 1D column DCT of length `COLS` in place on `output` (except
  `32x32`, which transposes in place in contiguous `data` before copying rows to
  `output` to avoid `1024`-byte stride L1D cache set aliasing; similarly
  `4x8`/`8x4` IDCTs downgrade to 128-bit to avoid YMM/XMM store-forwarding
  stalls).
- **Wide (`ROWS < COLS`) reinterpreting DCT**: 1D column DCT of length `ROWS` in
  place, rectangular `(ROWS, COLS)` transpose into temporary storage, 1D column
  DCT of length `COLS` in place, and strided `(COLS, ROWS)` transpose into
  `output`.
