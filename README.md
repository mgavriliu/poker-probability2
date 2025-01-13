# README.md

# Poker Hand Probability Calculator

A high-performance poker hand probability calculator implemented in C++ and CUDA. This project is based on an earlier version (https://github.com/mgavriliu/poker-probability.git) and it explores various optimization techniques for both CPU and GPU implementations. Compared to the original this version achieves as much as 100x performance.

## Features

- Multiple implementations:
  - CPU versions (v1-v2): Multi-threaded implementations using different hand representation techniques
  - GPU versions (v1-v8): CUDA implementations with various optimizations
- Two RNG options: CURAND and Xoroshiro128+
- Detailed statistical analysis including standard deviation in sigmas
- Comprehensive performance comparison between implementations

## Project Structure

```
poker-probability/
├── src/
│   ├── main.cpp               # Application entry point
│   ├── poker.h                # Poker related logic
│   ├── simulation.h/cpp       # CPU simulation base
│   ├── simulation_vx.h        # CPU simulation versions
│   ├── threads.h/cpp          # CPU simple multithreading
│   ├── utils.h                # Various utilities
│   ├── test.h                 # A few useful unit tests
│   ├── test-hand.cpp          # Alternate test-only executable
│   ├── cuda_simrun.cuh/cu     # CUDA simulation base
│   ├── cuda_simkernel_vx.cuh  # CUDA simulation versions
│   ├── cuda_random.cuh        # CUDA random number utilities
│   ├── cuda_utils.cuh         # CUDA random miscellaneous utilities
│   └── cuda_referencedeck.cuh # CUDA reference deck for 8b to 64b hand conversion
├── CMakeLists.txt             # CMake build configuration
└── README.md                  # Project documentation
```

## Build Requirements

- CMake 3.18+
- CUDA Toolkit 11.0+
- C++17 compatible compiler
- Visual Studio 2019+ (Windows) or GCC 9+ (Linux)

## Building the Project

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Usage

```bash
poker-probability [options]

Options:
  -h, --help              Show help message
  -n NUMBER              Number of hands to simulate (default: 100M)
  -b, --blocksize        Global block size for GPU implementations (default: 32)
  -i, --implementation   Comma separated list of versions to run

Implementation Format:
  TTT:vvv[:sssx[:bbb]]  where:
    TTT   = Implementation type (cpu/gpu)
    vvv   = Version number (v1-v8)
    sssx  = Optional scaling factor (e.g., 2x)
    bbb   = Optional (gpu only) block size override

Available Versions:
CPU Implementations:
  cpu:v1  - 8-bit card representation, logical evaluation
  cpu:v2  - 64-bit hand representation, bitmask evaluation

GPU Implementations:
  gpu:v1[c|x] - Basic implementation
  gpu:v2[c|x] - Shared memory optimization
  gpu:v3[c|x] - 4x16-bit hand representation
  gpu:v4[c|x] - Constant memory reference deck
  gpu:v5[c|x] - Enhanced bitmask evaluation
  gpu:v6[c|x] - Shared memory reference deck
  gpu:v7[c|x] - 64-bit unified representation
  gpu:v8[c|x] - Additional shared memory optimizations

Suffix:
  c = CURAND RNG
  x = Xoroshiro128+ RNG
```

### Examples

Run all implementations:
```bash
./poker-probability -i all
```

Compare specific versions:
```bash
./poker-probability -i cpu:v1,gpu:v8x -n 1000000000
```

Run with scaling:
```bash
./poker-probability -i cpu:v1,gpu:v8x:100x -n 1000000000
```

Run with scaling and block size override:
```bash
./poker-probability -i cpu:v1,gpu:v7x:100x,gpu:v8x:100x:128 -b 512 -n 1000000000
```

## Performance

- cpu:v1: >30 million hands/sec (multi-threaded)
- cpu:v2: >700 million hands/sec (multi-threaded)
- GPU: >11 billion hands/sec
- Typical speedup: >100x GPU vs CPU, with similar algorithms
- Scales efficiently up to ~250 trillion hands
- gpu:v8 uses 64 bit counts throughout and can go much beyond that

## Implementation Details

### CPU Versions
- `v1`: Simple implementation using 8-bit card representation, 0..51
- `v2`: Enhanced version using 64-bit card and hand representation. A hand is the sum of its cards.
        Hand type is determined through bitwise operations with fixed masks.

### GPU Versions
- Memory optimizations:
  - Local memory (`v1`)
  - Shared memory for deck storage (`v2-v8`)
  - Constant memory reference deck (`v4-v5`)
- Hand representations:
  - 8-bit cards (`v1-v2`)
  - 4x16-bit version of cpu:v2 (`v3-v6`)
  - 64-bit, same as cpu:v2  (`v7-v8`)
- RNG options:
  - CURAND (versions with 'c' suffix)
  - Xoroshiro128+ (versions with 'x' suffix)
- Global counts are 64 bits thorughout
- On the GPU, local (per-thread) hand counts are 32 bits, for improved performance, with the
  exception of gpu:v8 which uses 64 bits for local counts as well, to allow for insane hand
  counts at the expense of some performance

## Performance Notes

- GPU versions generally outperform CPU versions by a large margin
- Shared memory versions (`v2+`) show significant improvement over basic implementations
- Xoroshiro RNG typically performs better than CURAND, both in terms of error and performance

## Output Format

The program provides detailed statistics including:
- Hand type distribution
- Theoretical vs calculated probabilities
- Statistical deviation (in sigmas)
- Performance metrics
- Comparative analysis for multiple versions

## Output Format

### Comparison: cpu:v1,cpu:v2:20x,gpu:v7x:1000x
```
Started cpu:v1 with 2,000,000,000 hands...

Implementation: cpu:v1
 -- settings: [mult: 1x, threads: 40]
=========================================================================================================
PokerHand Type                     Count     Calculated    Theoretical          Error          Sigma
---------------------------------------------------------------------------------------------------------
Royal Flush                        2,940      0.000147%      0.000154%     -0.000007%      -2.490134
Straight Flush                    27,640      0.001382%      0.001385%     -0.000003%      -0.380900
Four of a Kind                   479,667      0.023983%      0.024010%     -0.000026%      -0.758044
Full House                     2,878,964      0.143948%      0.144058%     -0.000109%      -1.290778
Flush                          3,930,978      0.196549%      0.196540%      0.000009%       0.087673
Straight                       7,853,209      0.392660%      0.392465%      0.000196%       1.399394
Three of a Kind               42,232,975      2.111648%      2.112845%     -0.001197%      -3.722508
Two Pair                      95,071,033      4.753550%      4.753902%     -0.000351%      -0.738597
One Pair                     845,108,687     42.255421%     42.256903%     -0.001482%      -1.341667
High PokerCard             1,002,414,547     50.120711%     50.117739%      0.002972%       2.658163
---------------------------------------------------------------------------------------------------------
Total:                     2,000,000,000
Time: 62.37s
Speed: 32,066,825 hands/s
RMS Error: 0.001124%
=========================================================================================================


Started cpu:v2 with 40,000,000,000 hands...
, threads: 40

Implementation: cpu:v2
 -- settings: [mult: 20x, threads: 40]
=========================================================================================================
PokerHand Type                     Count     Calculated    Theoretical          Error          Sigma
---------------------------------------------------------------------------------------------------------
Royal Flush                       61,544      0.000154%      0.000154%     -0.000000%      -0.076934
Straight Flush                   554,588      0.001386%      0.001385%      0.000001%       0.698865
Four of a Kind                 9,601,891      0.024005%      0.024010%     -0.000005%      -0.629582
Full House                    57,630,047      0.144075%      0.144058%      0.000017%       0.922276
Flush                         78,600,224      0.196501%      0.196540%     -0.000040%      -1.788281
Straight                     156,988,032      0.392470%      0.392465%      0.000005%       0.172390
Three of a Kind              845,129,172      2.112823%      2.112845%     -0.000022%      -0.309788
Two Pair                   1,901,528,879      4.753822%      4.753902%     -0.000080%      -0.747363
One Pair                  16,903,031,262     42.257577%     42.256903%      0.000674%       2.729090
High PokerCard            20,046,875,641     50.117187%     50.117739%     -0.000552%      -2.207625
---------------------------------------------------------------------------------------------------------
Total:                    40,000,000,000
Time: 51.19s
Speed: 781,418,522 hands/s
RMS Error: 0.000277%
=========================================================================================================


Starting gpu:v7 with 2,000,000,000,000 hands...  finished!

Implementation: gpu:v7x
 -- settings: [mult: 10x, RNG: xoro, block size: 128, blocks: 480, threads: 61,440]
=========================================================================================================
PokerHand Type                     Count     Calculated    Theoretical          Error          Sigma
---------------------------------------------------------------------------------------------------------
Royal Flush                    3,077,178      0.000154%      0.000154%     -0.000000%      -0.558033
Straight Flush                27,707,001      0.001385%      0.001385%      0.000000%       0.681608
Four of a Kind               480,220,308      0.024011%      0.024010%      0.000001%       1.269167
Full House                 2,881,079,687      0.144054%      0.144058%     -0.000004%      -1.404073
Flush                      3,930,846,751      0.196542%      0.196540%      0.000002%       0.641801
Straight                   7,849,315,957      0.392465%      0.392465%      0.000001%       0.175081
Three of a Kind           42,256,984,199      2.112847%      2.112845%      0.000002%       0.217447
Two Pair                  95,078,507,898      4.753921%      4.753902%      0.000020%       1.305810
One Pair                 845,139,619,207     42.256944%     42.256903%      0.000041%       1.173418
High PokerCard         1,002,354,403,094     50.117676%     50.117739%     -0.000063%      -1.792787
---------------------------------------------------------------------------------------------------------
Total:                 2,000,000,000,000
Time: 33.73s
Speed: 59,299,796,250 hands/s
RMS Error: 0.000025%
=========================================================================================================


Performance Comparison:
=====================================================================================
Implementation         Time(s)             Hands/s        Speedup      Relative Perf.
-------------------------------------------------------------------------------------
gpu:v7                   33.73      59,299,796,250        1849.26         184,925.74%
cpu:v2                   51.19         781,418,497          24.37           2,436.84%
cpu:v1                   62.37          32,066,815           1.00             100.00%
=====================================================================================
```

## License

MIT License - See LICENSE file for details
