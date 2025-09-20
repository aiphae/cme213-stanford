#ifndef PARALLEL_RADIX_SORT
#define PARALLEL_RADIX_SORT

#include <algorithm>
#include <vector>
#include "test_util.h"

constexpr const uint kSizeTestVector = 4000000;
constexpr const uint kNumBits = 16; // must be a divider of 32 for this program to work
constexpr const uint kRandMax = 1 << 31;

/* Function: computeBlockHistograms
 * --------------------------------
 * Splits keys into numBlocks and computes a histogram with numBuckets buckets
 * Remember that numBuckets and numBits are related; same for blockSize and numBlocks.
 * Should work in parallel.
 */
std::vector<uint> computeBlockHistograms(
    const std::vector<uint> &keys,
    uint numBlocks,
    uint numBuckets,
    uint numBits,
    uint startBit,
    uint blockSize
) {
    std::vector<uint> blockHistograms(numBlocks * numBuckets, 0);

    #pragma omp parallel for shared(blockHistograms)
    for (uint block = 0; block < numBlocks; ++block) {
        uint start = block * blockSize;
        uint end = std::min(start + blockSize, static_cast<uint>(keys.size()));
        for (uint i = start; i < end; ++i) {
            uint key = keys[i];
            uint bucket = (key >> startBit) & ((1 << numBits) - 1);
            blockHistograms[block * numBuckets + bucket]++;
        }
    }

    return blockHistograms;
}

/* Function: reduceLocalHistoToGlobal
 * ----------------------------------
 * Takes as input the local histogram of size numBuckets * numBlocks and "merges"
 * them into a global histogram of size numBuckets.
 */
std::vector<uint> reduceLocalHistoToGlobal(
    const std::vector<uint> &blockHistograms,
    uint numBlocks,
    uint numBuckets
) {
    std::vector<uint> globalHisto(numBuckets, 0);

    for (uint block = 0; block < numBlocks; ++block) {
        for (uint bucket = 0; bucket < numBuckets; ++bucket) {
            globalHisto[bucket] += blockHistograms[block * numBuckets + bucket];
        }
    }

    return globalHisto;
}

/* Function: scanGlobalHisto
 * -------------------------
 * This function should simply scan the global histogram.
 */
std::vector<uint> scanGlobalHisto(
    const std::vector<uint> &globalHisto,
    uint numBuckets
) {
    std::vector<uint> globalHistoExScan(numBuckets, 0);

    uint sum = 0;
    for (uint bucket = 0; bucket < numBuckets; ++bucket) {
        globalHistoExScan[bucket] = sum;
        sum += globalHisto[bucket];
    }

    return globalHistoExScan;
}

/* Function: computeBlockExScanFromGlobalHisto
 * -------------------------------------------
 * Takes as input the globalHistoExScan that contains the global histogram after the scan
 * and the local histogram in blockHistograms. Returns a local histogram that will be used
 * to populate the sorted array.
 */
std::vector<uint> computeBlockExScanFromGlobalHisto(
    uint numBuckets,
    uint numBlocks,
    const std::vector<uint> &globalHistoExScan,
    const std::vector<uint> &blockHistograms
) {
    std::vector<uint> blockExScan(numBuckets * numBlocks, 0);

    #pragma omp parallel for
    for (uint bucket = 0; bucket < numBuckets; ++bucket) {
        uint sum = globalHistoExScan[bucket];
        for (uint block = 0; block < numBlocks; ++block) {
            blockExScan[block * numBuckets + bucket] = sum;
            sum += blockHistograms[block * numBuckets + bucket];
        }
    }

    return blockExScan;
}

/* Function: populateOutputFromBlockExScan
 * ---------------------------------------
 * Takes as input the blockExScan produced by the splitting of the global histogram
 * into blocks and populates the vector sorted.
 */
void populateOutputFromBlockExScan(
    const std::vector<uint> &blockExScan,
    uint numBlocks,
    uint numBuckets,
    uint startBit,
    uint numBits,
    uint blockSize,
    const std::vector<uint> &keys,
    std::vector<uint> &sorted
) {
    std::vector<uint> localBlockExScan = blockExScan;

    #pragma omp parallel for
    for (uint block = 0; block < numBlocks; ++block) {
        uint start = block * blockSize;
        uint end = std::min(start + blockSize, (uint)keys.size());

        for (uint i = start; i < end; ++i) {
            uint key = keys[i];
            uint bucket = (key >> startBit) & ((1 << numBits) - 1);

            uint pos = localBlockExScan[block * numBuckets + bucket];
            sorted[pos] = key;
            localBlockExScan[block * numBuckets + bucket]++;
        }
    }
}

/* Function: radixSortParallelPass
 * -------------------------------
 * A pass of radixSort on numBits starting after startBit.
 */
void radixSortParallelPass(
    std::vector<uint> &keys,
    std::vector<uint> &sorted,
    uint numBits,
    uint startBit,
    uint blockSize
) {
    uint numBuckets = 1 << numBits;

    // Choose numBlocks so that numBlocks * blockSize is always greater than keys.size().
    uint numBlocks = (keys.size() + blockSize - 1) / blockSize;

    // go over each block and compute its local histogram
    std::vector<uint> blockHistograms = computeBlockHistograms(keys, numBlocks,
                                        numBuckets, numBits, startBit, blockSize);

    // first reduce all the local histograms into a global one
    std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms,
                                    numBlocks, numBuckets);

    // now we scan this global histogram
    std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);

    // now we do a local histogram in each block and add in the global value to get global position
    std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets,
                                    numBlocks, globalHistoExScan, blockHistograms);

    // populate the sorted vector
    populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, startBit,
                                  numBits, blockSize, keys, sorted);
}

int radixSortParallel(
        std::vector<uint>& keys,
        std::vector<uint>& keys_tmp,
        uint numBits,
        uint numBlocks
) {
    uint blockSize = keys.size() / numBlocks;

    for (uint startBit = 0; startBit < 32; startBit += 2 * numBits) {
        radixSortParallelPass(keys, keys_tmp, numBits, startBit, blockSize);
        radixSortParallelPass(keys_tmp, keys, numBits, startBit + numBits, blockSize);
    }

    return 0;
}

void runBenchmark(std::vector<uint>& keys_parallel,
                  std::vector<uint>& temp_keys,
                  uint kNumBits,
                  uint n_blocks,
                  uint n_threads
){
    omp_set_num_threads(n_threads);

    double startRadixParallel = omp_get_wtime();
    radixSortParallel(keys_parallel, temp_keys, kNumBits, n_blocks);

    double endRadixParallel = omp_get_wtime();
    printf("%8.3f", endRadixParallel - startRadixParallel);
}

#endif