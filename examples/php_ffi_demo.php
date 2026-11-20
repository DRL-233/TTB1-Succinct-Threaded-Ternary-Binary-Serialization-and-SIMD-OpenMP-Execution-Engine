<?php
declare(strict_types=1);

/**
 * TTB2 v0.2.0 — Native PHP FFI Execution Example
 *
 * Build the shared library first:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC ttb2/ttb2_lib.c -o libttb2.so
 */

$libPath = dirname(__DIR__) . '/libttb2.so';
if (!file_exists($libPath)) {
    fwrite(STDERR, "Shared library not found at {$libPath}. Compile ttb2/ttb2_lib.c first.\n");
    exit(1);
}

$ffi = FFI::cdef("
    typedef struct TTB2Handle TTB2Handle;
    TTB2Handle* ttb2_compile(const uint8_t *buf, size_t len, int64_t skip_sentinel);
    int64_t     ttb2_sum_leaves(const TTB2Handle *h);
    int64_t     ttb2_query_skip(const TTB2Handle *h);
    int64_t     ttb2_broadcast_center(const TTB2Handle *h);
    int64_t     ttb2_jump_sibling(const TTB2Handle *h, uint32_t src_spoke, uint32_t dst_spoke, uint32_t *out_hops);
    int32_t     ttb2_ascent_to_center(const TTB2Handle *h, uint32_t node_idx, uint32_t *out_hub_offset);
    void        ttb2_free(TTB2Handle *h);
", $libPath);

$packetFile = $argv[1] ?? __DIR__ . '/sample.ttb2';
if (!file_exists($packetFile)) {
    fwrite(STDERR, "Pass a valid .ttb2 binary packet path: php examples/php_ffi_demo.php packet.ttb2\n");
    exit(1);
}

$blob = file_get_contents($packetFile);
$handle = $ffi->ttb2_compile($blob, strlen($blob), 999999999);

if ($handle === null) {
    fwrite(STDERR, "Failed to compile TTB2 wire packet.\n");
    exit(1);
}

echo "Total leaf sum (SIMD/OpenMP):        " . $ffi->ttb2_sum_leaves($handle) . PHP_EOL;
echo "Active-only sum (branchless mask):   " . $ffi->ttb2_query_skip($handle) . PHP_EOL;
echo "One-stroke center broadcast:         " . $ffi->ttb2_broadcast_center($handle) . PHP_EOL;

$hops = FFI::new("uint32_t");
$targetVal = $ffi->ttb2_jump_sibling($handle, 0, 2, FFI::addr($hops));
echo "Dynamic hub jump (spoke 0 -> 2):     ({$targetVal}, {$hops->cdata})" . PHP_EOL;

$ffi->ttb2_free($handle);
