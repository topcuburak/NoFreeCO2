/* A6 gem5 guest -- a static binary that faults in N GB and then loops forever over it.
 * Run INSIDE gem5 SE mode; its page touches make the gem5 HOST process allocate ~N GB of
 * backing store, giving a large-RSS criu target (the gem5 simulator process itself).
 *
 *   gcc -O2 -static bigmem.c -o bigmem      # static: gem5 SE has no dynamic linker
 *   ./bigmem <GB> [passes]                  # default 32 GB; passes=0 (default) loops forever,
 *                                           # passes>0 does K passes then exits (job baseline)
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

int main(int argc, char **argv) {
    size_t gb     = (argc > 1) ? strtoull(argv[1], 0, 10) : 32;
    long   passes = (argc > 2) ? strtol(argv[2], 0, 10) : 0;   /* 0 = infinite (criu target) */
    size_t n  = gb * (1ULL << 30);
    volatile uint8_t *p = (volatile uint8_t *)malloc(n);
    if (!p) { perror("malloc"); return 1; }
    for (size_t i = 0; i < n; i += 4096) p[i] = (uint8_t)i;    /* fault in every page */
    printf("bigmem: %zu GB resident%s\n", gb, passes ? ", running fixed passes" : ", looping");
    fflush(stdout);
    uint64_t s = 0;
    for (long it = 0; passes == 0 || it < passes; it++)        /* infinite, or K passes then exit */
        for (size_t i = 0; i < n; i += 4096) { s += p[i]; p[i]++; }
    return (int)s;
}
