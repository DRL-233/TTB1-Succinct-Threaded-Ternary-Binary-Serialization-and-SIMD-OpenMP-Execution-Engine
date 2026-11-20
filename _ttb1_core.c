#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

typedef struct {
    uint32_t start;
    uint32_t end;
    int32_t next_block;
} BasicBlock;

typedef struct {
    uint8_t* ops;
    int32_t* values;
    uint32_t* skip_target;
    BasicBlock* blocks;
    uint32_t n_nodes;
    uint32_t n_blocks;
    uint8_t has_jumps;
} TTB1Program;

TTB1Program* ttb1_compile(const uint8_t* blob, const int32_t* vals) {
    uint32_t n_nodes = *(const uint32_t*)(blob + 4);
    uint32_t n_threads = *(const uint32_t*)(blob + 8);

    uint32_t deg_len = (n_nodes + 4) / 5;
    uint32_t dir_len = (n_nodes + 7) / 8;

    const uint8_t* deg_ptr = blob + 12;
    const uint8_t* dir_ptr = deg_ptr + deg_len;
    const uint16_t* thr_ptr = (const uint16_t*)(dir_ptr + dir_len);

    TTB1Program* prog = (TTB1Program*)malloc(sizeof(TTB1Program));
    prog->ops = (uint8_t*)malloc(n_nodes * sizeof(uint8_t));
    prog->values = (int32_t*)malloc(n_nodes * sizeof(int32_t));
    prog->skip_target = (uint32_t*)malloc(n_nodes * sizeof(uint32_t));
    prog->n_nodes = n_nodes;
    prog->has_jumps = (n_threads > 0) ? 1 : 0;

    uint32_t idx = 0;
    for (uint32_t b = 0; b < deg_len; b++) {
        uint8_t val = deg_ptr[b];
        uint8_t trits[5];
        for (int j = 4; j >= 0; j--) {
            trits[j] = val % 3;
            val /= 3;
        }
        uint32_t rem = n_nodes - idx;
        uint32_t count = (rem < 5) ? rem : 5;
        uint32_t start = 5 - count;
        for (uint32_t k = 0; k < count; k++) {
            uint8_t deg = trits[start + k];
            uint8_t dir = (dir_ptr[idx >> 3] >> (7 - (idx & 7))) & 1;
            prog->ops[idx] = (deg << 1) | dir;
            prog->values[idx] = vals ? vals[idx] : 0;
            prog->skip_target[idx] = idx + 1;
            idx++;
        }
    }

    int32_t* jump_map = (int32_t*)malloc(n_nodes * sizeof(int32_t));
    uint8_t* is_leader = (uint8_t*)calloc(n_nodes + 1, sizeof(uint8_t));
    for (uint32_t i = 0; i < n_nodes; i++) jump_map[i] = -1;
    is_leader[0] = 1;

    for (uint32_t t = 0; t < n_threads; t++) {
        uint16_t src = thr_ptr[t * 2];
        uint16_t dst = thr_ptr[t * 2 + 1];
        if (src < n_nodes && dst <= n_nodes) {
            jump_map[src] = dst;
            prog->skip_target[src] = dst;
            is_leader[dst] = 1;
            if (src + 1 < n_nodes) is_leader[src + 1] = 1;
        }
    }

    uint32_t max_blocks = n_threads * 2 + 2;
    prog->blocks = (BasicBlock*)malloc(max_blocks * sizeof(BasicBlock));
    int32_t* node_to_block = (int32_t*)malloc((n_nodes + 1) * sizeof(int32_t));

    uint32_t b_cnt = 0;
    uint32_t cur = 0;
    while (cur < n_nodes) {
        uint32_t b_start = cur;
        node_to_block[b_start] = b_cnt;
        while (cur < n_nodes && jump_map[cur] == -1 && (cur == b_start || !is_leader[cur])) {
            cur++;
        }
        uint32_t b_end = (cur < n_nodes && jump_map[cur] != -1) ? (cur + 1) : cur;
        prog->blocks[b_cnt].start = b_start;
        prog->blocks[b_cnt].end = b_end;
        b_cnt++;
        cur = b_end;
    }
    node_to_block[n_nodes] = -1;

    for (uint32_t b = 0; b < b_cnt; b++) {
        uint32_t last_node = prog->blocks[b].end - 1;
        int32_t target_node = (jump_map[last_node] != -1) ? jump_map[last_node] : (int32_t)prog->blocks[b].end;
        prog->blocks[b].next_block = (target_node >= (int32_t)n_nodes) ? -1 : node_to_block[target_node];
    }
    prog->n_blocks = b_cnt;

    free(jump_map);
    free(is_leader);
    free(node_to_block);
    return prog;
}

int64_t ttb1_sum_leaves(const TTB1Program* prog) {
    int64_t sum = 0;
    uint32_t n = prog->n_nodes;
    const uint8_t* ops = prog->ops;
    const int32_t* vals = prog->values;

    #pragma omp parallel for simd reduction(+:sum) schedule(static)
    for (uint32_t i = 0; i < n; i++) {
        sum += (ops[i] == 0) ? vals[i] : 0;
    }
    return sum;
}

int64_t ttb1_exec_fused(const TTB1Program* prog) {
    int64_t acc = 0;
    int32_t b_id = 0;
    const uint8_t* ops = prog->ops;

    while (b_id != -1) {
        BasicBlock blk = prog->blocks[b_id];
        int64_t blk_sum = 0;
        #pragma omp simd reduction(+:blk_sum)
        for (uint32_t i = blk.start; i < blk.end; i++) {
            blk_sum += (ops[i] & 1) ? 1 : -1;
        }
        acc += blk_sum;
        b_id = blk.next_block;
    }
    return acc;
}

int64_t ttb1_query_skip(const TTB1Program* prog, int32_t skip_tag) {
    int64_t sum = 0;
    uint32_t pc = 0;
    uint32_t n = prog->n_nodes;
    while (pc < n) {
        if ((prog->ops[pc] & 1) && prog->values[pc] == skip_tag) {
            pc = prog->skip_target[pc];
        } else {
            if (prog->ops[pc] == 0) sum += prog->values[pc];
            pc++;
        }
    }
    return sum;
}

void ttb1_free(TTB1Program* prog) {
    if (!prog) return;
    free(prog->ops);
    free(prog->values);
    free(prog->skip_target);
    free(prog->blocks);
    free(prog);
}

static PyMethodDef TTB1Methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ttb1module = {
    PyModuleDef_HEAD_INIT,
    "_ttb1_core",
    "Native C/OpenMP/SIMD backend for TTB1",
    -1,
    TTB1Methods
};

PyMODINIT_FUNC PyInit__ttb1_core(void) {
    return PyModule_Create(&ttb1module);
}
