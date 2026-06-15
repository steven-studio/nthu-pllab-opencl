__kernel void relu(__global float* x,
                   __global float* out,
                   const int total) {
    int gid = get_global_id(0);
    if (gid < total) {
        out[gid] = fmax(x[gid], 0.0f);
    }
}

__kernel void bn_relu6_fused(
    __global const float* x,
    __global const float* mean,
    __global const float* std,
    __global const float* gamma,
    __global const float* beta,
    __global float* out,
    const int HW,
    const int C)
{
    int gid = get_global_id(0);
    int c = gid / HW;

    float v = x[gid];
    v = (v - mean[c]) / std[c];   // 除，不是乘
    v = v * gamma[c] + beta[c];
    v = fmax(0.0f, fmin(6.0f, v));
    out[gid] = v;
}

__kernel void add_same_shape(__global const float* a,
                             __global const float* b,
                             __global float* out,
                             const int total)
{
    int gid = get_global_id(0);
    if (gid < total) {
        out[gid] = a[gid] + b[gid];
    }
}

__kernel void add_bias_4d(__global const float* x,
                          __global const float* bias,
                          __global float* out,
                          const int N,
                          const int C,
                          const int H,
                          const int W)
{
    int gid = get_global_id(0);
    int total = N * C * H * W;

    if (gid < total) {
        int c = (gid / (H * W)) % C;
        out[gid] = x[gid] + bias[c];
    }
}

__kernel void batch_normalization(
    __global const float* x,
    __global const float* mean,
    __global const float* variance,
    __global const float* offset,
    __global const float* scale,
    __global float* out,
    const int n,
    const int c,
    const int h,
    const int w,
    const float epsilon,
    const int total)
{
    int gid = get_global_id(0);
    if (gid >= total) return;

    int hw = h * w;
    int c_idx = (gid / hw) % c;

    float v = x[gid];
    float m = mean[c_idx];
    float var = variance[c_idx];
    float off = offset[c_idx];
    float sc = scale[c_idx];

    out[gid] = sc * ((v - m) / sqrt(var + epsilon)) + off;
}

__kernel void bn_sqrt_kernel(__global const float* x,
                             __global float* out,
                             const int total,
                             const float epsilon)
{
    int gid = get_global_id(0);
    if (gid < total) {
        out[gid] = sqrt(x[gid] + epsilon);
    }
}

__kernel void softmax_exp(__global const float* x,
                          __global float* out,
                          const int total)
{
    int gid = get_global_id(0);
    if (gid < total) {
        out[gid] = exp(x[gid]);
    }
}

__kernel void softmax_div(__global float* data,
                          const float sum,
                          const int total)
{
    int gid = get_global_id(0);
    if (gid < total) {
        data[gid] = data[gid] / sum;
    }
}

__kernel void matmul_kernel(__global const float* x,
                            __global const float* y,
                            __global float* out,
                            const int m,
                            const int p,
                            const int n)
{
    int j = get_global_id(0);  // column
    int i = get_global_id(1);  // row

    if (i < m && j < n) {
        float sum = 0.0f;

        for (int k = 0; k < p; k++) {
            sum += x[i * p + k] * y[k * n + j];
        }

        out[i * n + j] = sum;
    }
}

__kernel void convxbias_kernel(
    __global const float* in_x,
    __global const float* filter,
    __global float* out,
    const float bias,
    const int in_n,
    const int in_c,
    const int in_h,
    const int in_w,
    const int filter_n,
    const int filter_c,
    const int filter_h,
    const int filter_w,
    const int out_c,
    const int out_h,
    const int out_w,
    const int stride,
    const int pad)
{
    int gid = get_global_id(0);
    int total = in_n * out_c * out_h * out_w;
    if (gid >= total) return;

    int w = gid % out_w;
    int h = (gid / out_w) % out_h;
    int c = (gid / (out_w * out_h)) % out_c;
    int n = gid / (out_w * out_h * out_c);

    float sum = 0.0f;

    for (int z = 0; z < filter_c; z++) {
        for (int y = 0; y < filter_h; y++) {
            for (int x = 0; x < filter_w; x++) {
                int in_h_idx = h * stride + y - pad;
                int in_w_idx = w * stride + x - pad;

                if (in_h_idx >= 0 && in_h_idx < in_h &&
                    in_w_idx >= 0 && in_w_idx < in_w) {

                    int input_index =
                        n * in_c * in_h * in_w +
                        z * in_h * in_w +
                        in_h_idx * in_w +
                        in_w_idx;

                    int filter_index =
                        c * filter_c * filter_h * filter_w +
                        z * filter_h * filter_w +
                        y * filter_w +
                        x;

                    sum += in_x[input_index] * filter[filter_index];
                }
            }
        }
    }

    out[gid] = sum + bias;
}


#define TS 16

__kernel void matmul_tiled_kernel(__global const float* A,
                                  __global const float* B,
                                  __global float* C,
                                  const int M,
                                  const int P,
                                  const int N)
{
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    int global_col = get_global_id(0);   // j
    int global_row = get_global_id(1);   // i

    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    float sum = 0.0f;

    int numTiles = (P + TS - 1) / TS;

    for (int t = 0; t < numTiles; t++) {
        int tiled_col_A = t * TS + local_col;
        int tiled_row_B = t * TS + local_row;

        // Load tile of A into local memory
        if (global_row < M && tiled_col_A < P)
            Asub[local_row][local_col] = A[global_row * P + tiled_col_A];
        else
            Asub[local_row][local_col] = 0.0f;

        // Load tile of B into local memory
        if (tiled_row_B < P && global_col < N)
            Bsub[local_row][local_col] = B[tiled_row_B * N + global_col];
        else
            Bsub[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}

__kernel void matmul_vec_kernel(__global const float* A,
                                __global const float* B,
                                __global float* C,
                                const int M,
                                const int P,
                                const int N)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    int k = 0;
    for (; k + 3 < P; k += 4) {
        float4 a4 = vload4(0, &A[row * P + k]);

        float4 b4 = (float4)(
            B[(k + 0) * N + col],
            B[(k + 1) * N + col],
            B[(k + 2) * N + col],
            B[(k + 3) * N + col]
        );

        sum += dot(a4, b4);
    }

    for (; k < P; k++) {
        sum += A[row * P + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

__kernel void matmul_tiled_vec_kernel(__global const float* A,
                                  __global const float* B,
                                  __global float* C,
                                  const int M,
                                  const int P,
                                  const int N)
{
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    int global_col = get_global_id(0);   // j
    int global_row = get_global_id(1);   // i

    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    float sum = 0.0f;

    int numTiles = (P + TS - 1) / TS;

    for (int t = 0; t < numTiles; t++) {
        int tiled_col_A = t * TS + local_col;
        int tiled_row_B = t * TS + local_row;

        // Load tile of A into local memory
        if (global_row < M && tiled_col_A < P)
            Asub[local_row][local_col] = A[global_row * P + tiled_col_A];
        else
            Asub[local_row][local_col] = 0.0f;

        // Load tile of B into local memory
        if (tiled_row_B < P && global_col < N)
            Bsub[local_row][local_col] = B[tiled_row_B * N + global_col];
        else
            Bsub[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k += 4) {
            float4 a4 = vload4(0, &Asub[local_row][k]);

            float4 b4 = (float4)(
                Bsub[k + 0][local_col],
                Bsub[k + 1][local_col],
                Bsub[k + 2][local_col],
                Bsub[k + 3][local_col]
            );

            sum += dot(a4, b4);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}