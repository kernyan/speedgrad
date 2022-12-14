inline float get_arg_0(__global const float *x,int gid,int subidx)
{
    int idx = gid + subidx;
    return x[idx];
}

inline float get_arg_1(__global const float *x,int gid,int subidx)
{
    int idx = gid + subidx;
    return x[idx];
}

__kernel void reduce_0(__global float* restrict output,
                       __global const float *arg_0_g,
                       __global const float *arg_1_g)
{
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    float acc = 0.0;
    int gid = get_global_id(0);
    for (int subidx = 0; subidx < 10; subidx++) {
        float arg_0 = get_arg_0(arg_0_g, gid, subidx);
        float arg_1 = get_arg_1(arg_1_g, gid, subidx);
        acc = (acc + (arg_0*arg_1));
    }
    {
      output[gid] = acc;
    }
}

