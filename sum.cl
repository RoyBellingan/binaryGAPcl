//https://stackoverflow.com/a/16130111
float rand(uint *state, int g1){

    uint x = state[0];
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state[0] = x;
    float t1 = x;
    float max = 4294967295;
    //return t1;
    return t1 / max;

	// Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs"
//	uint s, t = state[3];
//	t ^= t >> 2;
//	t ^= t << 1;
//	state[3] = state[2];
//	state[2] = state[1];
//	state[1] = s = state[0];
//	t ^= s;
//	t ^= s << 4;
//	state[0] = t;
//	//t + (state[4] += 362437); ???

//	float t1 = t;
//	float max = 4294967295;
//	//return t1;
//	return t1 / max;
	//*/
}

//The event are rare, so we just go for atomic instead of having a local buffer and performing operation at the end
//https://groups.google.com/forum/#!topic/boost-compute/xJS05dkQEJk
//https://stackoverflow.com/q/11364567
__kernel void add_numbers(__global uint* iteration,__global uint* nuclide , __global float2* prob, __global uint* seed ) {

	//usual declaration of variable
	float4 input1, input2, sum_vector;
	uint global_addr, local_addr;

	local_addr = get_local_id(0);
	int g1 = get_global_id(0);

	//Each GPU Thread will need a local state for the random generator
	uint state[5];
	state[0] = g1 + (local_addr + 1 + ( (local_addr * 0xFFCC) << 4)) + ((*seed << 23) * *seed << 3);
	state[1] = state[0] ^ ((*seed << 23) | (*seed << 3));
	state[2] = g1 + g1 ^ (g1 | 5567898) + 0xFF4 + iteration[1];
	state[3] =  5567898 + 0xFF4 * g1 + iteration[0];
	state[4] =  5567898 + 0xFF4 * g1 + local_addr;


	//first thread will compute the remainder
	ulong max = iteration[0];
	if(g1 == 0){//first kernel run the remnant
		max = iteration[1];
	}

	//Radium LOOP
	for(ulong j = 0;
	    j <= max;
	    j++){
		float v = rand(&state,g1);
		if(v  < prob->x){// compare r with probability density function, here p = const.: is there a decay?
			atom_dec(nuclide);
			atom_inc(nuclide + 1);
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(g1 == (24 * 192) - 1){//LAST kernel run the remnant and do the swap!
		atomic_xchg((nuclide + 2),*(nuclide + 1));
		max = atomic_max((nuclide + 2),0) % ((24 * 192) - 1);
		//printf("k: %u - iter %u swapped %u  old %u\n",g1, max, atomic_max((nuclide + 2),0), *(nuclide + 1));
	}
	//all the thread will wait here that the last perform the swap and than restart
	barrier(CLK_GLOBAL_MEM_FENCE);

	if(g1 != (24 * 192) - 1){
		max = atomic_max((nuclide + 2),0) / ((24 * 192) - 1);
	}

	//printf("k: %u - iter %u swapped %u  old %u\n",g1, max, atomic_max((nuclide + 2),0), *(nuclide + 1));
	if(g1 == 1){//avoid spamming
		//printf("k: %u - iter %u swapped %u  old %u\n",g1, max, atomic_max((nuclide + 2),0), *(nuclide + 1));
	}

	//Radon LOOP
	for(ulong j = 0;
	    j < max;
	    j++) {
		float v = rand(&state,g1);
		if(v < prob->y){ //compare r with probability density function, here p = const.: is there a decay?
			atom_dec(nuclide + 1);
		}
	}
}
