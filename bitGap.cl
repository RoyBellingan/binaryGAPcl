__kernel void computeGap(__global uint* longest4num, __global uint* longestGap) {

	size_t g1   = get_global_id(0);
	uint   tMax = 0;
	//This needs to be aligned with global_size
	const uint batchSize = 1 << 20;
	uint       from      = g1 * batchSize;
	uint       to        = from + batchSize;

	for (uint var = from; var < to; ++var) {
		uint m   = 0;
		uint cur = 0;
		//used to mark after we find the first set bit, to skip the initial 0 sequence
		bool primed = false;
		//start scanning from MSB (most significant bit)
		//uint msb = log2((float)var);
		//for (int i = msb + 1; i >= 0; i--) {
		for (int i = 31; i >= 0; i--) {
			uint mask = 1 << i;
			if (!(var & mask)) {
				if (primed) {
					//str.append("0");
					cur++;
				}
			} else {
				m = max(m, cur);
				//str.append("1");
				primed = true;
				cur    = 0;
			}
		}

		if (m > tMax) {
			//printf("%u\n", g1);
			tMax            = m;
			longestGap[g1]  = m;
			longest4num[g1] = var;
		}
	}
}
/*
function solution(N) {
	uint tMax = 0;
	bool primed = false;
	for (int i = 31; i >= 0; i--) {
		uint mask = 1 << i;
		if (!(N & mask)) {
			if (primed) {
				cur++;
			}
		} else {
			m = max(m, cur);
			primed = true;
			cur    = 0;
		}
	}
	if (m > tMax) {
		tMax = m;
	}
	return tMax;
}
*/
