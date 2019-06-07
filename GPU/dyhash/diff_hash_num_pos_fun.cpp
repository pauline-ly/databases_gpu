

注意，n为2 的时候 , 把 198哪边改一下
TYPE table_no = choose_hash(key_in_bucket);
ballot = __ballot(true);
ballot = (ballot >> bitmove) & ((1 << ELEM_NUM) - 1);
其他的时候用这个
TYPE table_no = choose_hash(key_in_bucket);
ballot = __ballot(table_no_hash != table_no);
ballot = (ballot >> bitmove) & ((1 << ELEM_NUM) - 1);



//n=2
// 其实无需选择
__device__ __forceinline__ uint32_t
choose_hash(uint32_t sig) {
    return sig&1;
}

__device__ __forceinline__ uint32_t
pos1(uint32_t pos) {
    return pos;
}

__device__ __forceinline__ uint32_t
pos2(uint32_t pos) {
    return pos^1;
}







//n=3  if 位移版本
__device__ __forceinline__ uint32_t
choose_hash(uint32_t sig) {
    return (get_hash5(sig) % 3);
}

__device__ __forceinline__ uint32_t
pos1(uint32_t pos) {
    if (pos & 1)  // xx1 12
        return 1;
    return 0;
}

__device__ __forceinline__ uint32_t
pos2(uint32_t pos) {
    if (pos & 1)  // xx1 01 02 03: 001 011 101 :00 01 10 :+1  1 2 3
        return 2;
    return (pos >> 1) + 1;  // 01 02
}



// n=3 switch case 版本
__device__ __forceinline__ uint32_t
choose_hash(uint32_t sig) {
    return (get_hash5(sig) % 3);
}

__device__ __forceinline__ uint32_t
pos1(uint32_t pos) {
    switch (pos){
        case 0,1:  // 01 02 
            return 0;
        case 2:    // 12
            return 1;
    }
}

__device__ __forceinline__ uint32_t
pos2(uint32_t pos) {
    switch (pos){
        case 0,1:
            return pos+1;
        case 2:
            return 2;
    }
}





// n=4
__device__ __forceinline__ uint32_t
choose_hash(uint32_t sig) {
    return (get_hash5(sig) % 6);
}


// xx1 01 02 03 x00 12 13 x10 23
__device__ __forceinline__ uint32_t
pos1(uint32_t pos) {
    if (pos & 1)  // xx1 01 02 03
        return 0;
    if (pos & 2)  // 23
        return 2;
    return 1;  // 12 13
}

__device__ __forceinline__ uint32_t
pos2(uint32_t pos) {
    if (pos & 1)  // xx1 01 02 03: 001 011 101 :00 01 10 :+1  1 2 3
        return (pos >> 1) + 1;
    if (pos & 2)  // 23
        return 3;
    return (pos >> 2) + 2;  // 12 13
}





// n=5
__device__ __forceinline__ uint32_t
choose_hash(uint32_t sig) {
    return (get_hash5(sig) % 10);
}


__device__ __forceinline__ uint32_t
pos1(uint32_t pos) {
	if(pos == 9) 
		return 3;
    if (pos & 1)  // xx1 01 02 03 04
        return 0;
    if (pos & 2)  // 23 24
        return 2;
    return 1;  // 12 13 14
}

__device__ __forceinline__ uint32_t
pos2(uint32_t pos) {
	if(pos == 9) 
		return 4;
    if (pos & 1)  // xx1 01 02 03 04
        return (pos >> 1) + 1;
    if (pos & 2)  // 23 24
        return (pos >> 2) + 3;
    return (pos >> 2) + 2;  // 12 13 14
}





// n=6
__device__ __forceinline__ uint32_t
choose_hash(uint32_t sig) {
    return (get_hash5(sig) % 15);
}

__device__ __forceinline__ uint32_t
pos1(uint32_t pos) {
	switch (pos){
		case 0,1,2,3,4:
			return 0;
		case 5,6,7,8:
			return 1;
		case 9,10,11:
			return 2;
		case 12,13:
			return 3;
		case 14:
			return 4;
	}
}

__device__ __forceinline__ uint32_t
pos2(uint32_t pos) {
	switch (pos){
		case 0,1,2,3,4:
			return pos+1;
		case 5,6,7,8:
			return pos-3;
		case 9,10,11:
			return pos-6;
		case 12,13:
			return pos-8;
		case 14:
			return 5;
	}
}




