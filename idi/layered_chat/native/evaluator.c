/* Sixty-four independent Boolean inputs per uint64_t lane word.
 * The exported ZAL kernels are rechecked for all u64 pairs in their ISA model.
 * This host supplies storage and sums each lane's class votes independently.
 */
#include <stdint.h>
#include <stddef.h>
#define DECL(N) extern uint64_t gate_##N(uint64_t,uint64_t);
DECL(0) DECL(1) DECL(2) DECL(3) DECL(4) DECL(5) DECL(6) DECL(7)
DECL(8) DECL(9) DECL(10) DECL(11) DECL(12) DECL(13) DECL(14) DECL(15)
static uint64_t (*const gates[16])(uint64_t,uint64_t)={gate_0,gate_1,gate_2,gate_3,gate_4,gate_5,gate_6,gate_7,gate_8,gate_9,gate_10,gate_11,gate_12,gate_13,gate_14,gate_15};
static uint64_t reference(uint32_t f,uint64_t a,uint64_t b){
 return ((0-((uint64_t)(f>>3)&1))&~a&~b)|((0-((uint64_t)(f>>2)&1))&~a&b)|((0-((uint64_t)(f>>1)&1))&a&~b)|((0-((uint64_t)f&1))&a&b);
}
int evaluate_batch(uint64_t *bits,uint32_t inputs,const uint32_t *nodes,uint32_t count,const uint32_t *votes,uint32_t vote_count,const int32_t *bias,int32_t *scores,uint32_t classes,uint32_t lanes,int use_zal){
 if(inputs>65536||count>1000000||classes>1024||lanes>64)return -1;
 for(uint32_t j=0;j<count;j++){uint32_t a=nodes[j*3],b=nodes[j*3+1],f=nodes[j*3+2];if(a>=inputs+j||b>=inputs+j||f>15)return -2;bits[inputs+j]=use_zal?gates[f](bits[a],bits[b]):reference(f,bits[a],bits[b]);}
 for(uint32_t c=0;c<classes;c++)for(uint32_t lane=0;lane<lanes;lane++)scores[lane*classes+c]=bias[c];
 for(uint32_t j=0;j<vote_count;j++){uint32_t v=votes[j*3],p=votes[j*3+1],c=votes[j*3+2];if(v>=inputs+count||p>1||c>=classes)return -3;uint64_t word=bits[v];for(uint32_t lane=0;lane<lanes;lane++)scores[lane*classes+c]+=((word>>lane)&1)^p;}
 return 0;
}
