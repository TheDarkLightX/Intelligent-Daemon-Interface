.text
.globl zeno_kernel
.type zeno_kernel,@function
zeno_kernel:
.byte 0x48,0x31,0xc0,0xc3
.size zeno_kernel,.-zeno_kernel
.section .note.GNU-stack,"",@progbits
