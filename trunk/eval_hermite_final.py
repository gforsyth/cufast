"""
Evaluate hermite polynomials using cuda
* use modified version of previously written polynomial evaluator
"""
import warnings
warnings.simplefilter('ignore',Warning)

import numpy
import time
from math import sqrt, log10, exp, ceil
import sys

from optparse import OptionParser

from scipy.special import hermite

import pycuda.driver as cuda
#from pycuda.compiler import SourceModule

cuda.init()
dev = cuda.Device(0)
ctx = dev.make_context()


# pad an array a with zeros until it is of desired
# length len
# need to pad at the beginning, as highest power of 
# x is held in a[0]
def pad(a,l):
	a_len = len(a)
	pad_length = l-a_len

	return numpy.append(numpy.zeros(pad_length),a)

# generate an array of hermite polynomials of degree < n
# all padded to length of maximum poly.
def gen_hermite_array(n):
	pad_len = n
	# initial polynomial
	h = pad(hermite(0).coeffs,pad_len)
	for i in range(1,n):
		h_temp = pad(hermite(i).coeffs,pad_len)
		h = numpy.append(h,h_temp)

	return h
	
# pre-compute the hermite polynomials to be used throughout
# the calculations (for speed)
# input - p - number of terms kept in hermite / taylor series
def gen_hermite_polys(p):
	H = []
	for i in range(p):
		H.append(hermite(i))

	return H

def gen_alpha_cuda(p):
    alpha = []
    for i in range(0,p):
        for j in range(0,p-i):
			alpha.append(i)
			alpha.append(j)

    return alpha

def build_fact_cache(q):
	f = []
	for i in range(q+1):
		f.append(factorial(i))

	return f

def factorial(n):
    if n <= 0:
        return 1

    return n*factorial(n-1)

def gen_alpha(p):
    alpha = []
    for i in range(0,p):
        for j in range(0,p-i):
            alpha.append([i,j])

    return alpha

# alpha! = product(alpha[i])
def a_fact(alpha):
    a_fact = factorial(alpha[0])*factorial(alpha[1])
    return a_fact

# get nCp needed for number of coefficients for taylor series
# input - n
#		  p
def nCp(n,p):
	top    = factorial(n)
	bottom = factorial(p)*factorial(n-p)

	return 1.*(top/bottom)

delta = 0.25
p = 9

parser = OptionParser()
parser.add_option("-p",type="int",dest="p",help="Number of terms")
parser.add_option("-n",type="int",dest="n_points",help="number of blocks (n_points / 512)")

(options, args) = parser.parse_args()

p = options.p

alpha = numpy.array(gen_alpha_cuda(p)).astype(numpy.float32)
H = gen_hermite_array(p).astype(numpy.float32)

# general parameters
H_orig = gen_hermite_polys(p)
num_terms = nCp(p+1,2)
num_clusters = 36
A_gen = numpy.random.randn(num_clusters*num_terms).astype(numpy.float32)

f_cache = build_fact_cache(p)

# start off with python generated A_alpha values for one center
blocksize = 512
n_points = options.n_points
n_blocks = options.n_points / blocksize

# computing evaluations from many source clusters
# use same center for all calculations
# uniform target points for speed of generation
sb = numpy.array([0.25,0.25]*num_clusters).astype(numpy.float32)
tx = numpy.ones(n_points).astype(numpy.float32)
ty = numpy.ones(n_points).astype(numpy.float32)

results = numpy.zeros(n_points).astype(numpy.float32)
r_orig = numpy.empty_like(results)

alpha = numpy.array(gen_alpha_cuda(p)).astype(numpy.float32)
alpha_orig = gen_alpha(p)

# setup cuda
r = numpy.zeros(n_points).astype(numpy.float32)
r_out = numpy.empty_like(r)

# assign memory on gpu
r_gpu = cuda.mem_alloc(r.size * r.dtype.itemsize)
tx_gpu = cuda.mem_alloc(tx.size * tx.dtype.itemsize)
ty_gpu = cuda.mem_alloc(ty.size * ty.dtype.itemsize)
alpha_gpu = cuda.mem_alloc(alpha.size * alpha.dtype.itemsize)
H_gpu = cuda.mem_alloc(H.size * H.dtype.itemsize)

# copy memory from host to gpu
cuda.memcpy_htod(r_gpu,r)
cuda.memcpy_htod(tx_gpu,tx)
cuda.memcpy_htod(ty_gpu,ty)
cuda.memcpy_htod(alpha_gpu,alpha)
cuda.memcpy_htod(H_gpu,H)

# decide how many clusters are evaluated per instance
avail_threads = blocksize - len(H) - len(alpha)
threads_per_call = len(alpha)/2 + 2
clusters_per_call = int(avail_threads*1. / threads_per_call)
num_calls_needed = int(ceil(num_clusters*1. / clusters_per_call))

if num_calls_needed == 0:
	num_calls_needed += 1

clusters_called = 0
total_kernel_time = 0.
loop_time = 0.

tic = time.time()

for k in range(num_calls_needed):
	# split the data
	len_A = len(alpha)/2
	A_curr = A_gen[k*clusters_per_call*len_A:k*clusters_per_call*len_A+clusters_per_call*len_A]
	sb_curr = sb[k*2*clusters_per_call:k*2*clusters_per_call+2*clusters_per_call]
	
	# keep track of how much of this call we have evaluated
	clusters_called += clusters_per_call
	if clusters_called <= num_clusters:
		clusters_this_call = clusters_per_call
	else:
		clusters_this_call = num_clusters-clusters_called+clusters_per_call

	# assign cuda memory for current set of evaluations
	A_gpu = cuda.mem_alloc(A_curr.size * A_curr.dtype.itemsize)
	sb_gpu = cuda.mem_alloc(sb_curr.size * sb_curr.dtype.itemsize)
	
	# copy to device
	cuda.memcpy_htod(A_gpu,A_curr)
	cuda.memcpy_htod(sb_gpu,sb_curr)

	# cuda source
	hermite_eval = cuda.SourceModule("""
	#define A_TERMS %(lenA)d
	#define TERMS %(termsAlpha)d
	#define CLUSTERS %(clusters)d
	#define POLY_TERMS %(polyTerms)d
	#define BLOCKSIZE %(blocksize)d
	#define SQRT_2 1.4142135623730951f

	#define SIGMA %(sigma)f
	#define LEN_ALPHA %(len_alpha)d
	#define NUM_TERMS %(num_terms)d
	#define OPTS3 %(opts3)d
	#define NUM_CLUSTERS %(num_clusters)d

	#define DEST_PER_THREAD 2

	
		// slightly optimised evaluation -- do all calculations for source clusters
		// at once -- save on memory bandwidth
		__global__ void eval_hermite2(float *r, float *A, float *tx, float *ty, float *sb,
		 							 float *alpha, float *H)
		{		
			float result, x, y;
			int alpha1;
			float h1, h2;
			int i, k;
		
			float pre_mult, t_x, t_y;
		
			// shared memory
			__shared__ float shared_alpha[TERMS];
			__shared__ float shared_A[A_TERMS];
			__shared__ float shared_sb[CLUSTERS];
			__shared__ float shared_H[POLY_TERMS];
		
			////////////////////////////////
			// Read vars into shared memory
			// WARNING: Each block needs more threads than (TERMS + A_TERMS + POLY_TERMS + CLUSTERS)
			// otherwise it won't work. 
			////////////////////////////////
			// select what each thread reads
			if (threadIdx.x < TERMS){
			// shared_alpha case
				i = 0;
				k = 0;
			} else if (threadIdx.x < TERMS + A_TERMS) {
			// shared_A case
				i = 1;
				k = - TERMS;
			} else if (threadIdx.x < TERMS + A_TERMS + POLY_TERMS) {
			// shared_H case
				i = 2;
				k = - TERMS - A_TERMS;
			} else if (threadIdx.x < TERMS + A_TERMS + POLY_TERMS + CLUSTERS) {
			// shared_sb case
				i = 3;
				k = - TERMS - A_TERMS - POLY_TERMS;
			} else {
			// No read case
				i = 4;
				k = 0;
			}
			// diverge the threads to have independent reads
			switch (i){
				case 0:
					shared_alpha[threadIdx.x + k] = alpha[threadIdx.x + k];
					break;
				case 1:
					shared_A[threadIdx.x + k] = A[threadIdx.x + k];
					break;
				case 2:
					shared_H[threadIdx.x + k] = H[threadIdx.x + k];
					break;
				case 3:
					shared_sb[threadIdx.x + k] = sb[threadIdx.x + k];
					break;
				default:
					break;
			}
		
			//__threadfence_block();
			__syncthreads();

			if (OPTS3 < threadIdx.x + BLOCKSIZE*blockIdx.x)
			{
				return;
			}

			t_x = tx[threadIdx.x + BLOCKSIZE*blockIdx.x];
			t_y = ty[threadIdx.x + BLOCKSIZE*blockIdx.x];
			result = 0.0;
		
			///////////////////////////////
			// Main loop, flops: (NumClusters * (19 + LenAlpha/2 * (14 + 4 * NumTerms)) + 2)
			///////////////////////////////
		
			// run through this code for each cluster center
			for (k=0; k < NUM_CLUSTERS; k++) {

				// distance operator
				x = (t_x - shared_sb[k*2+0]) / SQRT_2 / SIGMA;
				//x = (t_x - sb[k*2+0]) / SQRT_2 / SIGMA;
				y = (t_y - shared_sb[k*2+1]) / SQRT_2 / SIGMA;
				//y = (t_y - sb[k*2+1]) / SQRT_2 / SIGMA;

				pre_mult = exp(-(x*x))*exp(-(y*y));

				// look at shared memory - all variables called in
				// poly_eval should be in shared memory

				for (i=0; i < LEN_ALPHA/2; i++)
				{
					alpha1 = shared_alpha[i*2];
				
					// I avoid the inner loop and get a superb speedup, but it needs to be hardcoded
					// is it possible to do the same using MACROS? or generating this from python?
					// ONLY USE p=5 here
					h1 = h2 = 0.0f;
					h1 = shared_H[NUM_TERMS*alpha1 + 0] + x*h1;
					h1 = shared_H[NUM_TERMS*alpha1 + 1] + x*h1;
					h1 = shared_H[NUM_TERMS*alpha1 + 2] + x*h1;
					/*
					h1 = shared_H[NUM_TERMS*alpha1 + 3] + x*h1;
					h1 = shared_H[NUM_TERMS*alpha1 + 4] + x*h1;
					
					h1 = shared_H[NUM_TERMS*alpha1 + 5] + x*h1;
					h1 = shared_H[NUM_TERMS*alpha1 + 6] + x*h1;
					h1 = shared_H[NUM_TERMS*alpha1 + 7] + x*h1;
					h1 = shared_H[NUM_TERMS*alpha1 + 8] + x*h1;
					
					h1 = shared_H[NUM_TERMS*alpha1 + 9] + x*h1;
					h1 = shared_H[NUM_TERMS*alpha1 + 10] + x*h1;
					h1 = shared_H[NUM_TERMS*alpha1 + 11] + x*h1;
					*/
				
					//result += alpha1;
				
					alpha1 = shared_alpha[i*2 + 1];
					h2 = shared_H[NUM_TERMS*alpha1 + 0] + y*h2;
					h2 = shared_H[NUM_TERMS*alpha1 + 1] + y*h2;
					h2 = shared_H[NUM_TERMS*alpha1 + 2] + y*h2;
					
					/*
					h2 = shared_H[NUM_TERMS*alpha1 + 3] + y*h2;
					h2 = shared_H[NUM_TERMS*alpha1 + 4] + y*h2;
					
					h2 = shared_H[NUM_TERMS*alpha1 + 5] + y*h2;
					h2 = shared_H[NUM_TERMS*alpha1 + 6] + y*h2;
					h2 = shared_H[NUM_TERMS*alpha1 + 7] + y*h2;
					h2 = shared_H[NUM_TERMS*alpha1 + 8] + y*h2;
					
					h2 = shared_H[NUM_TERMS*alpha1 + 9] + y*h2;
					h2 = shared_H[NUM_TERMS*alpha1 + 10] + y*h2;
					h2 = shared_H[NUM_TERMS*alpha1 + 11] + y*h2;
					*/
					result += shared_A[k*LEN_ALPHA/2+i]*pre_mult*h1*h2;
				}
			}
			r[threadIdx.x + BLOCKSIZE*blockIdx.x] += result;
		}
	""" % {'lenA':len(A_curr),'termsAlpha':len(alpha),'clusters':len(sb_curr),'polyTerms':len(H), 'blocksize':blocksize,
	'sigma': delta, 'len_alpha': len(alpha), 'num_terms': p, 'opts3': len(tx), 'num_clusters':clusters_this_call},
	nvcc="nvcc",options=['-use_fast_math'], keep=False, no_extern_c=False)

	start_time = time.time()
	func = hermite_eval.get_function("eval_hermite2")
	kernel_time = func(r_gpu,A_gpu,tx_gpu,ty_gpu,sb_gpu,alpha_gpu,H_gpu,block=(blocksize,1,1),grid=(n_blocks,1),time_kernel=True)
	total_kernel_time += kernel_time

	cuda_time = time.time() - start_time
	A_gpu.free()
	sb_gpu.free()

cuda.memcpy_dtoh(r_out,r_gpu)

toc = time.time()
loop_time = toc-tic

# FLOPS
# p = truncation level, alpha = f(p)
# n_points, num_clusters defined in program
flops = n_points*(5+num_clusters*(17+len(alpha)/2*(10+8*p))) / 10**9

print '# points\tKernel time (s)\tParticles/s\tGFLOPS'
print '%d\t%f\t%f\t%f' %(n_points, total_kernel_time, n_points / total_kernel_time, flops / total_kernel_time)


ctx.pop()
