'''
Test suite for the two-dimensional 1/r^2 multipole to local GPU kernel.
'''
import pycuda.driver as cuda
from pycuda.driver import SourceModule
import numpy
import time
from support import findInteractionList
from numpy import zeros, array, ones, arange, log10, alltrue, isfinite, sqrt

class cudaKernel:
  precomputed_division = """
    // One block translates one ME to a new location
    #define BLOCKSIZE %(blocksize)d
    #define TERMS %(terms)d
    #define TERMS_C %(terms_c)d
    #define SOURCE (blockIdx.x + gridDim.x * blockIdx.y)
    #define NID threadIdx.x
    #define SIZEIL 27
  
    ///////////////////////////////////////////////////////////////////////////////////
    // One block per source-ME. The block translates the source ME to a group of destinations.
    // 
    // Input:
    // float *MEterms_ref   Container of ME terms
    // float *MEpoints_ref  Container of ME centers
    // float *DestPoint_lst Container of destination points
    // float *LEterms_ref   LE data containers
    ///////////////////////////////////////////////////////////////////////////////////
    __global__ void me2me(float *MEterms_ref, float *MEpoints_ref, float *DestPoint_lst, float *LEterms_ref)
    {
      int m, ME_dest, blockLoop, Nterm;
      float ac, bd;
      float tnm1_r, tnm1_i;
      float comb, MATmn_r, MATmn_i;
      float LE_r, LE_i;
      float tX, tY;
  
      ///////////////////////////////////////////
      // Memory allocation part
      ///////////////////////////////////////////
      __shared__ float MEterms[TERMS_C];         // Common source point
      float srcX = MEpoints_ref[SOURCE*2];
      float srcY = MEpoints_ref[SOURCE*2 +1];
  
      // Multipole expansions
      if (NID < 2*TERMS) MEterms[NID] = MEterms_ref[SOURCE*TERMS_C + NID];
  
      // Destination points
      __shared__ float dest_point_local[2*SIZEIL];
      if (NID < 2*SIZEIL){
        dest_point_local[NID] = DestPoint_lst[2*SOURCE*SIZEIL + NID];
      }
      __syncthreads();
  
      ///////////////////////////////////////////
      // Computing part: M2L translation
      ///////////////////////////////////////////
  
      // Loop over the translations, assign threads for LE coefficients to be computed
      blockLoop = 0;
      while (NID + blockLoop * BLOCKSIZE < SIZEIL * TERMS){
        
        // Choose "destination point and term of the LE" to work on (this avoid the use of module op)
        #pragma unroll
        for(m = 0; m < SIZEIL; m++){
          if (NID + blockLoop * BLOCKSIZE >= m * TERMS){
            ME_dest = m; // Destination point
            Nterm   = (NID + blockLoop * BLOCKSIZE) - m * TERMS; // LE term
          }
        }
  
        // translation distance
        tX = dest_point_local[ME_dest * 2]     - srcX;
        tY = dest_point_local[ME_dest * 2 + 1] - srcY;
  
        // Precompute t^(n+1)
        tnm1_r = tX;
        tnm1_i = tY;
        #pragma unroll
        for (m = 1; m < TERMS; m++){
          if (Nterm >= m){  // tnm1 = tnm1 * t
            ac = tnm1_r;
            bd = tnm1_i;
            tnm1_r = ac * tX - bd * tY;
            tnm1_i = ac * tY + bd * tX;
          }
        }
  
        if (Nterm & 1 == 1) {  // if n is even number, change of sign
          tnm1_r = -tnm1_r;
          tnm1_i = -tnm1_i;
        }
  
        // Initialization for comb(n+m, m)
        comb = 1.0f;
  
        float tx_inv, ty_inv;
        tx_inv = tX / (tX*tX + tY*tY);
        ty_inv = - tY / (tX*tX + tY*tY);
  
        float tnm1_inv_r, tnm1_inv_i;
        tnm1_inv_r = tnm1_r / (tnm1_r * tnm1_r + tnm1_i * tnm1_i);
        tnm1_inv_i = - tnm1_i / (tnm1_r * tnm1_r + tnm1_i * tnm1_i);
  
        // update_complex = MEm_complex * comb * tnm1_inv
        LE_r = MEterms[0] * tnm1_inv_r - MEterms[1] * tnm1_inv_i;
        LE_i = MEterms[0] * tnm1_inv_i + MEterms[1] * tnm1_inv_r;
  
        // Do the dot product (mat_row, ME terms) for m >= 1
        #pragma unroll 
        for (m = 1; m < TERMS; m++){
          float float_m = (float) m;
          comb = (Nterm == 0) ? 1.0f : comb * (Nterm + float_m) / float_m;  // comb (m+n, m) for next term
  
          // update tnm1 with contribution of next term. tnm1 = tnm1 * t
          ac = tnm1_inv_r;
          bd = tnm1_inv_i;
          tnm1_inv_r = ac * tx_inv - bd * ty_inv;
          tnm1_inv_i = ac * ty_inv + bd * tx_inv;
  
          // mat_nm * tnm1_inv
          MATmn_r = comb * tnm1_inv_r;
          MATmn_i = comb * tnm1_inv_i;
  
          // update_complex = MEm_complex * mat_mn
          int tmp_2m = 2 * m;
          LE_r += MEterms[tmp_2m] * MATmn_r - MEterms[tmp_2m +1] * MATmn_i;
          LE_i += MEterms[tmp_2m] * MATmn_i + MEterms[tmp_2m +1] * MATmn_r;
        }
  
        float2 tmp_f2;
        tmp_f2.x = LE_r;
        tmp_f2.y = LE_i;
        int out_offset = (SOURCE*SIZEIL*TERMS_C) + 2*(Nterm + ME_dest * TERMS);
        *((float2*) &LEterms_ref[out_offset]) = tmp_f2;
  
        blockLoop += 1;  // increase loop counter
      }
    }
  """
  
  local_expansion_reduction = """
    // One block translates one ME to a new location
    #define BLOCKSIZE %(blocksize)d
    #define TERMS %(terms)d
    #define TERMS_C %(terms_c)d
    #define SOURCE (blockIdx.x + gridDim.x * blockIdx.y)
    #define NID threadIdx.x
    #define SIZEIL 27
  
  __global__ void localExpansionReduction(float* sourceLE, int* reductionList, float* destinationLE){
    float term;
    int i;
    
    // read inverse interaction list into shared memory
    __shared__ int inverse_interaction_list[SIZEIL];
    if (NID < SIZEIL) inverse_interaction_list[NID] = reductionList[SIZEIL*SOURCE + NID];
    // initialize the threadblock local expansion
    __shared__ float local_expansion[TERMS_C];
    if (NID < TERMS_C) local_expansion[NID] = 0.0f;
    __syncthreads();
    
    // Reduction loop; one local expansion at each loop.
    if (NID < TERMS_C) {
      for (i = 0; i < SIZEIL; i++) {
        if (inverse_interaction_list[i] < 0) break;
        // each thread gets one term
        term = sourceLE[inverse_interaction_list[i]*SIZEIL + NID];
        local_expansion[NID] += term;
      }
      __threadfence_block();
      // copy back to global
      destinationLE[TERMS_C*SOURCE + NID] = local_expansion[NID];
    }
  }
  
  """


class dataset:
  p                = 0
  dim              = 0
  max_level        = 0
  sizeIL           = 0
  num_ME           = 0
  num_sources      = 0
  num_translations = 0
  MEterms_ref      = []
  MEpoints_ref     = []
  Trans_lst        = []
  DestPoint_lst    = []
  DestP_offset     = []
  DestP_length     = []
  LEterms_ref      = []
  LEout_offset     = []
  LEout_length     = []
  LEterms_cpu      = []
  LEreduction      = []
  start_cpu        = 0
  end_cpu          = 0
  n_blocks         = 0
  blocksize        = 0
  time_gpu         = 0
  time_transfer    = 0


def generateTreeTestData(data):
  ''' This function generates a simple case of an FMM dataset for a regular tree decomposition.
  '''
  data.num_ME = 0
  for i in range(2, data.max_level+1):
    data.num_ME += 4**i
  
  data.MEterms_ref   = ones(data.num_ME * 2*data.p)  # Container of ME terms
  data.MEpoints_ref  = zeros(data.num_ME * 2)        # Container of ME centers
  data.LEreduction   = -ones(data.num_ME * 27, int)  # Contain the destination of the LE
  data.LEcounter     = zeros(data.num_ME, int)       # Counts the number of sources per LE
  for i in range(data.num_ME):
    data.MEpoints_ref[data.dim * i]    = 0.0
    data.MEpoints_ref[data.dim * i +1] = 0.0
  
  # Data for the reduction step
  le_counter = 0
  for level_counter in range(2, data.max_level+1):
    for source_counter in range(4**level_counter):
      interaction_list = findInteractionList(source_counter, level_counter)
      for ilist_counter in range(len(interaction_list)):
        box_destination = le_counter + interaction_list[ilist_counter]
        box_counter     = data.LEcounter[box_destination]
        data.LEreduction[27 * box_destination + box_counter] = source_counter
        data.LEcounter[box_destination] = box_counter + 1
    le_counter += 4**level_counter
  
  data.Trans_lst        = arange(data.num_ME)                      # Number of 'the ME' to be translated
  data.num_sources      = len(data.Trans_lst)
  data.num_translations = data.sizeIL * data.num_ME                # Number of translations to be performed
  data.DestPoint_lst    = zeros(data.sizeIL * data.num_ME * data.dim)   # Destination points
  dst_arr1              = array([-1.60, 2.80, 0.1])
  dst_arr2              = array([-1.60, -0.5, -1.40, 0.1, 1.80, 1.60])
  dst_arr3              = array([-1.80, -1.40, 1.40])
  dst_arr4              = array([-1.60, -1.80, 1.60])
  
  for idME in range(data.num_ME):
    offset      = idME * data.sizeIL
    point_index = 0
    # combine point sets (1&2) to form coordinates
    for x in dst_arr1:
      for y in dst_arr2:
        data.DestPoint_lst[data.dim*(offset + point_index)]    = x
        data.DestPoint_lst[data.dim*(offset + point_index) +1] = y
        point_index += 1
    # combine point sets (3&4) to form coordinates
    for x in dst_arr3:
      for y in dst_arr4:
        data.DestPoint_lst[data.dim*(offset + point_index)]    = x
        data.DestPoint_lst[data.dim*(offset + point_index) +1] = y
        point_index += 1
  
  # Translation destination Offset & Size
  offset_lst   = zeros(3*data.num_ME, dtype=int) 
  data.DestP_offset = zeros(data.num_ME, dtype=int) # Translation destination start at offset
  data.DestP_length = zeros(data.num_ME, dtype=int) # Number of translation destinations per ME
  for i in range(data.num_ME):
    data.DestP_offset[i] = i*data.sizeIL
    data.DestP_length[i] = data.sizeIL

  # Destination Output Offset & Size
  data.LEout_offset  = zeros(data.num_ME, dtype=int) # Output of translation starts at offset
  data.LEout_length  = zeros(data.num_ME, dtype=int) # Length of translation output
  for i in range(data.num_ME):
    data.LEout_offset[i] = i*data.sizeIL
    data.LEout_length[i] = data.sizeIL
  data.LEterms_ref = zeros(data.num_translations * data.dim*data.p) # LE data containers


def generateSingleTestData(data):
  ''' One source - One destination case '''
  data.p      = 20
  data.num_ME = 1
  data.dim    = 2

  data.MEterms_ref   = 0.3*ones(data.num_ME * 2*data.p)  # Container of ME terms
  data.MEpoints_ref  = array([0.51, 0.52])   # Container of ME centers
  data.Trans_lst     = array([0])  # Number of 'the ME' to be translated
  data.num_sources   = len(data.Trans_lst)
  data.num_translations = 1           # Number of translations to be performed
  data.DestPoint_lst = array([0.72, 0.74])
  data.DestP_offset  = array([0]) # Translation destination start at offset
  data.DestP_length  = array([1]) # Number of translation destinations per ME
  data.LEout_offset  = array([0]) # Output of translation starts at offset
  data.LEout_length  = array([1]) # Length of translation output
  data.LEterms_ref   = zeros(data.num_translations * 2*data.p) # LE data containers


def flops(ilz, bs, p, num_blocks):
    '''
    Function with estimates for the number of floating point operations for a m2l call.
    
    ilz        Interaction List size
    bs         Blocksize (num threads in a block)
    p          Terms in the expansion
    num_blocks Total number of blocks executed
    '''
    return 1.0*num_blocks * (ilz * p * (35 + ilz*7 + (p-1)*28))


def bandwidth(ilz, p, num_blocks):
    '''
    Computes the number of effective bytes moved by the kernel call
    '''
    return num_blocks * (2*p + 2*ilz + 2*ilz*p + 4) * 4.0
code
def printTable(data):
  print data.p, ' & ',
  print data.num_translations, ' & ',
  print '%.2e & ' % data.time_gpu,
  print '%.2e & ' % data.time_reduction,
  print '%.2e & ' % data.time_transfer_in,
  print '%.2e & ' % data.time_transfer_out,
  print '%.2f & ' % (flops(27, data.blocksize, data.p, data.n_blocks) / 10**9 / data.time_gpu),
  print '%.2f & ' % (bandwidth(27, data.p, data.n_blocks) / 1024**3 / data.time_gpu),
  print '%.2f ' % ((1.0 * data.num_translations) / 10**6 / data.time_gpu), '\\\\'


def printRun(data):
  print '\nNum coefficient: ', data.p, 
  print '\tNum Sources: ', data.num_sources,
  print '\tNum translations: ', data.num_translations,
  print '\tNum Threads: ', data.blocksize, 
  print '\tAll finite: ', alltrue(isfinite(data.LEterms_ref))
  print 'GPU time: %(gpu_time)e' % {'gpu_time' : data.time_gpu},
  print '\tTransfer time: %(transfer_time)e' % {'transfer_time' : data.time_transfer},
  print 'GIGAOP: ', flops(27, data.blocksize, data.p, data.n_blocks) / 10**9,
  print '\tGIGAOP/S: ', flops(27, data.blocksize, data.p, data.n_blocks) / 10**9 / data.time_gpu,
  print '\tEffective Bandwidth [GB/s]: ', bandwidth(27, data.p, data.n_blocks) / 1024**3 / data.time_gpu
  print 'Translations per second (in millions) [MTPS]: ', (1.0 * data.num_translations) / 10**6 / data.time_gpu


def compareCoefficients(data):
  l2_numerator   = 0
  l2_denominator = 0
  for i in range(data.num_translations):
    first_coefficient = True
    print_coefficient = False
    for j in range(2 * data.p):
      coeff_gpu       = data.LEterms_ref[i*2*data.p + j]
      coeff_cpu       = data.LEterms_cpu[i*2*data.p + j]
      coeff_error     = abs((coeff_gpu - coeff_cpu) / coeff_cpu)
      l2_numerator   += coeff_error**2
      l2_denominator += coeff_cpu**2
      no_print_term   = True
      if no_print_term:
        continue
  max_error = max(abs((data.LEterms_ref - data.LEterms_cpu) / data.LEterms_cpu))
  print 'Max relative error: ', max_error
  print 'L2 relative error norm: ', sqrt(l2_numerator / l2_denominator)


def cpuComputeM2L(data):
  ''' M2L translation using python.
  '''
  data.start_cpu= time.time()
  
  m2l = zeros((data.p,data.p), complex)
  
  for source in range(len(data.Trans_lst)):
      ME_src = data.Trans_lst[source] 
      offset = data.DestP_offset[source]
      length = data.DestP_length[source]
      
      # output data
      LE_out = zeros(length * 2*data.p)
      LE_offset = data.LEout_offset[source]
      LE_length = data.LEout_length[source]
      
      # get ME data
      MEterms  = data.MEterms_ref[ME_src*2*data.p:ME_src*2*data.p+2*data.p]
      MEpoints = data.MEpoints_ref[ME_src*2:ME_src*2+2]
      
      # local output offset (translate into local memory)
      out_offset = 0
      for dest in range(length):
          # destination point
          ME_dest   = offset + dest
          destPoint = data.DestPoint_lst[ME_dest*2:ME_dest*2+2]
          
          # translate source
          transDist = destPoint - MEpoints
          t_complex = complex(transDist[0], transDist[1])
          
          # loop over the terms
          for n in range(data.p):
              # precompute t**(n+1)
              taux = t_complex**(n+1)
              vaux = 1. # variable for comb(n+m, m)
              
              # (-1)**n
              if ((n & 1) == 0):
                  taux = vaux*taux
              else:
                  taux = -vaux*taux
                  
              # Do dot product for m = 0
              mat_mn = vaux / taux
              m2l[n][0] = mat_mn
              
              MEm_complex    = MEterms[0] + MEterms[1] * 1j
              update_complex = MEm_complex * mat_mn
              LE_out[out_offset + 2*n + 0] = update_complex.real
              LE_out[out_offset + 2*n + 1] = update_complex.imag
              
              # Do the dot product (mat_row, ME) for all m > 0
              for m in range(1,data.p):
                  # comb(m+n, m)
                  if n == 0:
                      vaux = 1.
                  else:
                      vaux = vaux * (n + m) / m
                  
                  # update t_aux with contribution
                  taux = taux * t_complex
                  
                  # compute element operation
                  mat_mn = vaux / taux 
                  MEm_complex    = MEterms[2*m] + MEterms[2*m + 1] * 1j
                  update_complex = MEm_complex * mat_mn
                  m2l[n][m] = mat_mn
                  
                  # update LE
                  LE_out[out_offset + 2*n]    = LE_out[out_offset + 2*n]    + update_complex.real
                  LE_out[out_offset + 2*n +1] = LE_out[out_offset + 2*n +1] + update_complex.imag
          
          # update local output offset for the next translation
          out_offset += 2*data.p
      
      # save output to global
      LElocal_offset = LE_offset * 2*data.p
      data.LEterms_ref[LE_offset * 2*data.p:(LE_offset + LE_length) * 2*data.p] = LE_out
  #print 'LE terms: ', data.LEterms_ref
  data.LEterms_cpu = data.LEterms_ref.copy()
  data.end_cpu= time.time()


def gpuComputeM2L(data, cuda_kernel_string):
  # Cuda module
  cuda.init()
  
  assert cuda.Device.count() >= 1 # check that we can run
  dev = cuda.Device(0)            # Get device
  ctx = dev.make_context()        # create context
  
  data.LE_inter_ref = zeros(data.num_translations * data.dim*data.p) # LE data containers
  data.LEterms_ref      = zeros(data.num_ME * data.dim*data.p) # LE data containers
  
  # Convert data for the GPU
  data.MEterms_ref   = data.MEterms_ref.astype(numpy.float32)    # Container of ME terms
  data.MEpoints_ref  = data.MEpoints_ref.astype(numpy.float32)   # Container of ME centers
  data.DestPoint_lst = data.DestPoint_lst.astype(numpy.float32)  # Container of destination points
  data.LE_inter_ref  = data.LE_inter_ref.astype(numpy.float32)   # LE intermediate data containers
  data.LEterms_ref   = data.LEterms_ref.astype(numpy.float32)    # LE data containers
  data.LEreduc_ref   = data.LEreduction.astype(numpy.int32)
  
  # Allocate memory in the GPU
  gpu_MEterms_ref   = cuda.mem_alloc(data.MEterms_ref.size   * data.MEterms_ref.dtype.itemsize)
  gpu_MEpoints_ref  = cuda.mem_alloc(data.MEpoints_ref.size  * data.MEpoints_ref.dtype.itemsize)
  gpu_DestPoint_lst = cuda.mem_alloc(data.DestPoint_lst.size * data.DestPoint_lst.dtype.itemsize)
  gpu_LE_inter_ref  = cuda.mem_alloc(data.LE_inter_ref.size  * data.LE_inter_ref.dtype.itemsize)
  gpu_LEterms_ref   = cuda.mem_alloc(data.LEterms_ref.size   * data.LEterms_ref.dtype.itemsize)
  gpu_LEreduc_ref   = cuda.mem_alloc(data.LEreduc_ref.size   * data.LEreduc_ref.dtype.itemsize)
  
  # Transfer memory to device
  start_transfer = time.time()
  cuda.memcpy_htod(gpu_MEterms_ref,   data.MEterms_ref)
  cuda.memcpy_htod(gpu_MEpoints_ref,  data.MEpoints_ref)
  cuda.memcpy_htod(gpu_DestPoint_lst, data.DestPoint_lst)
  cuda.memcpy_htod(gpu_LEreduc_ref,   data.LEreduc_ref)
  end_transfer = time.time()
  data.time_transfer_in = end_transfer - start_transfer
  
  data.blocksize = 64 # one to start with
  data.n_blocks  = data.num_sources # one source per block
  
  mod = SourceModule(cuda_kernel_string % {'blocksize': data.blocksize,'terms':data.p,'terms_c':2*data.p}, 
  nvcc="nvcc",options=['-use_fast_math'], keep=False, no_extern_c=False)
  
  module_reduction = SourceModule(cudaKernel.local_expansion_reduction % {'blocksize': data.blocksize,'terms':data.p,'terms_c':2*data.p}, 
  nvcc="nvcc",options=['-use_fast_math'], keep=False, no_extern_c=False)
  
  if data.n_blocks > 512:
      n_blocks_x = data.n_blocks / 16
      n_blocks_y = 16
  else:
      n_blocks_x = data.n_blocks
      n_blocks_y = 1
  
  # Run multipole to local
  func = mod.get_function("me2me")
  data.time_gpu = func(gpu_MEterms_ref, gpu_MEpoints_ref, gpu_DestPoint_lst, gpu_LE_inter_ref, block=(data.blocksize,1,1), grid=(n_blocks_x, n_blocks_y), time_kernel=True)
  
  # Run reduction
  data.blocksize = 32
  func = module_reduction.get_function("localExpansionReduction")
  data.time_reduction = func(gpu_LE_inter_ref, gpu_LEreduc_ref, gpu_LEterms_ref, block=(data.blocksize,1,1), grid=(n_blocks_x, n_blocks_y), time_kernel=True)
  
  start_transfer = time.time()
  cuda.memcpy_dtoh(data.LEterms_ref, gpu_LEterms_ref)
  end_transfer = time.time()
  data.time_transfer_out = end_transfer - start_transfer

  ctx.pop() # context pop


# Performs multiple runs and outputs performance data
data           = dataset()
data.dim       = 2
data.sizeIL    = 27

for num_terms in [8, 12, 16]:
  for num_level in [3, 4, 5, 6, 7, 8]:
    data.p         = num_terms
    data.max_level = num_level
    generateTreeTestData(data)
    gpuComputeM2L(data, cudaKernel.precomputed_division)
    printTable(data)

#EOF

