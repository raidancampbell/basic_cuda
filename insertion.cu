// heavy assistance provided from nVidia's CUDA documentation and `vectorAdd.cu` piece of sample code
#include <stdio.h>
#include <sys/time.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

int* generate_array(int); // prototypes at the top of a non-header, because I hate C.
char* run_insertion_sort(int); // wraps the cuda_insertion_sort function

// this is quite possibly the stupidest piece of code I've written
// this is a single CUDA block for doing insertion sort
// insertion sort is not a parallelizable algorithm.
__global__ void cuda_insertion_sort(int *array, int num_elements) {
	int temp;
    for (int i = 1; i < num_elements; i++) {
      for(int j = i ; j > 0 ; j--){
        if(array[j] < array[j-1]){
          temp = array[j];
          array[j] = array[j-1];
          array[j-1] = temp;
        }
      }
    }
}

int main(void) {
    FILE *f;
    f = fopen("cuda_insertion.txt", "w");
	for(int i = 1000; i < 11000; i+= 1000) {
        printf("%d ", i);
        fprintf(f, "%d ", i);
        char* return_time = run_insertion_sort(i);
        fprintf(f, "%s ", return_time);
        printf("%s ", return_time);
        fflush(stdout);
        free(return_time);
        printf("\n");
        fprintf(f, "\n");
	}
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    fflush(f);
    fclose(f);
    printf("Done\n");
    return 0;
}

char* run_insertion_sort(int num_elements) {
	// initialize host's elements
    cudaError_t err = cudaSuccess;
    int* host_array = generate_array(num_elements);

    // initialize CUDA device's element
    int* cuda_array = NULL;
    size_t size = num_elements * sizeof(int);
    err = cudaMalloc((void **)&cuda_array, size);
    if (err != cudaSuccess) {  // check for errors on memory allocation
        fprintf(stderr, "Failed to allocate memory for array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	struct timeval tval_before, tval_after, tval_result; // declare some timing info
	gettimeofday(&tval_before, NULL);

    // copy the host element onto the CUDA device's element
    err = cudaMemcpy(cuda_array, host_array, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {  // check for errors on memory copy over to device
        fprintf(stderr, "Failed to copy array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cuda_insertion_sort<<<1,1>>>(cuda_array, num_elements);  // execute the kernel

    err = cudaGetLastError();  // check for any errors during kernel execution
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch `cuda_insertion_sort` kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // copy the result back from the CUDA device
    err = cudaMemcpy(host_array, cuda_array, size, cudaMemcpyDeviceToHost);  // this is a synchronous function.
    gettimeofday(&tval_after, NULL);
	timersub(&tval_after, &tval_before, &tval_result);  // finish up the timing
    if (err != cudaSuccess) {  // check for any errors on memory copy back to host
        fprintf(stderr, "Failed to copy array from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // and clean up
	err = cudaFree(cuda_array);
    if (err != cudaSuccess) {  // check for any errors on freeing the memory
        fprintf(stderr, "Failed to free device array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    free(host_array);

    // return info on the time spent
    char* return_string = (char*)malloc(100 * sizeof(char));
    sprintf(return_string, "%ld%03ld", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec / 1000);
    return return_string;
}

int* generate_array(int array_length) {
	int *return_var = (int*)malloc(sizeof(int) * array_length);

    for (int i = array_length - 1; i >= 0; i--) {
      return_var[array_length - i - 1] = i;
    }
    return return_var;
}


