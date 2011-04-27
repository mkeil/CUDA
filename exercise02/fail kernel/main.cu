__global__ void emptyKernel() {
	printf("Test");
}

int main() {
	dim3 threadsPerBlock(1);
	dim3 blocksPerGrid(1);
	emptyKernel<<<blocksPerGrid, threadsPerBlock>>>();
	cudaThreadSynchronize();
	return 0;
}