#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <utility>
#include <algorithm>
#include "DS_timer.h"

using namespace std;

#define PAGE (970)		//한계 970
#define DAMPING (0.85)
#define BLOCK_SIZE (512)
#define NUM_BLOCKS (ceil((float)PAGE / BLOCK_SIZE))

float cal_PR(int, float *, int *, vector<int>);
void gen_graph(bool gr[][PAGE]);
void print_graph(bool gr[][PAGE]);
void printResult(float *, float *);

__global__ void analyize_graph(bool *_graph, int *_link_num, int *_link_index) {
	//스레드 아이디
	int tID = threadIdx.x + blockIdx.x * blockDim.x;

	//예외처리
	if (tID >= PAGE) return;

	//페이지에서 나가는 링크 수 파악
	for (int i = 0; i < PAGE; i++) {
		int index = tID * PAGE + i;
		if (_graph[index])
			_link_num[tID]++;
	}

	//페이지를 가리키는 링크 인덱스 파악
	int links = 0;
	for (int i = 0; i < PAGE; i++) {
		int index = i * PAGE + tID;
		if (_graph[index]) {
			_link_index[tID * PAGE + links] = i;
			links++;
		}
	}
}
__global__ void cal_PR(float *_PR, int *_link_num, int *_link_index, int *_cvg) {
	//스레드 아이디
	bool first = (threadIdx.x == 0) ? true : false;
	int tID = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ int cvg_part;
	if (first) cvg_part = 0;

	__syncthreads();

	float tmp = 0.0;
	//PR계산
	if (tID < PAGE) {
		int present = tID * PAGE;
		int index = _link_index[present];

		while (index >= 0) {
			tmp += _PR[index] / _link_num[index];

			present++;
			index = _link_index[present];
		}
		tmp = ((1 - DAMPING) / (present % PAGE) + DAMPING * tmp);

		//갱신
		if (_PR[tID] == tmp)
			atomicAdd(&cvg_part, 1);
	}

	__syncthreads();
	_PR[tID] = tmp;

	if (first)
		atomicAdd(_cvg, cvg_part);
}
__global__ void reduct_PR(float *_PR, float *_sum) {
	//스레드 아이디
	int tID = threadIdx.x + blockIdx.x * blockDim.x;
	int ID = threadIdx.x;

	__shared__ float PR_tmp[BLOCK_SIZE];

	if (tID < PAGE) PR_tmp[ID] = _PR[tID];
	else PR_tmp[ID] = 0.0;
	__syncthreads();

	//합산
	for (int span = BLOCK_SIZE / 2; span > 0; span /= 2) {
		if (ID < span) {
			PR_tmp[ID] += PR_tmp[ID + span];
		}
		__syncthreads();
	}

	if (ID == 0) {
		atomicAdd(_sum, PR_tmp[0]);
	}
}
__global__ void normal_PR(float *_PR, float *_sum) {
	//스레드 아이디
	int tID = threadIdx.x + blockIdx.x * blockDim.x;

	_PR[tID] /= *_sum;
}

int main() {
	//변수 설정
	bool graph[PAGE][PAGE];
	int *link_num;
	vector<int> *link_index;
	float *PR, *PR_tmp, *PR_CUDA;

	//device변수
	bool *d_graph;
	int *d_link_num, *d_link_index, *d_cvg;
	float *d_PR, *d_sum;
	
	//할당
	link_num = new int[PAGE]; memset(link_num, 0, sizeof(int) * PAGE);
	link_index = new vector<int>[PAGE];
	PR = new float[PAGE];
	PR_tmp = new float[PAGE];
	PR_CUDA = new float[PAGE];

	//변수 초기화
	cudaMalloc(&d_graph, sizeof(bool) * PAGE * PAGE);
	cudaMalloc(&d_link_num, sizeof(int) * PAGE);
	cudaMalloc(&d_link_index, sizeof(int) * PAGE * PAGE);
	cudaMalloc(&d_PR, sizeof(float) * PAGE);
	cudaMalloc(&d_cvg, sizeof(int) * 1);
	cudaMalloc(&d_sum, sizeof(float) * 1);
	cudaMemset(d_PR, 1.0, sizeof(float) * PAGE);
	cudaMemset(d_link_num, 0, sizeof(int) * PAGE);
	cudaMemset(d_link_index, -1, sizeof(int) * PAGE * PAGE);
	cudaMemset(d_cvg, 0, sizeof(int) * 1);
	cudaMemset(d_sum, 0.0, sizeof(float) * 1);

	//초기 PR값 모두 1.0
	for (int i = 0; i < PAGE; i++)
		PR[i] = 1.0;

	DS_timer timer(5);
	//타이머 설정
	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"CUDA Total");
	timer.setTimerName(2, (char*)"Parallel Calculate");
	timer.setTimerName(3, (char*)"Data Transfer, H to D");
	timer.setTimerName(4, (char*)"Data Transfer, D to H");

	srand((unsigned int)time(0));

	//그래프 생성
	gen_graph(graph);		
	
	/********** Serial **********/
	timer.onTimer(0);

	//각 페이지에서 나가는 링크 수 파악
	for (int i = 0; i < PAGE; i++)
		for (int j = 0; j < PAGE; j++)
			if (graph[i][j])
				link_num[i]++;

	//각 페이지를 가리키는 링크 인덱스 파악
	for (int i = 0; i < PAGE; i++)
		for (int j = 0; j < PAGE; j++)
			if (graph[j][i])
				link_index[i].push_back(j);

	//PR계산
	int cvg = 0;
	float sum = 0.0;

	while (cvg != PAGE) {
		cvg = 0;
		sum = 0.0;

		//각각의 PR 계산
		for (int i = 0; i < PAGE; i++) {
			PR_tmp[i] = cal_PR(i, PR, link_num, link_index[i]);
			sum += PR_tmp[i];
		}

		//PR 갱신
		for (int i = 0; i < PAGE; i++) {
			if (PR[i] == PR_tmp[i]) cvg++;
			else PR[i] = PR_tmp[i];
		}
	}

	//정규화
	for (int i = 0; i < PAGE; i++) {
		PR[i] /= sum;
	}

	timer.offTimer(0);

	/********** CUDA **********/
	timer.onTimer(1);

	//Data Transfer HostToDevice
	timer.onTimer(3);
	cudaMemcpy(d_graph, graph, sizeof(bool) * PAGE * PAGE, cudaMemcpyHostToDevice);
	timer.offTimer(3);

	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim(NUM_BLOCKS);

	//Kernel Call
	timer.onTimer(2);
	analyize_graph << <gridDim, blockDim >> > (d_graph, d_link_num, d_link_index);
	cudaDeviceSynchronize();
	timer.offTimer(2);

	cvg = 0;
	while (cvg != PAGE) {
		timer.onTimer(2);
		cudaMemset(d_cvg, 0, sizeof(int) * 1);

		cal_PR << <gridDim, blockDim>> > (d_PR, d_link_num, d_link_index, d_cvg);
		cudaDeviceSynchronize();
		timer.offTimer(2);

		timer.onTimer(4);
		cudaMemcpy(&cvg, d_cvg, sizeof(int) * 1, cudaMemcpyDeviceToHost);
		timer.offTimer(4);
	}

	timer.onTimer(2);
	reduct_PR << <gridDim, blockDim >> > (d_PR, d_sum);
	normal_PR << <gridDim, blockDim >> > (d_PR, d_sum);
	cudaDeviceSynchronize();
	timer.offTimer(2);

	//Data Transfer DeviceToHost
	timer.onTimer(4);
	cudaMemcpy(PR_CUDA, d_PR, sizeof(float) * PAGE, cudaMemcpyDeviceToHost);
	timer.offTimer(4);

	timer.offTimer(1);

	printResult(PR, PR_CUDA);
	timer.printTimer();
}

float cal_PR(int page, float *pageRank, int *link_num, vector<int> link_index) {
	float tmp = 0.0;
	int size = link_index.size();

	for (int i = 0; i < size; i++) {
		int index = link_index[i];

		tmp += pageRank[index] / link_num[index];
	}

	return ((1 - DAMPING) / size + DAMPING * tmp);
}
void gen_graph(bool gr[][PAGE]) {
	for (int i = 0; i < PAGE; i++)
		for (int j = 0; j < PAGE; j++) {
			if (i == j) gr[i][j] = 0;
			else gr[i][j] = (rand() % 2);
		}
}
void print_graph(bool gr[][PAGE]){
	for (int i = 0; i < PAGE; i++) {
		for (int j = 0; j < PAGE; j++)
			printf("%d ", gr[i][j]);
		printf("\n");
	}
}
void printResult(float *A, float *B) {
	vector<pair<float, int>> PR_A;
	vector<pair<float, int>> PR_B;

	for (int i = 0; i < PAGE; i++) {
		PR_A.push_back(pair<float, int>(A[i], i));
		PR_B.push_back(pair<float, int>(B[i], i));
	}

	sort(PR_A.begin(), PR_A.end());
	sort(PR_B.begin(), PR_B.end());

	int count = 25;

	printf("상위 %dpage PageRank 값\n", count);
	printf("=======================================================================\n");
	for (int i = 0; i < count; i++) {
		pair<float, int> a = PR_A.back();
		pair<float, int> b = PR_B.back();
		PR_A.pop_back();
		PR_B.pop_back();

		printf("%2d순위 : Serial - [%3d] %f, CUDA - [%3d] %f\n", 
			i + 1, a.second, a.first, b.second, b.first);
	}
	printf("...\n");
	printf("=======================================================================\n");
}