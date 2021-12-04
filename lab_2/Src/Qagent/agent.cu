/*************************************************************************
/* ECE 277: GPU Programmming 2021 WINTER
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#define ROWS			4
#define COLUMNS			4

#define RIGHT			0
#define DOWN			1
#define LEFT			2
#define UP				3
#define NUM_ACTIONS		4

#define GAMMA			0.9
#define ALPHA			0.1
#define DELTA_EPSILON	0.001


float* d_qtable;
curandState* d_randstate;
float* d_epsilon;
short* d_action;
float epsilon;





__global__ void Agent_init(curandState* d_randstate) {
	curand_init(clock(), 0, 0, d_randstate);
}


__global__ void Qtable_init(float* d_qtable) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	d_qtable[id] = 0;
}


void agent_init()
{
	epsilon = 1;

	int size_of_q = NUM_ACTIONS * ROWS * COLUMNS;

	cudaMalloc((void**)&d_qtable, sizeof(float)*size_of_q);
	cudaMalloc((void**)&d_randstate, sizeof(curandState));
	cudaMalloc((void**)&d_epsilon, sizeof(float));
	cudaMalloc((void**)&d_action, sizeof(float));

	Qtable_init << <ROWS*COLUMNS, NUM_ACTIONS >> > (d_qtable);
	Agent_init << <1, 1 >> > (d_randstate);

	cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);
}





float agent_adjustepsilon()
{
	if (epsilon - DELTA_EPSILON >= 1.0) {
		epsilon = 1.0 + DELTA_EPSILON;
	}
	else if (epsilon - DELTA_EPSILON <= 0.1) {
		epsilon = 0.1 + DELTA_EPSILON;
	}
	else {
		epsilon -= DELTA_EPSILON;
	}
	cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);
	return epsilon;
}





__global__ void Agent_action(int2* cstate, short* d_action, float* d_epsilon, float* d_qtable, curandState* d_randstate) {
	int x = cstate[0].x;
	int y = cstate[0].y;

	if (curand_uniform(d_randstate) < *d_epsilon){
		*d_action = (short)(curand_uniform(d_randstate)*NUM_ACTIONS);
	}
	else {
		int base_index = x*NUM_ACTIONS + y*COLUMNS*NUM_ACTIONS;
		float best_q = d_qtable[base_index];
		*d_action = (short)0;

		for (int i = 1; i < NUM_ACTIONS; i++) {
			if (d_qtable[base_index + i] > best_q) {
				best_q = d_qtable[base_index + i];
				*d_action = (short)i;
			}
		}
	}
}


short* agent_action(int2* cstate)
{
	Agent_action << <1, 1 >> > (cstate, d_action, d_epsilon, d_qtable, d_randstate);

	return d_action;
}





__global__ void Agent_update(int2* cstate, int2* nstate, float* rewards, float* d_qtable, short* d_action) {
	int x = cstate[0].x;
	int y = cstate[0].y;
	int current_index = x * NUM_ACTIONS + y * COLUMNS * NUM_ACTIONS + *d_action;

	int x_next = nstate[0].x;
	int y_next = nstate[0].y;
	int next_base_index = x_next * NUM_ACTIONS + y_next * COLUMNS * NUM_ACTIONS;

	float current_q = d_qtable[current_index];

	float reward = rewards[0];

	float max_q_next = d_qtable[next_base_index];
	for (int i = 1; i < NUM_ACTIONS; i++) {
		if (d_qtable[next_base_index + i] > max_q_next) {
			max_q_next = d_qtable[next_base_index + i];
		}
	}

	d_qtable[current_index] += ALPHA * (reward + GAMMA * max_q_next - current_q);
}


void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	Agent_update << <1, 1 >> > (cstate, nstate, rewards, d_qtable, d_action);
}