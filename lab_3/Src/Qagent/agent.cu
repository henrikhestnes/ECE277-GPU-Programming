/*************************************************************************
/* ECE 277: GPU Programmming 2021 WINTER quarter
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

#define NUM_AGENTS		128

#define ROWS			32
#define COLUMNS			32
#define NUM_ACTIONS		4

#define GAMMA			0.9
#define ALPHA			0.1
#define DELTA_EPSILON	0.001


float* d_qtable;
curandState* d_randstate;
float* d_epsilon;
float epsilon;
short *d_action;
char* d_isactive;


__global__ void Qtable_init(float* d_qtable) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	d_qtable[id] = 0;
}

__global__ void Agent_init(curandState* d_randstate, char* d_isactive) {
	int id = threadIdx.x;
	curand_init(clock() + id, id, 0, &d_randstate[id]);
	d_isactive[id] = 1;
}

void agent_init()
{
	epsilon = 1;

	int size_of_q = NUM_ACTIONS * ROWS * COLUMNS;

	cudaMalloc((void**)&d_qtable, sizeof(float)*size_of_q);
	cudaMalloc((void**)&d_randstate, sizeof(curandState)*NUM_AGENTS);
	cudaMalloc((void**)&d_epsilon, sizeof(float));
	cudaMalloc((void**)&d_action, sizeof(short)*NUM_AGENTS);
	cudaMalloc((void**)&d_isactive, sizeof(char)*NUM_AGENTS);

	Qtable_init << <ROWS*COLUMNS, NUM_ACTIONS >> > (d_qtable);
	Agent_init << <1, NUM_AGENTS >> > (d_randstate, d_isactive);

	cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void Agent_init_episode(char* d_isactive) {
	int id = threadIdx.x;
	d_isactive[id] = 1;
}

void agent_init_episode() 
{
	Agent_init_episode << <1, NUM_AGENTS >> > (d_isactive);
}

float agent_adjustepsilon() 
{
	if (epsilon - DELTA_EPSILON >= 1.0) {
		epsilon = 1.0 + DELTA_EPSILON;
	}
	else if (epsilon - DELTA_EPSILON <= 0) {
		epsilon = 0 + DELTA_EPSILON;
	}
	else {
		epsilon -= DELTA_EPSILON;
	}
	cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);
	return epsilon;
}


__global__ void Agent_action(int2* cstate, short* d_action, char* d_isactive, float* d_epsilon, float* d_qtable, curandState* d_randstate) {
	int id = threadIdx.x;
	
	if (!d_isactive[id]) {
		return;
	}

	int x = cstate[id].x;
	int y = cstate[id].y;

	if (curand_uniform(&d_randstate[id]) < *d_epsilon) {
		d_action[id] = (short)(ceil(curand_uniform(&d_randstate[id]) * NUM_ACTIONS) - 1);
	}
	else {
		int base_index = x * NUM_ACTIONS + y * COLUMNS * NUM_ACTIONS;
		float best_q = d_qtable[base_index];
		d_action[id] = (short)0;

		for (int i = 1; i < NUM_ACTIONS; i++) {
			if (d_qtable[base_index + i] > best_q) {
				best_q = d_qtable[base_index + i];
				d_action[id] = (short)i;
			}
		}
	}
}

short* agent_action(int2* cstate)
{
	Agent_action << <1, NUM_AGENTS >> > (cstate, d_action, d_isactive, d_epsilon, d_qtable, d_randstate);
	return d_action;
}


__global__ void Agent_update(int2* cstate, int2* nstate, float* rewards, float* d_qtable, short* d_action, char* d_isactive) {
	int id = threadIdx.x;

	if (!d_isactive[id]) {
		return;
	}

	int x = cstate[id].x;
	int y = cstate[id].y;
	int current_q_index = x * NUM_ACTIONS + y * COLUMNS * NUM_ACTIONS + (int)d_action[id];

	int x_next = nstate[id].x;
	int y_next = nstate[id].y;
	int next_q_index = x_next * NUM_ACTIONS + y_next * COLUMNS * NUM_ACTIONS;

	float current_q = d_qtable[current_q_index];
	float reward = rewards[id];

	if (reward == 0) {
		float max_q_next = d_qtable[next_q_index];
		for (int i = 1; i < NUM_ACTIONS; i++) {
			if (d_qtable[next_q_index + i] > max_q_next) {
				max_q_next = d_qtable[next_q_index + i];
			}
		}
		d_qtable[current_q_index] += ALPHA * (GAMMA * max_q_next - current_q);
	}
	else {
		d_qtable[current_q_index] += ALPHA * (reward - current_q);
		d_isactive[id] = 0;
	}
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	Agent_update << <1, NUM_AGENTS >> > (cstate, nstate, rewards, d_qtable, d_action, d_isactive);
}
