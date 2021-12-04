/*************************************************************************
/* ECE 277: GPU Programmming 2021 FALL
/* Author and Instructor: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#define RIGHT		0
#define DOWN		1
#define LEFT		2
#define UP			3

short *d_action;

__global__ void cuda_agent(int2* cstate, short* d_action) {
	// very hard-coded movements and solution, but I believe this is what the lab asked for
	if (cstate[0].x == 0 && cstate[0].y != 3) *d_action = DOWN;
	else if (cstate[0].x != 2 && cstate[0].y == 3) *d_action = RIGHT;
	else if (cstate[0].x == 2 && cstate[0].y == 3) *d_action = UP;

}

void agent_init() {
	cudaMalloc(&d_action, sizeof(short));
}

short* agent_action(int2* cstate) {
	cuda_agent << <1, 1 >> > (cstate, d_action);
	return d_action;
}