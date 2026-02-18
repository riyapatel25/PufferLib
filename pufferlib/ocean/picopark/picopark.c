/* Pure C demo file for Picopark. 
 */

#include "picopark.h"

int main() {
    Picopark env = {
        .view_width = 25,
        .view_height = 14,
        .height = 14,
        .scroll_rate = 5,
        .num_agents = 4 // Configurable: 1-8 agents
    };
    // Multi-channel obs: (channels * grid_size + extras) per agent
    int grid_size = env.view_width * env.view_height;
    int obs_stride = OBS_CHANNELS * grid_size + OBS_EXTRAS;
    int obs_size = obs_stride * env.num_agents;
    env.observations = (unsigned char*)calloc(obs_size, sizeof(unsigned char));
    env.actions = (int*)calloc(env.num_agents, sizeof(int));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Manual control mode (controls first agent)
            env.actions[0] = NOOP;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
            if (IsKeyDown(KEY_SPACE)) env.actions[0] = SHOOT;
            // Other agents act randomly
            for (int i = 1; i < env.num_agents; i++) {
                env.actions[i] = rand() % 6;
            }
        } else {
            // Random agent mode
            for (int i = 0; i < env.num_agents; i++) {
                env.actions[i] = rand() % 6;
            }
        }
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
