/* Picopark
 */

 #include <stdlib.h>
 #include <string.h>
 #include <stdint.h>
 #include "raylib.h"
  
#define MAX_AGENTS 8
#define NUM_COLORS 8
#define MAX_BLOCKS 256
#define TILE_SIZE 48  // Only used for rendering
#define AGENT_HEIGHT 2  // Agents are 2 tiles tall (row = feet, row-1 = head)
#define AGENT_SPEED 1.0f  // Tiles per tick (1.0 = full tile, 0.5 = half tile)
 
 // Grid dimensions for O(1) lookups (fixed max to avoid dynamic allocation)
 #define MAX_GRID_HEIGHT 32
 #define MAX_GRID_WIDTH 128
 #define GRID_EMPTY -1
  
 // Actions
 const unsigned char NOOP = 0;
 const unsigned char DOWN = 1;
 const unsigned char UP = 2;
 const unsigned char LEFT = 3;
 const unsigned char RIGHT = 4;
 const unsigned char SHOOT = 5;
 
 // =============================================================================
 // Multi-channel observation layout
 // =============================================================================


#define OBS_CHANNELS 5  // block_color, agent_color, bullet, goal, key_door
#define OBS_EXTRAS 11  // has_key, door_unlocked, door_row, key_row, key_col, key_holder_idx
 
// Channel indices for clarity
#define OBS_CH_BLOCK_COLOR 0
#define OBS_CH_AGENT_COLOR 1
#define OBS_CH_BULLET 2
#define OBS_CH_GOAL 3
#define OBS_CH_KEY_DOOR 4  // 0=empty, 1=key, 2=door
  
  typedef struct {
      float row, col;       // Sub-tile coordinates (collision uses (int)row, (int)col)
      float last_row, last_col;  // Previous position for rendering interpolation
      int color;
      int variant;      // 0 or 1 - which sprite variant to use
      int alive;
      int reached_goal; // Track if agent has been rewarded for reaching goal
      int has_key;      // 1 if this agent is holding the key
      int in_door;      // 1 if agent has entered the door
  } Agent;
  
  typedef struct {
      int row, col;     // Tile coordinates
      int active;
  } Bullet;
  
  // Individual block (replaces BoxColumn for mixed-color clumps)
  typedef struct {
      int row, col;
      int color;
      int exists;  // 1 if block exists, 0 if destroyed
  } Block;
  
  typedef struct {
      float perf;
      float score;
      float episode_return;
      float episode_length;
      float agents_at_goal;
      float progress;  // Dense metric: max_agent_col / goal_col at episode end
      float n;
  } Log;
  
  // Block color palettes: [border, fill] for each color index
  // Colors matched to agent character colors
  static const Color BLOCK_COLORS[NUM_COLORS][2] = {
      {{41, 98, 255, 255}, {100, 149, 255, 255}},    // 0: Blue - vibrant blue
      {{120, 120, 120, 255}, {180, 180, 180, 255}},  // 1: Gray
      {{76, 187, 97, 255}, {144, 228, 144, 255}},    // 2: Green - brighter, lighter
      {{230, 126, 34, 255}, {255, 167, 38, 255}},    // 3: Orange
      {{199, 97, 199, 255}, {255, 153, 255, 255}},   // 4: Pink - soft pastel pink (matches reference)
      {{150, 111, 214, 255}, {192, 162, 235, 255}},  // 5: Purple - lighter lavender purple
      {{211, 84, 84, 255}, {239, 134, 134, 255}},    // 6: Red/Coral
      {{192, 202, 51, 255}, {255, 255, 102, 255}},   // 7: Yellow - olive border, bright fill
  };
  
  // Rendering client with sprite textures
  typedef struct {
      Texture2D cats[NUM_COLORS][2];  // [color][variant] - two variants per color
      Texture2D blocks[NUM_COLORS];   // Keep for compatibility, but may not use
      Texture2D goal;
      Texture2D bullet;
      Texture2D key;
      Texture2D door;
  } Client;
  
 typedef struct {
     Log log;
     Client* client;
     Agent agents[MAX_AGENTS];
     Bullet bullets[MAX_AGENTS];
     Block blocks[MAX_BLOCKS];
     int num_blocks;
     int num_agents;    // Configurable number of agents (1-8)
     int goal_col;      // Tile column where goal/door is
     int level_width;   // Level width in tiles
     int camera_col;    // Camera position in tiles
     int view_width;    // Horizontal view in tiles
     int view_height;   // Vertical view in tiles
     int height;        // Height in tiles
     int scroll_rate;   // Ticks between camera scrolls
     int tick;
     float total_episode_reward;  // Accumulated rewards for current episode
     unsigned char* observations;
     int* actions;
     float* rewards;
     unsigned char* terminals;
    // Dense grids for O(1) lookups (indexed by [row * MAX_GRID_WIDTH + col])
    int8_t block_grid[MAX_GRID_HEIGHT * MAX_GRID_WIDTH];  // -1 = empty, 0-7 = block color
    uint8_t agent_grid[MAX_GRID_HEIGHT * MAX_GRID_WIDTH]; // 0 = empty, 1-8 = agent color+1
    uint8_t bullet_grid[MAX_GRID_HEIGHT * MAX_GRID_WIDTH]; // 0 = empty, 1 = bullet present
    
    // Key state
    int key_row, key_col;      // Key position (-1,-1 if picked up)
    int key_holder;            // Agent index holding key (-1 if none)
    
    // Door state
    int door_row;              // Door row position (door_col = goal_col)
    int door_unlocked;         // 1 if key-holder has entered door
    int friendly_fire;         // 1 if episode ended by team kill
} Picopark;
  
 // Grid index helper (inline for speed)
 static inline int grid_idx(int row, int col) {
     return row * MAX_GRID_WIDTH + col;
 }
 
 // Random integer in range [min, max] (inclusive)
 static inline int rand_int(int min, int max) {
     if (min >= max) return min;
     return min + rand() % (max - min + 1);
 }
 
 // Clamp value to [lo, hi]
 static inline int clamp_int(int v, int lo, int hi) {
     if (v < lo) return lo;
     if (v > hi) return hi;
     return v;
 }
 
 // Helper to add reward and track episode total
 static inline void add_reward(Picopark* env, int agent_idx, float reward) {
     env->rewards[agent_idx] += reward;
     env->total_episode_reward += reward;
 }
 
 // O(1) block color lookup from grid (-1 if empty or out of bounds)
 static inline int8_t get_block_color(Picopark* env, int row, int col) {
     if (row < 0 || row >= env->height || col < 0 || col >= MAX_GRID_WIDTH) {
         return GRID_EMPTY;
     }
     return env->block_grid[grid_idx(row, col)];
 }
 
 // Check if block exists at position (O(1))
 static inline int has_block_at(Picopark* env, int row, int col) {
     return get_block_color(env, row, col) != GRID_EMPTY;
 }
 
 // Legacy get_block_at for rendering (still uses block list for drawing)
 Block* get_block_at(Picopark* env, int row, int col) {
     for (int i = 0; i < env->num_blocks; i++) {
         if (env->blocks[i].exists && 
             env->blocks[i].row == row && 
             env->blocks[i].col == col) {
             return &env->blocks[i];
         }
     }
     return NULL;
 }
 
 // O(1) agent occupancy check from grid
 static inline int agent_at_fast(Picopark* env, int row, int col) {
     if (row < 0 || row >= env->height || col < 0 || col >= MAX_GRID_WIDTH) {
         return 0;
     }
     return env->agent_grid[grid_idx(row, col)];
 }
 
 // O(1) bullet occupancy check from grid
 static inline int bullet_at_fast(Picopark* env, int row, int col) {
     if (row < 0 || row >= env->height || col < 0 || col >= MAX_GRID_WIDTH) {
         return 0;
     }
     return env->bullet_grid[grid_idx(row, col)];
 }
 

int agent_at(Picopark* env, int row, int col, int exclude_idx) {
    for (int i = 0; i < env->num_agents; i++) {
        if (i == exclude_idx || !env->agents[i].alive) continue;
        int agent_tile_row = (int)env->agents[i].row;
        int agent_tile_col = (int)env->agents[i].col;
        // Check if row hits any part of this AGENT_HEIGHT-tile agent
        for (int dy = 0; dy < AGENT_HEIGHT; dy++) {
            if (agent_tile_row - dy == row && agent_tile_col == col) {
                return 1;
            }
        }
    }
    return 0;
}

// Returns agent index at position, or -1 if none
// Checks if any tile of a 2-tile tall agent is at the given position
// Uses integer tile coordinates from float agent positions
// Skips agents who are in_door (they're removed from world)
int get_agent_at(Picopark* env, int row, int col, int exclude_idx) {
    for (int i = 0; i < env->num_agents; i++) {
        if (i == exclude_idx || !env->agents[i].alive || env->agents[i].in_door) continue;
        int agent_tile_row = (int)env->agents[i].row;
        int agent_tile_col = (int)env->agents[i].col;
        for (int dy = 0; dy < AGENT_HEIGHT; dy++) {
            if (agent_tile_row - dy == row && agent_tile_col == col) {
                return i;
            }
        }
    }
    return -1;
}
 
 // Add a block to the level (updates both list and grid)
 void add_block(Picopark* env, int row, int col, int color) {
     if (env->num_blocks >= MAX_BLOCKS) return;
     if (row < 0 || row >= MAX_GRID_HEIGHT || col < 0 || col >= MAX_GRID_WIDTH) return;
     
     Block* block = &env->blocks[env->num_blocks++];
     block->row = row;
     block->col = col;
     block->color = color;
     block->exists = 1;
     
     // Update grid
     env->block_grid[grid_idx(row, col)] = (int8_t)color;
 }
 
 // Clear block from grid (called when block is destroyed)
 static inline void clear_block_grid(Picopark* env, int row, int col) {
     if (row >= 0 && row < MAX_GRID_HEIGHT && col >= 0 && col < MAX_GRID_WIDTH) {
         env->block_grid[grid_idx(row, col)] = GRID_EMPTY;
     }
 }
 
// Rebuild agent occupancy grid (call once per step, before observations)
// Stores color+1 so 0 = empty, 1-8 = agent with color 0-7
// Marks all AGENT_HEIGHT tiles that each agent occupies
// Uses integer tile coordinates from float agent positions
static void rebuild_agent_grid(Picopark* env) {
    memset(env->agent_grid, 0, sizeof(env->agent_grid));
    for (int i = 0; i < env->num_agents; i++) {
        Agent* a = &env->agents[i];
        if (!a->alive) continue;
        int tile_row = (int)a->row;
        int tile_col = (int)a->col;
        // Mark all tiles the agent occupies (row = feet, row-1 = head, etc.)
        for (int dy = 0; dy < AGENT_HEIGHT; dy++) {
            int r = tile_row - dy;
            if (r >= 0 && r < MAX_GRID_HEIGHT && tile_col >= 0 && tile_col < MAX_GRID_WIDTH) {
                env->agent_grid[grid_idx(r, tile_col)] = (uint8_t)(a->color + 1);
            }
        }
    }
}
 
 // Rebuild bullet occupancy grid (call once per step, before observations)
 static void rebuild_bullet_grid(Picopark* env) {
     memset(env->bullet_grid, 0, sizeof(env->bullet_grid));
     for (int i = 0; i < env->num_agents; i++) {
         Bullet* b = &env->bullets[i];
         if (b->active && b->row >= 0 && b->row < MAX_GRID_HEIGHT && 
             b->col >= 0 && b->col < MAX_GRID_WIDTH) {
             env->bullet_grid[grid_idx(b->row, b->col)] = 1;
         }
     }
 }
  
 void generate_level(Picopark* env) {
     env->num_blocks = 0;
     // Initialize block grid to empty
     memset(env->block_grid, GRID_EMPTY, sizeof(env->block_grid));
     
     int col_x = env->view_width;
     
     // Shuffle colors for random ordering
     int colors[MAX_AGENTS];
     for (int i = 0; i < env->num_agents; i++) {
         colors[i] = env->agents[i].color;
     }
     // Fisher-Yates shuffle
     for (int i = env->num_agents - 1; i > 0; i--) {
         int j = rand() % (i + 1);
         int temp = colors[i];
         colors[i] = colors[j];
         colors[j] = temp;
     }
     
     // === REGION 1: Alternating color columns (one per agent) ===
     // RANDOMIZE: stack1 width can vary from num_agents to num_agents+2
     int stack1_width = env->num_agents + rand_int(0, 2);
     // Each column is solid, full height, cycling through agent colors
     for (int c = 0; c < stack1_width; c++) {
         int world_col = col_x + c;
         int color_idx = c % env->num_agents;  // Cycle colors if width > num_agents
         for (int r = 0; r < env->height; r++) {
             add_block(env, r, world_col, colors[color_idx]);
         }
     }
     // RANDOMIZE: gap between stack1 and stack2 (3-7 columns)
     int stack1_gap = rand_int(3, 7);
     col_x += stack1_width + stack1_gap;
     
     // === REGION 2: Frame pattern with large rectangular regions ===
     // RANDOMIZE: frame dimensions while keeping structure
     int frame_width = rand_int(8, 14);
     int frame_left_cols = rand_int(1, 3);
     int frame_right_cols = rand_int(1, 3);
     // Ensure middle section has at least 4 columns
     int min_middle = 4;
     if (frame_width - frame_left_cols - frame_right_cols < min_middle) {
         // Adjust frame_width to ensure valid middle
         frame_width = frame_left_cols + frame_right_cols + min_middle;
     }
     int middle_cols = frame_width - frame_left_cols - frame_right_cols;
     
     // Pick 4 colors (cycle if fewer agents)
     int colorA = colors[0];
     int colorB = colors[1 % env->num_agents];
     int colorC = colors[2 % env->num_agents];
     int colorD = colors[3 % env->num_agents];
     
     // Pre-compute color map for region 2 (use 32x32 for safety)
     int color_map[32][32];
     memset(color_map, 0, sizeof(color_map));
     
     // Left frame columns (colorA) - full height, solid
     for (int c = 0; c < frame_left_cols; c++) {
         for (int r = 0; r < env->height && r < 32; r++) {
             color_map[r][c] = colorA;
         }
     }
     
     // Right frame columns (colorB) - full height, solid
     for (int c = 0; c < frame_right_cols; c++) {
         int map_col = frame_width - frame_right_cols + c;
         if (map_col < 32) {
             for (int r = 0; r < env->height && r < 32; r++) {
                 color_map[r][map_col] = colorB;
             }
         }
     }
     
     // Middle area: large horizontal bands (half height each)
     // RANDOMIZE: swap top/bottom colors randomly
     int mid_row = env->height / 2;
     int swap_middle = rand() % 2;
     int top_color = swap_middle ? colorD : colorC;
     int bot_color = swap_middle ? colorC : colorD;
     for (int c = frame_left_cols; c < frame_width - frame_right_cols && c < 32; c++) {
         for (int r = 0; r < env->height && r < 32; r++) {
             color_map[r][c] = (r < mid_row) ? top_color : bot_color;
         }
     }
     
     // Large embedded patch in the middle (spans most of middle area)
     // RANDOMIZE: patch start row within [0, height/2)
     int patch_h = clamp_int(env->height / 2, 2, env->height - 1);
     int patch_w = clamp_int(middle_cols, 2, middle_cols);
     int max_patch_start = (env->height / 2 > 0) ? env->height / 2 : 1;
     int patch_start_r = rand_int(0, max_patch_start - 1);
     int patch_start_c = frame_left_cols;
     int patch_color = top_color;  // Contrast with bottom half
     
     for (int r = patch_start_r; r < patch_start_r + patch_h && r < env->height && r < 32; r++) {
         for (int c = patch_start_c; c < patch_start_c + patch_w && c < frame_width - frame_right_cols && c < 32; c++) {
             // Override bottom portion to create embedded rectangle effect
             if (r >= mid_row) {
                 color_map[r][c] = patch_color;
             }
         }
     }
     
     // Add a second smaller embedded patch in the top area
     // RANDOMIZE: patch2 height (2-4), start row within top half
     int patch2_h = rand_int(2, 4);
     int patch2_w = clamp_int(middle_cols - 2, 2, middle_cols);
     int max_patch2_start = mid_row - patch2_h - 1;
     int patch2_start_r = (max_patch2_start > 0) ? rand_int(0, max_patch2_start) : 0;
     int patch2_start_c = frame_left_cols + rand_int(0, 1);
     
     for (int r = patch2_start_r; r < patch2_start_r + patch2_h && r < mid_row && r < 32; r++) {
         for (int c = patch2_start_c; c < patch2_start_c + patch2_w && c < frame_width - frame_right_cols && c < 32; c++) {
             color_map[r][c] = bot_color;
         }
     }
     
     // Place region 2 blocks using color map (solid, no gaps)
     for (int c = 0; c < frame_width && c < 32; c++) {
         int world_col = col_x + c;
         for (int r = 0; r < env->height && r < 32; r++) {
             add_block(env, r, world_col, color_map[r][c]);
         }
     }
     
     // Create hole for key in region 2 (random position in middle area)
     // Row must be at least AGENT_HEIGHT-1 to fit a 2-tile tall hole
     int key_hole_row = rand_int(AGENT_HEIGHT - 1, env->height - 1);
     int key_hole_local_col = frame_left_cols + rand_int(1, middle_cols - 2);
     if (key_hole_local_col < frame_left_cols) key_hole_local_col = frame_left_cols;
     if (key_hole_local_col >= frame_width - frame_right_cols) key_hole_local_col = frame_width - frame_right_cols - 1;
     int key_hole_col = col_x + key_hole_local_col;
     
     // Remove 2 blocks vertically (agent is 2 tiles tall)
     for (int dy = 0; dy < AGENT_HEIGHT; dy++) {
         int hole_row = key_hole_row - dy;
         if (hole_row >= 0) {
             clear_block_grid(env, hole_row, key_hole_col);
             Block* b = get_block_at(env, hole_row, key_hole_col);
             if (b) b->exists = 0;
         }
     }
     
     // Store key position (at bottom of hole where agent's feet will be)
     env->key_row = key_hole_row;
     env->key_col = key_hole_col;
     env->key_holder = -1;
     
     // RANDOMIZE: gap after stack2 (2-4 columns)
     int stack2_gap = rand_int(2, 4);
     col_x += frame_width + stack2_gap;
     env->goal_col = col_x + rand_int(2, 4);
     env->level_width = env->goal_col + 3;
     
    // Set door at random row in goal column (min AGENT_HEIGHT-1 so agents can reach it)
    env->door_row = rand_int(AGENT_HEIGHT - 1, env->height - 1);
     env->door_unlocked = 0;
 }
  

  
 // Check if a tile is blocked (for movement)
 int is_tile_blocked(Picopark* env, int row, int col, int agent_idx) {
     // Bounds check
     if (row < 0 || row >= env->height) return 1;
     if (col < env->camera_col || col >= env->camera_col + env->view_width) return 1;
     
     // Block check (O(1) grid lookup)
     if (has_block_at(env, row, col)) return 1;
     
     // Agent check (loop over 8 agents max - fast enough)
     for (int i = 0; i < env->num_agents; i++) {
         if (i != agent_idx && env->agents[i].alive &&
             env->agents[i].row == row && env->agents[i].col == col) {
             return 1;
         }
     }
     return 0;
 }
 

int can_agent_move(Picopark* env, float new_row, float new_col, int agent_idx) {
    int tile_row = (int)new_row;
    int tile_col = (int)new_col;
    
    // Bounds check - agent needs room for head (row-1)
    if (tile_row < (AGENT_HEIGHT - 1) || tile_row >= env->height) return 0;
    if (tile_col < env->camera_col || tile_col >= env->camera_col + env->view_width) return 0;
    
    // World boundary - can't move past 2 columns after door
    if (tile_col > env->goal_col + 2) return 0;
    
    // Block check - all tiles must be clear (O(1) grid lookup each)
    for (int dy = 0; dy < AGENT_HEIGHT; dy++) {
        if (has_block_at(env, tile_row - dy, tile_col)) return 0;
    }
    
    // Agent check - all tiles must be clear of other agents (skip in_door agents)
    for (int i = 0; i < env->num_agents; i++) {
        if (i == agent_idx || !env->agents[i].alive || env->agents[i].in_door) continue;
        int other_tile_row = (int)env->agents[i].row;
        int other_tile_col = (int)env->agents[i].col;
        // Check if any tile of this agent overlaps any tile of other agent
        for (int dy1 = 0; dy1 < AGENT_HEIGHT; dy1++) {
            for (int dy2 = 0; dy2 < AGENT_HEIGHT; dy2++) {
                if (tile_row - dy1 == other_tile_row - dy2 && tile_col == other_tile_col) {
                    return 0;
                }
            }
        }
    }
    return 1;
}
  
void compute_observations(Picopark* env) {
    // Rebuild occupancy grids once per observation pass (O(num_agents) total)
    rebuild_agent_grid(env);
    rebuild_bullet_grid(env);
    
    // Multi-channel grid layout: [ch0_all_cells | ch1_all_cells | ch2_all_cells | ch3_all_cells | extras]
    int grid_size = env->view_width * env->view_height;
    int obs_stride = OBS_CHANNELS * grid_size + OBS_EXTRAS;
    int half_w = env->view_width / 2;
    int half_h = env->view_height / 2;
    
    for (int a = 0; a < env->num_agents; a++) {
        Agent* agent = &env->agents[a];
        unsigned char* obs = &env->observations[a * obs_stride];
        
        // Zero out entire observation buffer for this agent
        memset(obs, 0, obs_stride);
        
        if (!agent->alive) continue;
        
        int self_color = agent->color;
        int self_row = agent->row;
        int self_col = agent->col;
        
        // Pointers to each channel's start
        unsigned char* ch_block = obs + OBS_CH_BLOCK_COLOR * grid_size;
        unsigned char* ch_agent = obs + OBS_CH_AGENT_COLOR * grid_size;
        unsigned char* ch_bullet = obs + OBS_CH_BULLET * grid_size;
        unsigned char* ch_goal = obs + OBS_CH_GOAL * grid_size;
        unsigned char* ch_key_door = obs + OBS_CH_KEY_DOOR * grid_size;
        
        for (int dr = -half_h; dr <= half_h; dr++) {
            int world_r = self_row + dr;
            int obs_r = dr + half_h;
            
            // Skip entire row if out of bounds vertically
            if (world_r < 0 || world_r >= env->height) continue;
            
            for (int dc = -half_w; dc <= half_w; dc++) {
                int world_c = self_col + dc;
                int obs_c = dc + half_w;
                int cell_idx = obs_r * env->view_width + obs_c;
                
                // Skip if out of bounds horizontally
                if (world_c < 0) continue;
                
                // Channel 0: block_color (0 = empty, 1-8 = color+1)
                int8_t block_color = get_block_color(env, world_r, world_c);
                if (block_color != GRID_EMPTY) {
                    ch_block[cell_idx] = (unsigned char)(block_color + 1);
                }
                
                // Channel 1: agent_color (0 = none, 1-8 = color+1)
                // agent_at_fast already returns color+1, or 0 if empty
                ch_agent[cell_idx] = (unsigned char)agent_at_fast(env, world_r, world_c);
                
                // Channel 2: bullet (0/1)
                if (bullet_at_fast(env, world_r, world_c)) {
                    ch_bullet[cell_idx] = 1;
                }
                
                // Channel 3: door (0/1) - only at door position, not full column
                if (world_c == env->goal_col && world_r == env->door_row) {
                    ch_goal[cell_idx] = 1;
                }
                
                // Channel 4: key_door (0=empty, 1=key on ground, 2=door)
                // Key on ground (not picked up)
                if (env->key_row >= 0 && world_r == env->key_row && world_c == env->key_col) {
                    ch_key_door[cell_idx] = 1;
                }
                // Door position
                if (world_c == env->goal_col && world_r == env->door_row) {
                    ch_key_door[cell_idx] = 2;
                }
            }
        }
        
        // Extras vector (after grid channels)
        unsigned char* extras = obs + OBS_CHANNELS * grid_size;
        extras[0] = (unsigned char)(self_color + 1);  // self_color: 1-8
        extras[1] = (unsigned char)(self_row < 255 ? self_row : 255);  // self_row
        extras[2] = (unsigned char)(self_col < 255 ? self_col : 255);  // self_col
        extras[3] = (unsigned char)(env->goal_col < 255 ? env->goal_col : 255);  // goal_col
        extras[4] = (unsigned char)(env->camera_col < 255 ? env->camera_col : 255);  // camera_col
        
       // Key/door info
       extras[5] = (unsigned char)(agent->has_key);  // 0 or 1: does this agent have key
       extras[6] = (unsigned char)(env->door_unlocked);  // 0 or 1: is door unlocked
       extras[7] = (unsigned char)(env->door_row < 255 ? env->door_row : 255);  // door row position
       
       // Key position - always available (use holder's coords if key is held)
       if (env->key_holder >= 0 && env->key_holder < env->num_agents) {
           // Key is held - show holder's position
           int holder_row = (int)env->agents[env->key_holder].row;
           int holder_col = (int)env->agents[env->key_holder].col;
           extras[8] = (unsigned char)(holder_row < 255 ? holder_row : 255);
           extras[9] = (unsigned char)(holder_col < 255 ? holder_col : 255);
       } else {
           // Key is on ground
           extras[8] = (unsigned char)(env->key_row >= 0 ? env->key_row : 255);
           extras[9] = (unsigned char)(env->key_col >= 0 ? env->key_col : 255);
       }
       
       // Key holder index (0-7 or 255 if none)
       extras[10] = (unsigned char)(env->key_holder >= 0 ? env->key_holder : 255);
   }
}
  
  void c_reset(Picopark* env) {
      env->tick = 0;
      env->total_episode_reward = 0.0f;
      env->camera_col = 0;
      
      // Clamp num_agents to valid range (1-8)
      if (env->num_agents < 1) env->num_agents = 1;
      if (env->num_agents > MAX_AGENTS) env->num_agents = MAX_AGENTS;
      
     // Shuffle colors for random character selection
     int colors[NUM_COLORS] = {0, 1, 2, 3, 4, 5, 6, 7};
     for (int i = NUM_COLORS - 1; i > 0; i--) {
         int j = rand() % (i + 1);
         int temp = colors[i];
         colors[i] = colors[j];
         colors[j] = temp;
     }
     

     int rows[MAX_GRID_HEIGHT];
     int max_rows = (env->height < MAX_GRID_HEIGHT) ? env->height : MAX_GRID_HEIGHT;
     int valid_count = 0;
     
     // Only use rows that leave room for the agent's head
     for (int i = AGENT_HEIGHT - 1; i < max_rows; i++) {
         rows[valid_count++] = i;
     }
     
     // Fisher-Yates shuffle for unique random rows
     for (int i = valid_count - 1; i > 0; i--) {
         int j = rand() % (i + 1);
         int temp = rows[i];
         rows[i] = rows[j];
         rows[j] = temp;
     }
     
     // Reset active agents with randomized unique rows, fixed col=1
     // Initialize float positions and last positions for sub-tile movement
     for (int i = 0; i < env->num_agents; i++) {
         env->agents[i].row = (float)rows[i % valid_count];  // Unique random row (mod for safety)
         env->agents[i].col = 1.0f;        // All agents start in column 1
         env->agents[i].last_row = env->agents[i].row;
         env->agents[i].last_col = env->agents[i].col;
         env->agents[i].color = colors[i];
         env->agents[i].variant = rand() % 2;  // Randomly pick sprite variant (0 or 1)
         env->agents[i].alive = 1;
         env->agents[i].reached_goal = 0;
         env->agents[i].has_key = 0;      // No agent has key at start
         env->agents[i].in_door = 0;      // No agent in door at start
         env->bullets[i].active = 0;
     }
     
     // Key/door state will be set by generate_level()
      
      // Mark remaining agent slots as inactive
      for (int i = env->num_agents; i < MAX_AGENTS; i++) {
          env->agents[i].alive = 0;
          env->bullets[i].active = 0;
      }
      
      generate_level(env);
      compute_observations(env);
  }
  
void c_step(Picopark* env) {
    // All rewards scaled to fit within [-1, 1] to avoid PufferLib clipping
     env->tick++;
     float team_reward = 0;
      
      for (int i = 0; i < env->num_agents; i++) {
          env->rewards[i] = 0;
          env->terminals[i] = 0;
      }
      
     // Camera scrolls by 1 tile every scroll_rate ticks
     if (env->tick % env->scroll_rate == 0) {
         env->camera_col++;
         
         for (int i = 0; i < env->num_agents; i++) {
             Agent* agent = &env->agents[i];
             // Skip dead agents and agents safely inside door
             if (!agent->alive || agent->in_door) continue;
             
             // Agent dies if pushed off left edge (use integer tile position)
             if ((int)agent->col < env->camera_col) {
                 agent->alive = 0;
                team_reward = -0.5f;  // Death penalty (scaled for [-1,1])
                
                for (int j = 0; j < env->num_agents; j++) {
                    env->terminals[j] = 1;
                }
                 
                 // Calculate progress before reset
                 int max_col = 0;
                 int in_door_count = 0;
                 for (int j = 0; j < env->num_agents; j++) {
                     int agent_col = (int)env->agents[j].col;
                     if (agent_col > max_col) max_col = agent_col;
                     if (env->agents[j].in_door) in_door_count++;
                 }
                 env->log.progress += (float)max_col / (float)env->goal_col;
                 env->log.score += (float)in_door_count;  // Agents who entered door
                 env->log.perf += (float)in_door_count / (float)env->num_agents;
                 env->log.agents_at_goal += in_door_count;

               // Add terminal rewards - only penalize agents NOT in door
               // Extra penalty for the agent that died
               add_reward(env, i, -0.3f);  // Extra penalty for dying agent
               for (int j = 0; j < env->num_agents; j++) {
                   if (!env->agents[j].in_door) {
                       // Team penalty for agents who didn't make it to door
                       add_reward(env, j, team_reward);  // -0.5f
                   }
                   // Agents in_door are safe - no penalty
               }

                // Log with full episode reward (includes shaping + terminal)
                env->log.episode_length += env->tick;
                env->log.episode_return += env->total_episode_reward;
                env->log.n++;

                c_reset(env);
                return;
             }
         }
          
          // Team survived the camera scroll - progress reward scales with distance
          // NOTE: scroll_reward disabled for ablation testing.
          /*
          float progress = (float)env->camera_col / (float)env->goal_col;
          float scroll_reward = progress * 0.1f;  // More reward as you get closer to goal
          for (int i = 0; i < env->num_agents; i++) {
              add_reward(env, i, scroll_reward);
          }
          */
     }
      
     // Agent movement - sub-tile per action (MOBA-style smooth movement)
     for (int i = 0; i < env->num_agents; i++) {
         Agent* agent = &env->agents[i];
         // Skip dead agents and agents safely inside door
         if (!agent->alive || agent->in_door) continue;
          
          // Save last position for rendering interpolation
          agent->last_row = agent->row;
          agent->last_col = agent->col;
          
          int action = env->actions[i];
          float new_row = agent->row;
          float new_col = agent->col;
          
          if (action == DOWN) new_row += AGENT_SPEED;
          else if (action == UP) new_row -= AGENT_SPEED;
          else if (action == LEFT) new_col -= AGENT_SPEED;
          else if (action == RIGHT) new_col += AGENT_SPEED;
          else if (action == SHOOT && !env->bullets[i].active) {
              env->bullets[i].active = 1;
              // Bullet spawns at integer tile position of agent
              int agent_tile_row = (int)agent->row;
              int agent_tile_col = (int)agent->col;
              env->bullets[i].row = agent_tile_row;
              env->bullets[i].col = agent_tile_col + 1;
              
              // Check collision with agent at spawn position (friendly fire!)
              int victim_idx = get_agent_at(env, agent_tile_row, agent_tile_col + 1, i);
              if (victim_idx >= 0) {
                  // Friendly fire! Agent killed teammate
                  env->friendly_fire = 1;
                  env->agents[victim_idx].alive = 0;
                  env->bullets[i].active = 0;
                  
                  // Penalties (scaled for [-1,1])
                  add_reward(env, i, -0.5f);           // Heavy penalty to shooter
                  add_reward(env, victim_idx, -0.2f); // Slight penalty to victim
                  
                  // End game for all
                  for (int j = 0; j < env->num_agents; j++) {
                      env->terminals[j] = 1;
                      if (j != i && j != victim_idx) {
                          add_reward(env, j, -0.3f);  // Team penalty to others
                      }
                  }
                  
                  // Calculate progress metrics
                  int max_col = 0;
                  int in_door_count = 0;
                  for (int j = 0; j < env->num_agents; j++) {
                      int agent_col = (int)env->agents[j].col;
                      if (agent_col > max_col) max_col = agent_col;
                      if (env->agents[j].in_door) in_door_count++;
                  }
                  env->log.progress += (float)max_col / (float)env->goal_col;
                  env->log.score += (float)in_door_count;
                  env->log.perf += (float)in_door_count / (float)env->num_agents;
                  env->log.agents_at_goal += in_door_count;
                  
                  // Log and reset
                  env->log.episode_length += env->tick;
                  env->log.episode_return += env->total_episode_reward;
                  env->log.n++;
                  c_reset(env);
                  return;
              }
              
             // Check collision at spawn position with blocks (O(1) grid check)
             if (env->bullets[i].active) {
                 int spawn_row = agent_tile_row;
                 int spawn_col = agent_tile_col + 1;
                 int8_t spawn_color = get_block_color(env, spawn_row, spawn_col);
                 if (spawn_color != GRID_EMPTY) {
                    if (spawn_color == agent->color) {
                        // Destroy block: clear grid and mark block as non-existent
                       clear_block_grid(env, spawn_row, spawn_col);
                      Block* spawn_block = get_block_at(env, spawn_row, spawn_col);
                      if (spawn_block) spawn_block->exists = 0;
                      add_reward(env, i, 0.05f);  // Reward for destroying own-color block
                   }
                     env->bullets[i].active = 0;  // Bullet stops here
                 }
             }
             
             // Penalty if no blocks ahead in this row (useless shot)
             if (env->bullets[i].active) {
                 int has_block_ahead = 0;
                 for (int c = agent_tile_col + 1; c <= env->goal_col; c++) {
                     if (has_block_at(env, agent_tile_row, c)) {
                         has_block_ahead = 1;
                         break;
                     }
                 }
                if (!has_block_ahead) {
                    add_reward(env, i, -0.02f);  // Penalty for shooting with nothing to hit
                }
             }
          }
          
          if (can_agent_move(env, new_row, new_col, i)) {
              int prev_row = (int)agent->row;
              int prev_col = (int)agent->col;
              int prev_dist = -1;
              int target_row, target_col;
              
              // Phase 1: key not picked up - everyone toward key. Phase 2: door locked - only key-holder toward door. Phase 3: door unlocked - everyone toward door.
              float proximity_coef = 0.01f;
              if (env->key_holder == -1 && env->key_row >= 0) {
                  // Phase 1: Key not picked up - everyone moves toward key
                  target_row = env->key_row;
                  target_col = env->key_col;
                  prev_dist = abs(prev_col - target_col) + abs(prev_row - target_row);
                  proximity_coef = 0.01f;
              } else if (env->key_holder >= 0 && !agent->in_door) {
                  if (!env->door_unlocked) {
                      // Phase 2: Door locked - only key-holder gets door proximity (non-key-holders don't crowd door)
                      if (i == env->key_holder) {
                          target_row = env->door_row;
                          target_col = env->goal_col;
                          prev_dist = abs(prev_col - target_col) + abs(prev_row - target_row);
                          proximity_coef = 0.05f;
                      }
                  } else {
                      // Phase 3: Door unlocked - everyone rushes to door
                      target_row = env->door_row;
                      target_col = env->goal_col;
                      prev_dist = abs(prev_col - target_col) + abs(prev_row - target_row);
                      proximity_coef = 0.05f;
                  }
              }
              
              agent->row = new_row;
              agent->col = new_col;
              
              // Delta-based proximity shaping (symmetric, discourages oscillation)
              if (prev_dist >= 0) {
                  int new_dist = abs((int)agent->col - target_col) + abs((int)agent->row - target_row);
                  int delta = prev_dist - new_dist;
                  if (delta != 0) {
                      add_reward(env, i, proximity_coef * (float)delta);
                  }
              }
          }
          
     }
      
     // Key pickup check (after all movement)
     if (env->key_holder == -1 && env->key_row >= 0) {  // Key not yet picked up
         for (int i = 0; i < env->num_agents; i++) {
             Agent* agent = &env->agents[i];
             // Skip dead or in_door agents
             if (!agent->alive || agent->in_door) continue;
             int tile_row = (int)agent->row;
             int tile_col = (int)agent->col;
             if (tile_row == env->key_row && tile_col == env->key_col) {
                 // Agent picks up key
                 env->key_holder = i;
                 agent->has_key = 1;
                 env->key_row = -1;  // Key no longer on ground
                 env->key_col = -1;
                 add_reward(env, i, 0.5f);  // Reward for picking up key (scaled)
                 break;
             }
         }
     }
      
      // Bullet updates - move 1 tile right per tick
      for (int i = 0; i < env->num_agents; i++) {
          Bullet* bullet = &env->bullets[i];
          if (!bullet->active) continue;
          
          bullet->col++;
          
          // Check collision with agents first (friendly fire!)
          int victim_idx = get_agent_at(env, bullet->row, bullet->col, i);
          if (victim_idx >= 0) {
              // Friendly fire! Agent killed teammate
              env->friendly_fire = 1;
              env->agents[victim_idx].alive = 0;
              bullet->active = 0;
              
              // Penalties (scaled for [-1,1])
              add_reward(env, i, -0.5f);           // Heavy penalty to shooter
              add_reward(env, victim_idx, -0.2f); // Slight penalty to victim
              
              // End game for all
              for (int j = 0; j < env->num_agents; j++) {
                  env->terminals[j] = 1;
                  if (j != i && j != victim_idx) {
                      add_reward(env, j, -0.3f);  // Team penalty to others
                  }
              }
              
              // Calculate progress metrics
              int max_col = 0;
              int in_door_count = 0;
              for (int j = 0; j < env->num_agents; j++) {
                  int agent_col = (int)env->agents[j].col;
                  if (agent_col > max_col) max_col = agent_col;
                  if (env->agents[j].in_door) in_door_count++;
              }
              env->log.progress += (float)max_col / (float)env->goal_col;
              env->log.score += (float)in_door_count;
              env->log.perf += (float)in_door_count / (float)env->num_agents;
              env->log.agents_at_goal += in_door_count;
              
              // Log and reset
              env->log.episode_length += env->tick;
              env->log.episode_return += env->total_episode_reward;
              env->log.n++;
              c_reset(env);
              return;
          }
          
         // Check collision with blocks (O(1) grid check)
         int8_t hit_color = get_block_color(env, bullet->row, bullet->col);
         if (hit_color != GRID_EMPTY) {
            if (hit_color == env->agents[i].color) {
                // Destroy block: clear grid and mark block as non-existent
               clear_block_grid(env, bullet->row, bullet->col);
              Block* hit_block = get_block_at(env, bullet->row, bullet->col);
              if (hit_block) hit_block->exists = 0;
              add_reward(env, i, 0.05f);  // Reward for destroying own-color block
           }
             bullet->active = 0;
         }
          
          // Deactivate if past goal
          if (bullet->col > env->goal_col + 5) {
              bullet->active = 0;
          }
      }
      
      // // Progress reward - encourage moving right toward goal
      // int max_agent_col = 0;
      // for (int i = 0; i < env->num_agents; i++) {
      //     if (env->agents[i].alive && env->agents[i].col > max_agent_col) {
      //         max_agent_col = env->agents[i].col;
      //     }
      // }
      // float progress = (float)max_agent_col / (float)env->goal_col;
      // team_reward += 0.005f * progress;
      
      // Door entry check
      for (int i = 0; i < env->num_agents; i++) {
          Agent* agent = &env->agents[i];
          if (!agent->alive || agent->in_door) continue;
          
          int tile_row = (int)agent->row;
          int tile_col = (int)agent->col;
          
          // Check if agent is at door position
          if (tile_col == env->goal_col && tile_row == env->door_row) {
              if (!env->door_unlocked) {
                  // Door is locked
                  if (agent->has_key) {
                      // Key holder unlocks and enters
                      env->door_unlocked = 1;
                      agent->in_door = 1;
                      add_reward(env, i, 0.5f);  // Reward for unlocking door (scaled)
                  } else {
                      // Non-key agent tries to enter locked door - penalty
                      add_reward(env, i, -0.1f);  // Try locked door penalty (scaled)
                  }
              } else {
                  // Door is unlocked, agent can enter
                  agent->in_door = 1;
                  add_reward(env, i, 0.3f);  // Reward for entering (scaled)
             }
         }
     }
     
     // Penalty for agents past door column (encourage going to door, not past it)
     for (int i = 0; i < env->num_agents; i++) {
         Agent* agent = &env->agents[i];
         if (!agent->alive || agent->in_door) continue;
         int tile_col = (int)agent->col;
        if (tile_col > env->goal_col) {
            add_reward(env, i, -0.01f);  // Small per-step penalty for being past door
        }
     }
     
     // Win check: all agents in door
     int all_in_door = 1;
     for (int i = 0; i < env->num_agents; i++) {
         if (env->agents[i].alive && !env->agents[i].in_door) {
             all_in_door = 0;
             break;
         }
     }
     
    if (all_in_door && env->door_unlocked) {
        // WIN!
        team_reward = 1.0f;  // Scaled for [-1,1]
         for (int i = 0; i < env->num_agents; i++) {
             env->terminals[i] = 1;
         }
         
        env->log.score += (float)env->num_agents;  // Raw count: all agents in door
        env->log.perf += 1.0f;  // Normalized: 100% success
        env->log.agents_at_goal += env->num_agents;
        env->log.progress += 1.0f;  // Full progress on win
        
        // Add terminal rewards first so total_episode_reward is complete
        for (int i = 0; i < env->num_agents; i++) {
            add_reward(env, i, team_reward);  // +1.0f WIN reward
        }
         
         // Log with full episode reward
         env->log.episode_length += env->tick;
         env->log.episode_return += env->total_episode_reward;
         env->log.n++;
         
         c_reset(env);
         return;
     }
 
     compute_observations(env);
 }
  
  Client* make_client(Picopark* env) {
      int px = TILE_SIZE;
      int strip_height = 24;
      int window_w = px * env->view_width;
      int window_h = px * env->height + 2 * strip_height;
      
      InitWindow(window_w, window_h, "PufferLib PicoPark");
      SetTargetFPS(15);
      
      Client* client = (Client*)calloc(1, sizeof(Client));
      
      // Load new character sprites with two variants per color
      client->cats[0][0] = LoadTexture("resources/picopark/chars/blue1.png");
      client->cats[0][1] = LoadTexture("resources/picopark/chars/blue2.png");
      client->cats[1][0] = LoadTexture("resources/picopark/chars/gray1.png");
      client->cats[1][1] = LoadTexture("resources/picopark/chars/gray2.png");
      client->cats[2][0] = LoadTexture("resources/picopark/chars/green1.png");
      client->cats[2][1] = LoadTexture("resources/picopark/chars/green2.png");
      client->cats[3][0] = LoadTexture("resources/picopark/chars/orange1.png");
      client->cats[3][1] = LoadTexture("resources/picopark/chars/orange2.png");
      client->cats[4][0] = LoadTexture("resources/picopark/chars/pink1.png");
      client->cats[4][1] = LoadTexture("resources/picopark/chars/pink2.png");
      client->cats[5][0] = LoadTexture("resources/picopark/chars/purple1.png");
      client->cats[5][1] = LoadTexture("resources/picopark/chars/purple2.png");
      client->cats[6][0] = LoadTexture("resources/picopark/chars/red1.png");
      client->cats[6][1] = LoadTexture("resources/picopark/chars/red2.png");
      client->cats[7][0] = LoadTexture("resources/picopark/chars/yellow1.png");
      client->cats[7][1] = LoadTexture("resources/picopark/chars/yellow2.png");
      
      client->blocks[0] = LoadTexture("resources/picopark/block_blue.png");
      client->blocks[1] = LoadTexture("resources/picopark/block_gray.png");
      client->blocks[2] = LoadTexture("resources/picopark/block_green.png");
      client->blocks[3] = LoadTexture("resources/picopark/block_orange.png");
      client->blocks[4] = LoadTexture("resources/picopark/block_pink.png");
      client->blocks[5] = LoadTexture("resources/picopark/block_purple.png");
      client->blocks[6] = LoadTexture("resources/picopark/block_red.png");
      client->blocks[7] = LoadTexture("resources/picopark/block_yellow.png");
      
      client->goal = LoadTexture("resources/picopark/goal.png");
      client->bullet = LoadTexture("resources/picopark/red_bullet.png");
      client->key = LoadTexture("resources/picopark/key.png");
      client->door = LoadTexture("resources/picopark/door.png");
      
      return client;
  }
  
  void c_render(Picopark* env) {
      if (env->client == NULL) {
          env->client = make_client(env);
      }
      
      Client* client = env->client;
      int px = TILE_SIZE;
      int strip_height = 24;
      int game_area_y = strip_height;
      int window_w = px * env->view_width;
      int window_h = px * env->height + 2 * strip_height;
  
      if (IsKeyDown(KEY_ESCAPE)) {
          exit(0);
      }
  
      BeginDrawing();
      ClearBackground(WHITE);
      
      // Draw green strips
      DrawRectangle(0, 0, window_w, strip_height, (Color){76, 175, 80, 255});
      DrawRectangle(0, window_h - strip_height, window_w, strip_height, (Color){76, 175, 80, 255});
      
      // Draw door (single tile at door_row, goal_col)
      if (env->goal_col >= env->camera_col && env->goal_col < env->camera_col + env->view_width) {
          int screen_x = (env->goal_col - env->camera_col) * px;
          int screen_y = game_area_y + env->door_row * px;
          DrawTexturePro(
              client->door,
              (Rectangle){0, 0, (float)client->door.width, (float)client->door.height},
              (Rectangle){(float)screen_x, (float)screen_y, (float)px, (float)px},
              (Vector2){0, 0}, 0.0f, WHITE
          );
      }
      
      // Draw blocks with border+fill style (no gaps between tiles)
      int border_thickness = 3;  // Pixels for border
      for (int i = 0; i < env->num_blocks; i++) {
          Block* block = &env->blocks[i];
          if (!block->exists) continue;
          if (block->col < env->camera_col || block->col >= env->camera_col + env->view_width) continue;
          
          int screen_x = (block->col - env->camera_col) * px;
          int screen_y = game_area_y + block->row * px;
          
          Color border_color = BLOCK_COLORS[block->color][0];
          Color fill_color = BLOCK_COLORS[block->color][1];
          
          // Draw border (full tile)
          DrawRectangle(screen_x, screen_y, px, px, border_color);
          // Draw fill (inset by border thickness)
          DrawRectangle(screen_x + border_thickness, screen_y + border_thickness, 
                        px - 2 * border_thickness, px - 2 * border_thickness, fill_color);
      }
      
      // Draw key on ground (if not picked up)
      if (env->key_row >= 0 && env->key_col >= env->camera_col && 
          env->key_col < env->camera_col + env->view_width) {
          int screen_x = (env->key_col - env->camera_col) * px;
          int screen_y = game_area_y + env->key_row * px;
          // Draw key smaller, centered in tile
          float key_size = px * 0.6f;
          float offset = (px - key_size) / 2.0f;
          DrawTexturePro(
              client->key,
              (Rectangle){0, 0, (float)client->key.width, (float)client->key.height},
              (Rectangle){screen_x + offset, screen_y + offset, key_size, key_size},
              (Vector2){0, 0}, 0.0f, WHITE
          );
      }
      
      // Draw bullets (red bullet sprite - maintain aspect ratio)
      for (int i = 0; i < env->num_agents; i++) {
          Bullet* bullet = &env->bullets[i];
          if (!bullet->active) continue;
          if (bullet->col < env->camera_col || bullet->col >= env->camera_col + env->view_width) continue;
          
          // Scale bullet while maintaining aspect ratio
          float scale = (float)(px * 2 / 3) / (float)client->bullet.width;  // Scale based on width
          int bullet_w = (int)(client->bullet.width * scale);
          int bullet_h = (int)(client->bullet.height * scale);
          
          int screen_x = (bullet->col - env->camera_col) * px + (px - bullet_w) / 2;  // Center in tile
          int screen_y = game_area_y + bullet->row * px + (px - bullet_h) / 2;
          
          DrawTexturePro(
              client->bullet,
              (Rectangle){0, 0, (float)client->bullet.width, (float)client->bullet.height},
              (Rectangle){(float)screen_x, (float)screen_y, (float)bullet_w, (float)bullet_h},
              (Vector2){0, 0}, 0.0f, WHITE
          );
      }
      
      // Draw agents (collision is 2 tiles, visual is 1.3x tile height)
      for (int i = 0; i < env->num_agents; i++) {
          Agent* agent = &env->agents[i];
          if (!agent->alive) continue;
          if (agent->col < env->camera_col || agent->col >= env->camera_col + env->view_width) continue;
          
          Texture2D cat_tex = client->cats[agent->color][agent->variant];
          
          // Visual height: 1.3x tile size (smaller than collision box)
          float visual_height = px * 1.3f;
          // Maintain aspect ratio based on original sprite dimensions
          float aspect_ratio = (float)cat_tex.width / (float)cat_tex.height;
          float visual_width = visual_height * aspect_ratio;
          
          // Center horizontally within tile
          float screen_x = (agent->col - env->camera_col) * px + (px - visual_width) / 2.0f;
          // Position at bottom of collision box (feet on ground)
          float screen_y = game_area_y + (agent->row + 1) * px - visual_height;
          
          DrawTexturePro(
              cat_tex,
              (Rectangle){0, 0, (float)cat_tex.width, (float)cat_tex.height},
              (Rectangle){screen_x, screen_y, visual_width, visual_height},
              (Vector2){0, 0}, 0.0f, WHITE
          );
      }
      
      // Draw key following key-holder agent
      if (env->key_holder >= 0) {
          Agent* holder = &env->agents[env->key_holder];
          if (holder->alive && holder->col >= env->camera_col && 
              holder->col < env->camera_col + env->view_width) {
              float holder_screen_x = (holder->col - env->camera_col) * px;
              float holder_screen_y = game_area_y + (holder->row - 0.8f) * px;  // Above agent's head
              float key_size = px * 0.4f;
              DrawTexturePro(
                  client->key,
                  (Rectangle){0, 0, (float)client->key.width, (float)client->key.height},
                  (Rectangle){holder_screen_x + px/2 - key_size/2, holder_screen_y, key_size, key_size},
                  (Vector2){0, 0}, 0.0f, WHITE
              );
          }
      }
      
      // HUD
      DrawText(TextFormat("Tick: %d  Camera: %d  Blocks: %d", env->tick, env->camera_col, env->num_blocks), 10, 4, 16, WHITE);
  
      EndDrawing();
  }
  
  void c_close(Picopark* env) {
      if (env->client != NULL) {
          Client* client = env->client;
          
          for (int i = 0; i < NUM_COLORS; i++) {
              UnloadTexture(client->cats[i][0]);
              UnloadTexture(client->cats[i][1]);
              UnloadTexture(client->blocks[i]);
          }
          UnloadTexture(client->goal);
          UnloadTexture(client->bullet);
          UnloadTexture(client->key);
          UnloadTexture(client->door);
          
          free(client);
          env->client = NULL;
      }
      
      if (IsWindowReady()) {
          CloseWindow();
      }
  }