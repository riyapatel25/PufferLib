#include "picopark.h"

#define Env Picopark
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->view_width = unpack(kwargs, "view_width");
    env->view_height = unpack(kwargs, "view_height");
    env->height = unpack(kwargs, "height");
    env->scroll_rate = unpack(kwargs, "scroll_rate");
    env->num_agents = unpack(kwargs, "num_agents");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "agents_at_goal", log->agents_at_goal);
    assign_to_dict(dict, "progress", log->progress);
    return 0;
}
