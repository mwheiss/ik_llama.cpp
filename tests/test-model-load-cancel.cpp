#include "llama.h"
#include "get-model.h"

#include <cstdlib>

int main(int argc, char *argv[] ) {
    auto * model_path = get_model_or_exit(argc, argv);
    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    bool cancellation_requested = false;
    params.progress_callback_user_data = &cancellation_requested;
    params.progress_callback = [](float, void * ctx) {
        *static_cast<bool *>(ctx) = true;
        return false;
    };
    auto * model = llama_model_load_from_file(model_path, params);
    const bool cancelled = model == nullptr && cancellation_requested;
    llama_free_model(model);
    llama_backend_free();
    return cancelled ? EXIT_SUCCESS : EXIT_FAILURE;
}
