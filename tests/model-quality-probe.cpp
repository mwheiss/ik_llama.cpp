#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "usage: logit-probe config.json output-directory");
        std::ifstream input(argv[1]);
        json config; input >> config;
        std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = config.value("n_gpu_layers", 0);
        auto * model = llama_model_load_from_file(config.at("model").get<std::string>().c_str(), mp);
        require(model != nullptr, "model load failed");
        auto * vocab = llama_model_get_vocab(model);
        const auto * tokenizer = model;
        const int n_vocab = llama_vocab_n_tokens(vocab);
        auto cp = llama_context_default_params();
        cp.n_ctx = config.value("n_ctx", 256);
        cp.n_batch = config.value("n_batch", 32);
        cp.n_ubatch = config.value("n_ubatch", 32);
        cp.n_seq_max = config.value("n_seq_max", 2);
        cp.n_threads = cp.n_threads_batch = config.value("threads", 4);
        cp.type_k = config.value("cache_k", "f32") == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.type_v = config.value("cache_v", "f32") == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.rope_freq_scale = config.value("rope_freq_scale", 1.0f);
        cp.rope_freq_base = config.value("rope_freq_base", 0.0f);
        cp.flash_attn = config.value("flash_attn", false);
        cp.graph_reuse = config.value("graph_reuse", true);
        cp.fused_moe_up_gate = config.value("fused_moe_up_gate", true);
        cp.fused_up_gate = config.value("fused_up_gate", true);
        cp.fused_mmad = config.value("fused_mmad", true);
        auto * ctx = llama_init_from_model(model, cp);
        require(ctx != nullptr, "context creation failed");

        llama_lora_adapter * adapter = nullptr;
        if (config.contains("adapter")) {
            adapter = llama_lora_adapter_init(model, config.at("adapter").get<std::string>().c_str());
            require(adapter != nullptr, "adapter load failed");
        }
        auto set_adapter = [&](float scale) {
            if (!adapter) return;
            llama_lora_adapter_clear(ctx);
            if (scale != 0) require(llama_lora_adapter_set(ctx, adapter, scale) == 0, "adapter update failed");
        };

        std::vector<llama_token> prompt;
        if (config.contains("prompt_tokens")) {
            prompt = config.at("prompt_tokens").get<std::vector<llama_token>>();
        } else {
            const std::string text = config.at("prompt");
            prompt.resize(text.size() * 4 + 128);
            int n = llama_tokenize(tokenizer, text.data(), text.size(), prompt.data(), prompt.size(), true, true);
            require(n >= 0, "token buffer too small");
            prompt.resize(n);
        }
        require(!prompt.empty(), "empty tokenized prompt");
        int position = 0;
        json shifts = json::array();
        auto batch = llama_batch_init(cp.n_batch, 0, 1);
        auto evaluate = [&](const std::vector<llama_token> & tokens, bool final_output) {
            for (size_t offset = 0; offset < tokens.size(); offset += cp.n_batch) {
                int count = std::min<size_t>(cp.n_batch, tokens.size() - offset);
                int limit = config.value("context_limit", 0);
                if (limit && position + count >= limit) {
                    int keep = 1, discard = (position - keep) / 2;
                    require(discard > 0, "context limit too small for batch");
                    require(llama_kv_cache_seq_rm(ctx, 0, keep, keep + discard), "KV remove failed");
                    llama_kv_cache_seq_add(ctx, 0, keep + discard, -1, -discard);
                    shifts.push_back({{"position",position},{"keep",keep},{"discard",discard}});
                    position -= discard;
                }
                batch.n_tokens = count;
                for (int i = 0; i < count; ++i) {
                    batch.token[i] = tokens[offset + i];
                    batch.pos[i] = position++;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = final_output && offset + i + 1 == tokens.size();
                }
                require(llama_decode(ctx, batch) == 0, "decode failed");
            }
        };

        if (adapter && config.contains("hot_from_scale")) {
            set_adapter(config.at("hot_from_scale"));
            evaluate(prompt, true);
            // Preserve the reusable graph while making the logical prompt state empty.
            llama_kv_cache_clear(ctx);
            position = 0;
        }
        set_adapter(config.value("adapter_scale", 0.0f));
        evaluate(prompt, true);
        std::vector<float> rows;
        std::vector<llama_token> generated;
        std::vector<llama_token> evaluated = prompt;
        auto save_logits = [&]() {
            const float * logits = llama_get_logits_ith(ctx, -1);
            require(logits != nullptr, "missing logits");
            rows.insert(rows.end(), logits, logits + n_vocab);
            return int(std::max_element(logits, logits + n_vocab) - logits);
        };
        int token = save_logits();
        if (config.contains("teacher_tokens")) {
            for (int forced : config.at("teacher_tokens").get<std::vector<int>>()) {
                evaluate({forced}, true);
                evaluated.push_back(forced);
                save_logits();
            }
        } else {
            int count = config.value("n_predict", 8);
            for (int i = 0; i < count; ++i) {
                generated.push_back(token);
                if (i + 1 < count) {
                    evaluate({token}, true);
                    evaluated.push_back(token);
                    token = save_logits();
                }
            }
        }
        std::string content;
        for (int id : generated) {
            char piece[512];
            int n = llama_token_to_piece(tokenizer, id, piece, sizeof(piece), 0, false);
            require(n >= 0, "token piece buffer too small");
            content.append(piece, n);
        }
        std::ofstream raw(output / "logits.f32", std::ios::binary);
        raw.write(reinterpret_cast<const char *>(rows.data()), rows.size() * sizeof(float));
        json meta = {
            {"config", config}, {"context_shifts",shifts},
            {"n_vocab", n_vocab}, {"n_logit_rows", rows.size() / n_vocab},
            {"actual_n_ctx", llama_n_ctx(ctx)}, {"actual_n_batch", llama_n_batch(ctx)},
            {"actual_n_ubatch", llama_n_ubatch(ctx)},
            {"prompt_tokens", prompt}, {"evaluated_tokens", evaluated},
            {"generated_tokens", generated}, {"content", content},
        };
        std::ofstream(output / "result.json") << meta.dump(2) << "\n";
        std::cout << json{{"output", output.string()}, {"rows", rows.size() / n_vocab},
                          {"content", content}}.dump() << "\n";
        llama_batch_free(batch);
        llama_free(ctx);
        if (adapter) {
            llama_lora_adapter_free(adapter);
        }
        llama_free_model(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
