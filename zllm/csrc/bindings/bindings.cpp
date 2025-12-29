// =============================================================================
// bindings.cpp - CUDA Bindings for Python
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <iostream>

// Include our CUDA kernels
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/swiglu.cuh"
#include "../attention/flash_attention.cuh"
#include "../attention/flash_infer.cuh"
#include "../memory/memory_pool.cuh"

namespace py = pybind11;

// =============================================================================
// Helper Functions
// =============================================================================

// Check if array is contiguous and has correct dtype
template<typename T>
void check_array(py::array_t<T> &arr, const std::string &name) {
    if (!arr.is_c_contiguous()) {
        throw std::runtime_error(name + " array must be C-contiguous");
    }
}

// =============================================================================
// RMSNorm Bindings
// =============================================================================

void bind_rmsnorm(py::module &m) {
    m.def("rmsnorm_forward", [](py::array_t<float> input, 
                                py::array_t<float> weight, 
                                float epsilon) {
        check_array(input, "input");
        check_array(weight, "weight");
        
        auto input_buf = input.request();
        auto weight_buf = weight.request();
        
        int num_tokens = input_buf.shape[0];
        int hidden_dim = input_buf.shape[1];
        
        py::array_t<float> output({num_tokens, hidden_dim});
        auto output_buf = output.request();
        
        float *input_ptr = static_cast<float *>(input_buf.ptr);
        float *weight_ptr = static_cast<float *>(weight_buf.ptr);
        float *output_ptr = static_cast<float *>(output_buf.ptr);
        
        // Use optimized version
        mini_vllm::rmsnorm_forward_optimized(output_ptr, input_ptr, weight_ptr,
                                           num_tokens, hidden_dim, epsilon, 0);
        
        return output;
    }, py::arg("input"), py::arg("weight"), py::arg("epsilon") = 1e-6);

    m.def("rmsnorm_forward_fp16", [](py::array_t<uint16_t> input, 
                                     py::array_t<uint16_t> weight, 
                                     float epsilon) {
        check_array(input, "input");
        check_array(weight, "weight");
        
        auto input_buf = input.request();
        auto weight_buf = weight.request();
        
        int num_tokens = input_buf.shape[0];
        int hidden_dim = input_buf.shape[1];
        
        py::array_t<uint16_t> output({num_tokens, hidden_dim});
        auto output_buf = output.request();
        
        half *input_ptr = reinterpret_cast<half *>(input_buf.ptr);
        half *weight_ptr = reinterpret_cast<half *>(weight_buf.ptr);
        half *output_ptr = reinterpret_cast<half *>(output_buf.ptr);
        
        mini_vllm::rmsnorm_forward_fp16(output_ptr, input_ptr, weight_ptr,
                                       num_tokens, hidden_dim, epsilon, 0);
        
        return output;
    }, py::arg("input"), py::arg("weight"), py::arg("epsilon") = 1e-6);
}

// =============================================================================
// RoPE Bindings
// =============================================================================

void bind_rope(py::module &m) {
    m.def("rope_init_tables", [](int max_seq_len, int head_dim, float theta_base) {
        int half_head_dim = head_dim / 2;
        
        py::array_t<float> cos_table({max_seq_len, half_head_dim});
        py::array_t<float> sin_table({max_seq_len, half_head_dim});
        
        auto cos_buf = cos_table.request();
        auto sin_buf = sin_table.request();
        
        float *cos_ptr = static_cast<float *>(cos_buf.ptr);
        float *sin_ptr = static_cast<float *>(sin_buf.ptr);
        
        mini_vllm::rope_init_tables(cos_ptr, sin_ptr, max_seq_len, head_dim,
                                   theta_base, 0);
        
        return py::make_tuple(cos_table, sin_table);
    }, py::arg("max_seq_len"), py::arg("head_dim"), py::arg("theta_base") = 10000.0f);

    m.def("rope_forward", [](py::array_t<float> query, 
                             py::array_t<float> key,
                             py::array_t<float> cos_table,
                             py::array_t<float> sin_table,
                             py::array_t<int> positions,
                             int num_heads, int num_kv_heads, int head_dim) {
        check_array(query, "query");
        check_array(key, "key");
        check_array(cos_table, "cos_table");
        check_array(sin_table, "sin_table");
        check_array(positions, "positions");
        
        auto query_buf = query.request();
        auto key_buf = key.request();
        auto cos_buf = cos_table.request();
        auto sin_buf = sin_table.request();
        auto pos_buf = positions.request();
        
        int num_tokens = query_buf.shape[0];
        
        float *query_ptr = static_cast<float *>(query_buf.ptr);
        float *key_ptr = static_cast<float *>(key_buf.ptr);
        float *cos_ptr = static_cast<float *>(cos_buf.ptr);
        float *sin_ptr = static_cast<float *>(sin_buf.ptr);
        int *pos_ptr = static_cast<int *>(pos_buf.ptr);
        
        mini_vllm::rope_forward(query_ptr, key_ptr, cos_ptr, sin_ptr, pos_ptr,
                               num_tokens, num_heads, num_kv_heads, head_dim, 0);
        
        return py::make_tuple(query, key);
    }, py::arg("query"), py::arg("key"), py::arg("cos_table"), 
       py::arg("sin_table"), py::arg("positions"), py::arg("num_heads"),
       py::arg("num_kv_heads"), py::arg("head_dim"));

    m.def("rope_forward_fp16", [](py::array_t<uint16_t> query, 
                                  py::array_t<uint16_t> key,
                                  py::array_t<float> cos_table,
                                  py::array_t<float> sin_table,
                                  py::array_t<int> positions,
                                  int num_heads, int num_kv_heads, int head_dim) {
        check_array(query, "query");
        check_array(key, "key");
        check_array(cos_table, "cos_table");
        check_array(sin_table, "sin_table");
        check_array(positions, "positions");
        
        auto query_buf = query.request();
        auto key_buf = key.request();
        auto cos_buf = cos_table.request();
        auto sin_buf = sin_table.request();
        auto pos_buf = positions.request();
        
        int num_tokens = query_buf.shape[0];
        
        half *query_ptr = reinterpret_cast<half *>(query_buf.ptr);
        half *key_ptr = reinterpret_cast<half *>(key_buf.ptr);
        float *cos_ptr = static_cast<float *>(cos_buf.ptr);
        float *sin_ptr = static_cast<float *>(sin_buf.ptr);
        int *pos_ptr = static_cast<int *>(pos_buf.ptr);
        
        mini_vllm::rope_forward_fp16(query_ptr, key_ptr, cos_ptr, sin_ptr, pos_ptr,
                                    num_tokens, num_heads, num_kv_heads, head_dim, 0);
        
        return py::make_tuple(query, key);
    }, py::arg("query"), py::arg("key"), py::arg("cos_table"), 
       py::arg("sin_table"), py::arg("positions"), py::arg("num_heads"),
       py::arg("num_kv_heads"), py::arg("head_dim"));
}

// =============================================================================
// SwiGLU Bindings
// =============================================================================

void bind_swiglu(py::module &m) {
    m.def("swiglu_forward", [](py::array_t<float> gate, 
                               py::array_t<float> up) {
        check_array(gate, "gate");
        check_array(up, "up");
        
        auto gate_buf = gate.request();
        auto up_buf = up.request();
        
        if (gate_buf.shape[0] != up_buf.shape[0] || 
            gate_buf.shape[1] != up_buf.shape[1]) {
            throw std::runtime_error("gate and up must have same shape");
        }
        
        int num_tokens = gate_buf.shape[0];
        int intermediate_dim = gate_buf.shape[1];
        
        py::array_t<float> output({num_tokens, intermediate_dim});
        auto output_buf = output.request();
        
        float *gate_ptr = static_cast<float *>(gate_buf.ptr);
        float *up_ptr = static_cast<float *>(up_buf.ptr);
        float *output_ptr = static_cast<float *>(output_buf.ptr);
        
        mini_vllm::swiglu_forward(output_ptr, gate_ptr, up_ptr, num_tokens,
                                intermediate_dim, 0);
        
        return output;
    }, py::arg("gate"), py::arg("up"));

    m.def("swiglu_forward_fp16", [](py::array_t<uint16_t> gate, 
                                    py::array_t<uint16_t> up) {
        check_array(gate, "gate");
        check_array(up, "up");
        
        auto gate_buf = gate.request();
        auto up_buf = up.request();
        
        if (gate_buf.shape[0] != up_buf.shape[0] || 
            gate_buf.shape[1] != up_buf.shape[1]) {
            throw std::runtime_error("gate and up must have same shape");
        }
        
        int num_tokens = gate_buf.shape[0];
        int intermediate_dim = gate_buf.shape[1];
        
        py::array_t<uint16_t> output({num_tokens, intermediate_dim});
        auto output_buf = output.request();
        
        half *gate_ptr = reinterpret_cast<half *>(gate_buf.ptr);
        half *up_ptr = reinterpret_cast<half *>(up_buf.ptr);
        half *output_ptr = reinterpret_cast<half *>(output_buf.ptr);
        
        mini_vllm::swiglu_forward_fp16(output_ptr, gate_ptr, up_ptr, num_tokens,
                                      intermediate_dim, 0);
        
        return output;
    }, py::arg("gate"), py::arg("up"));
}

// =============================================================================
// Flash Attention Bindings
// =============================================================================

void bind_flash_attention(py::module &m) {
    py::class_<mini_vllm::FlashAttentionConfig>(m, "FlashAttentionConfig")
        .def(py::init<>())
        .def_readwrite("block_size", &mini_vllm::FlashAttentionConfig::block_size)
        .def_readwrite("num_heads", &mini_vllm::FlashAttentionConfig::num_heads)
        .def_readwrite("num_kv_heads", &mini_vllm::FlashAttentionConfig::num_kv_heads)
        .def_readwrite("head_dim", &mini_vllm::FlashAttentionConfig::head_dim)
        .def_readwrite("max_seq_len", &mini_vllm::FlashAttentionConfig::max_seq_len)
        .def_readwrite("softmax_scale", &mini_vllm::FlashAttentionConfig::softmax_scale)
        .def_readwrite("is_causal", &mini_vllm::FlashAttentionConfig::is_causal)
        .def_readwrite("dtype_size", &mini_vllm::FlashAttentionConfig::dtype_size);

    m.def("flash_attention_prefill_forward", [](py::array_t<float> query, 
                                                 py::array_t<float> key,
                                                 py::array_t<float> value,
                                                 py::array_t<int> positions,
                                                 mini_vllm::FlashAttentionConfig config) {
        check_array(query, "query");
        check_array(key, "key");
        check_array(value, "value");
        check_array(positions, "positions");
        
        auto query_buf = query.request();
        auto key_buf = key.request();
        auto value_buf = value.request();
        auto pos_buf = positions.request();
        
        int num_tokens = query_buf.shape[0];
        
        py::array_t<float> output({num_tokens, config.num_heads, config.head_dim});
        auto output_buf = output.request();
        
        float *output_ptr = static_cast<float *>(output_buf.ptr);
        float *query_ptr = static_cast<float *>(query_buf.ptr);
        float *key_ptr = static_cast<float *>(key_buf.ptr);
        float *value_ptr = static_cast<float *>(value_buf.ptr);
        int *pos_ptr = static_cast<int *>(pos_buf.ptr);
        
        mini_vllm::flash_attention_prefill_forward(output_ptr, query_ptr, key_ptr, value_ptr,
                                                  pos_ptr, num_tokens, config, 0);
        
        return output;
    }, py::arg("query"), py::arg("key"), py::arg("value"), py::arg("positions"),
       py::arg("config"));

    m.def("flash_attention_prefill_forward_fp16", [](py::array_t<uint16_t> query, 
                                                      py::array_t<uint16_t> key,
                                                      py::array_t<uint16_t> value,
                                                      py::array_t<int> positions,
                                                      mini_vllm::FlashAttentionConfig config) {
        check_array(query, "query");
        check_array(key, "key");
        check_array(value, "value");
        check_array(positions, "positions");
        
        auto query_buf = query.request();
        auto key_buf = key.request();
        auto value_buf = value.request();
        auto pos_buf = positions.request();
        
        int num_tokens = query_buf.shape[0];
        
        py::array_t<uint16_t> output({num_tokens, config.num_heads, config.head_dim});
        auto output_buf = output.request();
        
        half *output_ptr = reinterpret_cast<half *>(output_buf.ptr);
        half *query_ptr = reinterpret_cast<half *>(query_buf.ptr);
        half *key_ptr = reinterpret_cast<half *>(key_buf.ptr);
        half *value_ptr = reinterpret_cast<half *>(value_buf.ptr);
        int *pos_ptr = static_cast<int *>(pos_buf.ptr);
        
        mini_vllm::flash_attention_prefill_forward_fp16(output_ptr, query_ptr, key_ptr, value_ptr,
                                                       pos_ptr, num_tokens, config, 0);
        
        return output;
    }, py::arg("query"), py::arg("key"), py::arg("value"), py::arg("positions"),
       py::arg("config"));
}

// =============================================================================
// FlashInfer Bindings
// =============================================================================

void bind_flash_infer(py::module &m) {
    py::class_<mini_vllm::FlashInferConfig>(m, "FlashInferConfig")
        .def(py::init<>())
        .def_readwrite("num_heads", &mini_vllm::FlashInferConfig::num_heads)
        .def_readwrite("num_kv_heads", &mini_vllm::FlashInferConfig::num_kv_heads)
        .def_readwrite("head_dim", &mini_vllm::FlashInferConfig::head_dim)
        .def_readwrite("softmax_scale", &mini_vllm::FlashInferConfig::softmax_scale)
        .def_readwrite("dtype_size", &mini_vllm::FlashInferConfig::dtype_size);

    m.def("flash_infer_forward", [](py::array_t<float> query, 
                                     py::array_t<float> key_cache,
                                     py::array_t<float> value_cache,
                                     py::array_t<int> block_table,
                                     py::array_t<int> block_offsets,
                                     py::array_t<int> seq_lengths,
                                     mini_vllm::FlashInferConfig config) {
        check_array(query, "query");
        check_array(key_cache, "key_cache");
        check_array(value_cache, "value_cache");
        check_array(block_table, "block_table");
        check_array(block_offsets, "block_offsets");
        check_array(seq_lengths, "seq_lengths");
        
        auto query_buf = query.request();
        auto key_cache_buf = key_cache.request();
        auto value_cache_buf = value_cache.request();
        auto block_table_buf = block_table.request();
        auto block_offsets_buf = block_offsets.request();
        auto seq_lengths_buf = seq_lengths.request();
        
        int num_tokens = query_buf.shape[0];
        int num_sequences = seq_lengths_buf.shape[0];
        
        py::array_t<float> output({num_tokens, config.num_heads, config.head_dim});
        auto output_buf = output.request();
        
        float *output_ptr = static_cast<float *>(output_buf.ptr);
        float *query_ptr = static_cast<float *>(query_buf.ptr);
        float *key_cache_ptr = static_cast<float *>(key_cache_buf.ptr);
        float *value_cache_ptr = static_cast<float *>(value_cache_buf.ptr);
        int *block_table_ptr = static_cast<int *>(block_table_buf.ptr);
        int *block_offsets_ptr = static_cast<int *>(block_offsets_buf.ptr);
        int *seq_lengths_ptr = static_cast<int *>(seq_lengths_buf.ptr);
        
        mini_vllm::flash_infer_forward(output_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
                                      block_table_ptr, block_offsets_ptr, seq_lengths_ptr,
                                      num_tokens, num_sequences, config, 0);
        
        return output;
    }, py::arg("query"), py::arg("key_cache"), py::arg("value_cache"),
       py::arg("block_table"), py::arg("block_offsets"), py::arg("seq_lengths"),
       py::arg("config"));

    m.def("flash_infer_forward_fp16", [](py::array_t<uint16_t> query, 
                                          py::array_t<uint16_t> key_cache,
                                          py::array_t<uint16_t> value_cache,
                                          py::array_t<int> block_table,
                                          py::array_t<int> block_offsets,
                                          py::array_t<int> seq_lengths,
                                          mini_vllm::FlashInferConfig config) {
        check_array(query, "query");
        check_array(key_cache, "key_cache");
        check_array(value_cache, "value_cache");
        check_array(block_table, "block_table");
        check_array(block_offsets, "block_offsets");
        check_array(seq_lengths, "seq_lengths");
        
        auto query_buf = query.request();
        auto key_cache_buf = key_cache.request();
        auto value_cache_buf = value_cache.request();
        auto block_table_buf = block_table.request();
        auto block_offsets_buf = block_offsets.request();
        auto seq_lengths_buf = seq_lengths.request();
        
        int num_tokens = query_buf.shape[0];
        int num_sequences = seq_lengths_buf.shape[0];
        
        py::array_t<uint16_t> output({num_tokens, config.num_heads, config.head_dim});
        auto output_buf = output.request();
        
        half *output_ptr = reinterpret_cast<half *>(output_buf.ptr);
        half *query_ptr = reinterpret_cast<half *>(query_buf.ptr);
        half *key_cache_ptr = reinterpret_cast<half *>(key_cache_buf.ptr);
        half *value_cache_ptr = reinterpret_cast<half *>(value_cache_buf.ptr);
        int *block_table_ptr = static_cast<int *>(block_table_buf.ptr);
        int *block_offsets_ptr = static_cast<int *>(block_offsets_buf.ptr);
        int *seq_lengths_ptr = static_cast<int *>(seq_lengths_buf.ptr);
        
        mini_vllm::flash_infer_forward_fp16(output_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
                                           block_table_ptr, block_offsets_ptr, seq_lengths_ptr,
                                           num_tokens, num_sequences, config, 0);
        
        return output;
    }, py::arg("query"), py::arg("key_cache"), py::arg("value_cache"),
       py::arg("block_table"), py::arg("block_offsets"), py::arg("seq_lengths"),
       py::arg("config"));
}

// =============================================================================
// Memory Pool Bindings
// =============================================================================

void bind_memory_pool(py::module &m) {
    py::class_<mini_vllm::MemoryManager>(m, "MemoryManager")
        .def(py::init<>())
        .def("init", &mini_vllm::MemoryManager::init,
             py::arg("kv_cache_gb"), py::arg("block_size_tokens"),
             py::arg("num_kv_heads"), py::arg("head_dim"),
             py::arg("activation_gb") = 1.0f, py::arg("workspace_mb") = 256,
             py::arg("dtype_size") = 2)
        .def("get_memory_stats", &mini_vllm::MemoryManager::get_memory_stats)
        .def("print_stats", &mini_vllm::MemoryManager::print_stats);

    py::class_<mini_vllm::PoolStats>(m, "PoolStats")
        .def(py::init<>())
        .def_readwrite("total_bytes", &mini_vllm::PoolStats::total_bytes)
        .def_readwrite("used_bytes", &mini_vllm::PoolStats::used_bytes)
        .def_readwrite("peak_bytes", &mini_vllm::PoolStats::peak_bytes)
        .def_readwrite("num_allocations", &mini_vllm::PoolStats::num_allocations)
        .def_readwrite("num_frees", &mini_vllm::PoolStats::num_frees)
        .def("utilization", &mini_vllm::PoolStats::utilization);
}

// =============================================================================
// Module Definition
// =============================================================================

PYBIND11_MODULE(mini_vllm_cuda, m) {
    m.doc() = "Mini-vLLM CUDA kernels and bindings";

    // Initialize CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "Using CUDA device: " << prop.name << " (Compute " 
              << prop.major << "." << prop.minor << ")" << std::endl;

    // Bind all functions
    bind_rmsnorm(m);
    bind_rope(m);
    bind_swiglu(m);
    bind_flash_attention(m);
    bind_flash_infer(m);
    bind_memory_pool(m);

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("cuda_version") = prop.major * 10 + prop.minor;
}