#include <torch/nn/modules/rnn.h>

#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <c10/util/Exception.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace nn {
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNImplBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace detail {
template <typename Derived>
RNNImplBase<Derived>::RNNImplBase(RNNOptionsBase options_)
  : options(std::move(options_)) {
  reset();
}

template <typename Derived>
void RNNImplBase<Derived>::reset() {
  const int64_t num_directions = options.bidirectional() ? 2 : 1;

  TORCH_CHECK(
    0 <= options.dropout() && options.dropout() <= 1,
    "dropout should be a number in range [0, 1] ",
    "representing the probability of an element being ",
    "zeroed");

  if (options.dropout() > 0 && options.num_layers() == 1) {
    TORCH_WARN(
      "dropout option adds dropout after all but last ",
      "recurrent layer, so non-zero dropout expects ",
      "num_layers greater than 1, but got dropout=", options.dropout(), " and ",
      "num_layers=", options.num_layers());
    )
  }

  int64_t gate_size = 0;
  if (c10::get_if<enumtype::kLSTM>(&options.mode())) {
    gate_size = 4 * options.hidden_size();
  } else if (c10::get_if<enumtype::kGRU>(&options.mode())) {
    gate_size = 3 * options.hidden_size();
  } else if (c10::get_if<enumtype::kRNN_TANH>(&options.mode())) {
    gate_size = options.hidden_size();
  } else if (c10::get_if<enumtype::kRNN_RELU>(&options.mode())) {
    gate_size = options.hidden_size();
  } else {
    TORCH_CHECK(false, "Unrecognized RNN mode: " + torch::enumtype::get_enum_name(v));
  }

  _flat_weights_names = {};
  _all_weights = {};

  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t direction = 0; direction < num_directions; direction++) {
      int64_t layer_input_size = layer == 0 ? options.input_size() : options.hidden_size() * num_directions;

      auto w_ih = torch::empty({gate_size, layer_input_size});
      auto w_hh = torch::empty({gate_size, hidden_size});
      auto b_ih = torch::empty({gate_size});
      // Second bias vector included for CuDNN compatibility. Only one
      // bias vector is needed in standard definition.
      auto b_hh = torch::empty({gate_size});
      std::vector<Tensor> layer_params = {w_ih, w_hh, b_ih, b_hh};

      std::string suffix = direction == 1 ? "_reverse" : "";
      std::vector<std::string> param_names = {"weight_ih_l{layer}{suffix}", 'weight_hh_l{layer}{suffix}'};
      if (options.bias()) {
        param_names.emplace_back("bias_ih_l{layer}{suffix}");
        param_names.emplace_back("bias_hh_l{layer}{suffix}");
      }
      for (size_t i = 0; i < param_names.size(); i++) {
        std::string x = std::regex_replace(param_names[i], std::regex("\\{layer}"), layer);
        x = std::regex_replace(x, std::regex("\\{suffix}"), suffix);
        param_names[i] = x;
      }

      for (size_t i = 0; i < param_names.size(); i++) {
        auto name = param_names[i];
        auto param = layer_params[i];
        this->register_parameter(name, param);
      }
      _flat_weights_names.insert(_flat_weights_names.end(), param_names.begin(), param_names.end());
      _all_weights.emplace_back(param_names);
    }
  }

  _flat_weights = {};
  for (const auto& wn : _flat_weights_names) {
    auto named_parameters = this->named_parameters(/*recurse=*/false);
    if (named_parameters.contains(wn)) {
      _flat_weights.emplace_back(named_parameters[wn]);
    } else {
      _flat_weights.emplace_back(Tensor());
    }
  }

  this->flatten_parameters();
  this->reset_parameters(); 
}

template <typename Derived>
void RNNImplBase<Derived>::flatten_parameters() {
  // Resets parameter data pointer so that they can use faster code paths.
  //
  // Right now, this works only if the module is on the GPU and cuDNN is enabled.
  // Otherwise, it's a no-op.

  // Short-circuits if _flat_weights is only partially instantiated
  if (_flat_weights.size() != _flat_weights_names.size()) {
    return;
  }

  // Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
  // or the tensors in _flat_weights are of different dtypes

  auto first_fw = _flat_weights[0];
  auto dtype = first_fw.dtype();
  for (const auto& fw : _flat_weights) {
    if (!(fw.dtype() == dtype) ||
        !fw.is_cuda() ||
        !torch::cudnn_is_acceptable(fw)) {
      return;
    }
  }

  // If any parameters alias, we fall back to the slower, copying code path. This is
  // a sufficient check, because overlapping parameter buffers that don't completely
  // alias would break the assumptions of the uniqueness check in
  // Module::named_parameters().

  // yf225 TODO: fix the rest!
  unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
  if len(unique_data_ptrs) != len(self._flat_weights):
      return

  with torch.cuda.device_of(first_fw):
      import torch.backends.cudnn.rnn as rnn

      # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
      # an inplace operation on self._flat_weights
      with torch.no_grad():
          torch._cudnn_rnn_flatten_weight(
              self._flat_weights, (4 if self.bias else 2),
              self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
              self.batch_first, bool(self.bidirectional))


  // yf225 TODO: old code in this function:

  // Cache the flattened weight and bias vector.
  flat_weights_ = flat_weights();

  if (!cudnn_mode_ || !torch::cudnn_is_acceptable(w_ih.at(0))) {
    return;
  }

  NoGradGuard no_grad;
  torch::_cudnn_rnn_flatten_weight(
      flat_weights_,
      /*weight_stride0=*/options.with_bias() ? 4 : 2,
      options.input_size(),
      static_cast<int64_t>(*cudnn_mode_),
      options.hidden_size(),
      options.layers(),
      /*batch_first=*/options.batch_first(),
      /*bidirectional=*/options.bidirectional());
}

template <typename Derived>
void RNNImplBase<Derived>::reset_parameters() {
  const double stdv = 1.0 / std::sqrt(options.hidden_size());
  for (auto& weight : this->parameters()) {
    init::uniform_(weight, -stdv, stdv);
  }
}

template <typename Derived>
void RNNImplBase<Derived>::to(
    torch::Device device,
    torch::Dtype dtype,
    bool non_blocking) {
  nn::Module::to(device, dtype, non_blocking);
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Dtype dtype, bool non_blocking) {
  nn::Module::to(dtype, non_blocking);
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Device device, bool non_blocking) {
  nn::Module::to(device, non_blocking);
  const auto num_directions = options.bidirectional() ? 2 : 1;
  for (int64_t layer = 0; layer < options.layers(); layer++) {
    for (auto direction = 0; direction < num_directions; direction++) {
      const auto layer_idx = (layer * num_directions) + direction;
      w_ih[layer_idx] = w_ih[layer_idx].to(device, non_blocking);
      w_hh[layer_idx] = w_hh[layer_idx].to(device, non_blocking);
      if (options.with_bias()) {
        b_ih[layer_idx] = b_ih[layer_idx].to(device, non_blocking);
        b_hh[layer_idx] = b_hh[layer_idx].to(device, non_blocking);
      }
    }
  }
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::pretty_print(std::ostream& stream) const {
  const std::string name = this->name();
  const std::string name_without_impl = name.substr(0, name.size() - 4);
  stream << name_without_impl << "(input_size=" << options.input_size()
         << ", hidden_size=" << options.hidden_size()
         << ", layers=" << options.layers() << ", dropout=" << options.dropout()
         << ")";
}

template <typename Derived>
RNNOutput RNNImplBase<Derived>::generic_forward(
    std::function<RNNFunctionSignature> function,
    const Tensor& input,
    Tensor state) {
  if (!state.defined()) {
    // #layers, batch size, state size
    const auto batch_size = input.size(options.batch_first() ? 0 : 1);
    const auto num_directions = options.bidirectional() ? 2 : 1;
    state = torch::zeros(
      {options.layers() * num_directions, batch_size, options.hidden_size()},
      input.options());
  }
  Tensor output, new_state;
  std::tie(output, new_state) = function(
      input,
      std::move(state),
      flat_weights_,
      options.with_bias(),
      options.layers(),
      options.dropout(),
      this->is_training(),
      options.bidirectional(),
      options.batch_first());
  return {output, new_state};
}

template <typename Derived>
std::vector<Tensor> RNNImplBase<Derived>::flat_weights() const {
  // Organize all weights in a flat vector in the order
  // (w_ih, w_hh, b_ih, b_hh), repeated for each layer (next to each other).
  std::vector<Tensor> flat;
  const auto num_directions = options.bidirectional() ? 2 : 1;
  for (int64_t layer = 0; layer < options.layers(); layer++) {
    for (auto direction = 0; direction < num_directions; direction++) {
      const auto layer_idx = (layer * num_directions) + direction;
      flat.push_back(w_ih[layer_idx]);
      flat.push_back(w_hh[layer_idx]);
      if (options.with_bias()) {
        flat.push_back(b_ih[layer_idx]);
        flat.push_back(b_hh[layer_idx]);
      }
    }
  }
  return flat;
}

template <typename Derived>
bool RNNImplBase<Derived>::any_parameters_alias() const {
  // If any parameters alias, we fall back to the slower, copying code path.
  // This is a sufficient check, because overlapping parameter buffers that
  // don't completely alias would break the assumptions of the uniqueness check
  // in Module.named_parameters().
  std::unordered_set<void*> unique_data_ptrs;
  auto params = this->parameters();
  unique_data_ptrs.reserve(params.size());
  for (const auto& p : params) {
    unique_data_ptrs.emplace(p.data_ptr());
  }
  return unique_data_ptrs.size() != params.size();
}

template class RNNImplBase<LSTMImpl>;
template class RNNImplBase<GRUImpl>;
template class RNNImplBase<RNNImpl>;
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RNNImpl::RNNImpl(const RNNOptions& options_)
    : detail::RNNImplBase<RNNImpl>(
          detail::RNNOptionsBase(options_.input_size(), options_.hidden_size())
              .layers(options_.layers())
              .with_bias(options_.with_bias())
              .dropout(options_.dropout())
              .bidirectional(options_.bidirectional())
              .batch_first(options_.batch_first()),
          static_cast<CuDNNMode>(options_.activation())),
      options(options_) {}

void RNNImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::RNN(input_size=" << options.input_size()
         << ", hidden_size=" << options.hidden_size()
         << ", layers=" << options.layers() << ", dropout=" << options.dropout()
         << ", activation="
         << (options.activation() == RNNActivation::Tanh ? "tanh" : "relu")
         << ")";
}

RNNOutput RNNImpl::forward(const Tensor& input, Tensor state) {
  switch (options.activation()) {
    case RNNActivation::ReLU:
      return generic_forward(
          static_cast<RNNFunctionSignature*>(&torch::rnn_relu),
          input,
          std::move(state));
    case RNNActivation::Tanh:
      return generic_forward(
          static_cast<RNNFunctionSignature*>(&torch::rnn_tanh),
          input,
          std::move(state));
    default:
      AT_ERROR("Unhandled RNN activation function!");
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSTMImpl::LSTMImpl(const LSTMOptions& options_)
    : detail::RNNImplBase<LSTMImpl>(
          options_,
          CuDNNMode::LSTM,
          /*number_of_gates=*/4) {}

RNNOutput LSTMImpl::forward(const Tensor& input, Tensor state) {
  // It would be trickier to adapt the `generic_forward` for the LSTM because
  // its output has a different dimensionality (3-tuple vs. 2-tuple), while we
  // always return one state variable (stacking the hidden/cell state into one),
  // which also makes the state variables going into the `generic_forward`, and
  // the way we default-initialize the state when it is not passed, slightly
  // different. So we just re-implement it specifically for the LSTM here.
  if (!state.defined()) {
    // 2 for hidden state and cell state, then #layers, batch size, state size
    const auto batch_size = input.size(options.batch_first() ? 0 : 1);
    const auto num_directions = options.bidirectional() ? 2 : 1;
    state = torch::zeros(
        {2, options.layers() * num_directions, batch_size, options.hidden_size()},
        input.options());
  }
  Tensor output, hidden_state, cell_state;
  std::tie(output, hidden_state, cell_state) = torch::lstm(
      input,
      {state[0], state[1]},
      flat_weights_,
      options.with_bias(),
      options.layers(),
      options.dropout(),
      this->is_training(),
      options.bidirectional(),
      options.batch_first());
  return {output, torch::stack({hidden_state, cell_state})};
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRUImpl::GRUImpl(const GRUOptions& options_)
    : detail::RNNImplBase<GRUImpl>(
          options_,
          CuDNNMode::GRU,
          /*number_of_gates=*/3) {}

RNNOutput GRUImpl::forward(const Tensor& input, Tensor state) {
  return generic_forward(
      static_cast<RNNFunctionSignature*>(&torch::gru), input, std::move(state));
}
} // namespace nn
} // namespace torch
