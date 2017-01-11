#include <vector>

#include "caffe/layers/gradient_scaler_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <cmath>

namespace caffe {

  template <typename Dtype>
  void GradientScalerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    top[0]->ShareData(*bottom[0]);
  }

  template <typename Dtype>
  void GradientScalerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
      const int count = top[0]->count();
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

      caffe_cpu_scale(count, Dtype(-1), top_diff, bottom_diff);
    }
  }


#ifdef CPU_ONLY
STUB_GPU(GradientScalerLayer);
#endif

INSTANTIATE_CLASS(GradientScalerLayer);
REGISTER_LAYER_CLASS(GradientScaler);

}  // namespace caffe
