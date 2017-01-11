#ifndef CAFFE_GRADIENT_SCALER_LAYER_HPP_
#define CAFFE_GRADIENT_SCALER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class GradientScalerLayer : public NeuronLayer<Dtype> {
 public:
  explicit GradientScalerLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param), diff_() {}
  //virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GradientScaler"; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
};

}  // namespace caffe

#endif
