#include <vector>

#include "caffe/layers/log_rmse_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <cmath>

namespace caffe {


template <typename Dtype>
void LogRMSELossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {

        int count = bottom[0]->count();

        Blob<Dtype>* log_bottom_data = new Blob<Dtype>(bottom[0]->shape());
        Blob<Dtype>* log_bottom_label = new Blob<Dtype>(bottom[1]->shape());

        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* bottom_label = bottom[1]->gpu_data();

        Dtype* log_data = log_bottom_data->mutable_gpu_data();
        Dtype* log_label = log_bottom_label->mutable_gpu_data();

        caffe_copy(count, bottom_data, log_data);
        caffe_copy(count, bottom_label, log_label);

        Dtype shift = Dtype(0.0001);

        caffe_gpu_add_scalar(count, shift, log_data);
        caffe_gpu_add_scalar(count, shift, log_label);

        caffe_gpu_log(count,log_data,log_data);
        caffe_gpu_log(count,log_label,log_label);

        caffe_gpu_sub(
                count,
                log_data,
                log_label,
                diff_.mutable_gpu_data());

        Dtype dot;
        caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(),&dot);

        //dot = dot * std::sqrt(dot);

        Dtype loss = std::sqrt( dot / (bottom[0]->num()));

        top[0]->mutable_cpu_data()[0] = loss;

        delete(log_bottom_data);
        delete(log_bottom_label);

}

template <typename Dtype>
void LogRMSELossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        for (int i = 0; i < 2; ++i) {
                if (propagate_down[i]) {
                      const Dtype sign = (i == 0) ? 1 : -1;

                      Blob<Dtype>* gradient_no_const = new Blob<Dtype>(bottom[i]->shape());
                      Blob<Dtype>* bottom_copy = new Blob<Dtype>(bottom[i]->shape());
                      //Blob<Dtype>* pow_diff = new Blob<Dtype>(bottom[i]->shape());

                      Dtype alpha =  sign * top[0]->cpu_diff()[0] / bottom[i]->num() / top[0]->cpu_data()[0] ;

                      caffe_copy(bottom[i]->count(), bottom[i]->gpu_data(), bottom_copy->mutable_gpu_data());

                      Dtype shift = Dtype(0.0001);

                      caffe_gpu_add_scalar(bottom[i]->count(), shift, bottom_copy->mutable_gpu_data());

                      //caffe_gpu_mul(bottom[i]->count(),diff_.gpu_data(),diff_.gpu_data(),pow_diff->mutable_gpu_data());

                      caffe_gpu_div(bottom[i]->count(),diff_.gpu_data(),bottom_copy->gpu_data(),gradient_no_const->mutable_gpu_data());

                      caffe_gpu_axpby(
                         bottom[i]->count(),              // count
                         alpha,                              // alpha
                         gradient_no_const->gpu_data(),                   // a
                         Dtype(0),                           // beta
                         bottom[i]->mutable_gpu_diff()
                       );

                      delete(gradient_no_const);
                      delete(bottom_copy);
                      //delete(pow_diff);
                }
        }

}

INSTANTIATE_LAYER_GPU_FUNCS(LogRMSELossLayer);

}  // namespace caffe
