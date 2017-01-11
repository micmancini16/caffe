#include <vector>

#include "caffe/layers/log_rmse_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <cmath>

namespace caffe {

template <typename Dtype>
void LogRMSELossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::Reshape(bottom, top);
        CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
                << "Inputs must have the same dimension.";
        diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void LogRMSELossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        int count = bottom[0]->count();

        Blob<Dtype>* log_bottom_data = new Blob<Dtype>(bottom[0]->shape());
        Blob<Dtype>* log_bottom_label = new Blob<Dtype>(bottom[1]->shape());

        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* bottom_label = bottom[1]->cpu_data();

        Dtype* log_data = log_bottom_data->mutable_cpu_data();
        Dtype* log_label = log_bottom_label->mutable_cpu_data();

        caffe_copy(count, bottom_data, log_data);
        caffe_copy(count, bottom_label, log_label);

        Dtype shift = Dtype(0.0001);

        caffe_add_scalar(count, shift, log_data);
        caffe_add_scalar(count, shift, log_label);

        caffe_log(count,log_data,log_data);
        caffe_log(count,log_label,log_label);

        caffe_sub(
                count,
                log_data,
                log_label,
                diff_.mutable_cpu_data());

        Dtype dot=caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());

        //Dtype loss = std::sqrt( (dot * std::sqrt(dot)) / (bottom[0]->num()));
        Dtype loss = std::sqrt( dot  / (bottom[0]->num()));

        top[0]->mutable_cpu_data()[0] = loss;

        delete(log_bottom_data);
        delete(log_bottom_label);

}

template <typename Dtype>
void LogRMSELossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        for (int i = 0; i < 2; ++i) {
                if (propagate_down[i]) {
                        const Dtype sign = (i == 0) ? 1 : -1;

                        Blob<Dtype>* gradient_no_const = new Blob<Dtype>(bottom[i]->shape());
                        Blob<Dtype>* bottom_copy = new Blob<Dtype>(bottom[i]->shape());
                        //Blob<Dtype>* pow_diff = new Blob<Dtype>(bottom[i]->shape());

                        Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num() / top[0]->cpu_data()[0] ;

                        caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), bottom_copy->mutable_cpu_data());

                        Dtype shift = Dtype(0.0001);

                        caffe_add_scalar(bottom[i]->count(), shift, bottom_copy->mutable_cpu_data());

                        //caffe_mul(bottom[i]->count(),diff_.cpu_data(),diff_.cpu_data(),pow_diff->mutable_cpu_data());

                        caffe_div(bottom[i]->count(),diff_.cpu_data(),bottom_copy->cpu_data(),gradient_no_const->mutable_cpu_data());

                        caffe_cpu_axpby(
                                bottom[i]->count(),           // count
                                alpha,                           // alpha
                                gradient_no_const->cpu_data(),                // a
                                Dtype(0),                        // beta
                                bottom[i]->mutable_cpu_diff()
                                );

                        delete(gradient_no_const);
                        delete(bottom_copy);
                        //delete(pow_diff);
                }
        }
}
#ifdef CPU_ONLY
STUB_GPU(LogRMSELossLayer);
#endif

INSTANTIATE_CLASS(LogRMSELossLayer);
REGISTER_LAYER_CLASS(LogRMSELoss);

}  // namespace caffe
