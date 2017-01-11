#include <vector>

#include "caffe/layers/scale_inv_rmse_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <cmath>

namespace caffe {


template <typename Dtype>
void ScaleInvRMSELossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
        Dtype dot_reg = caffe_cpu_vec_sum(count, diff_.cpu_data());

        //dot = dot * std::sqrt(dot);

        Dtype loss = dot  / count - (Dtype(0.5) / std::pow(count,Dtype(2))) * std::pow(dot_reg,Dtype(2));

        top[0]->mutable_cpu_data()[0] = loss;

        delete(log_bottom_data);
        delete(log_bottom_label);

}

template <typename Dtype>
void ScaleInvRMSELossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        for (int i = 0; i < 2; ++i) {
                if (propagate_down[i]) {
                      const Dtype sign = (i == 0) ? 1 : -1;
                      Dtype count = bottom[i]->count();

                      Blob<Dtype>* division = new Blob<Dtype>(bottom[i]->shape());
                      Blob<Dtype>* first_term = new Blob<Dtype>(bottom[i]->shape());
                      Blob<Dtype>* second_term = new Blob<Dtype>(bottom[i]->shape());
                      Blob<Dtype>* bottom_copy = new Blob<Dtype>(bottom[i]->shape());

                      //Dtype alpha =  sign * top[0]->cpu_diff()[0] / bottom[i]->num() / top[0]->cpu_data()[0] ;

                      caffe_copy(count, bottom[i]->gpu_data(), bottom_copy->mutable_gpu_data());

                      Dtype shift = Dtype(0.0001);

                      caffe_gpu_add_scalar(count, shift, bottom_copy->mutable_gpu_data());

                      //caffe_gpu_mul(bottom[i]->count(),diff_.gpu_data(),diff_.gpu_data(),pow_diff->mutable_gpu_data());

                      caffe_gpu_div(count, diff_.gpu_data(), bottom_copy->gpu_data(), division->mutable_gpu_data());

                      const Dtype alpha = (sign * top[0]->cpu_diff()[0] * 2.0) /  bottom[i]->shape(0);
                      Dtype lambda = Dtype(0.5);

                      Dtype norm_factor = caffe_cpu_vec_sum(count, bottom_copy->cpu_data());

                      Dtype beta = ( sign * lambda * top[0]->cpu_diff()[0] * 2.0) / (std::pow(bottom[i]->shape(0),2.0) * norm_factor);                     

                      caffe_gpu_axpby(
                         bottom[i]->count(),              // count
                         alpha,                              // alpha
                         division->gpu_data(),                   // a
                         Dtype(0),                           // beta
                         first_term->mutable_gpu_data()
                       );



                      caffe_gpu_axpby(
                                count,           // count
                                beta,                           // alpha
                                diff_.gpu_data(),                // a
                                Dtype(0),                        // beta
                                second_term->mutable_gpu_data()
                                );

                      caffe_gpu_sub(count, first_term->gpu_data(), second_term->gpu_data(), bottom[i]->mutable_cpu_diff());


                      delete(first_term);
                      delete(second_term);
                      delete(division);
                      delete(bottom_copy);

                      //delete(pow_diff);
                }
        }

}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleInvRMSELossLayer);

}  // namespace caffe
