#include <vector>

#include "caffe/layers/scale_inv_rmse_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <cmath>
#include <algorithm>

namespace caffe {

template <typename Dtype>
void ScaleInvRMSELossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::Reshape(bottom, top);
        CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
                << "Inputs must have the same dimension.";
        diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ScaleInvRMSELossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
                                            
        //From "Depth map prediction from a Single Image using a multi-scale deep network" Eigen et. al NIPS 2014, equation (4)
        
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
        Dtype dot_reg = caffe_cpu_vec_sum(count, diff_.cpu_data());
        
        //Dtype loss = std::sqrt( (dot * std::sqrt(dot)) / (bottom[0]->num()));
        Dtype loss =  dot  / count - (Dtype(0.5) / std::pow(count,Dtype(2))) * std::pow(dot_reg,Dtype(2)) ;
        //TESTED. OK
        top[0]->mutable_cpu_data()[0] = loss;

        delete(log_bottom_data);
        delete(log_bottom_label);

}

template <typename Dtype>
void ScaleInvRMSELossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        for (int i = 0; i < 2; ++i) {
                if (propagate_down[i]) {
                
                        //DA SISTEMARE. VEDERE VERSIONE GPU CORRETTA
                        const Dtype sign = (i == 0) ? 1 : -1;
                        Dtype count = bottom[i]->count();
                        
                        Blob<Dtype>* division = new Blob<Dtype>(bottom[i]->shape());
                        Blob<Dtype>* first_term = new Blob<Dtype>(bottom[i]->shape());
                        Blob<Dtype>* second_term = new Blob<Dtype>(bottom[i]->shape());
                        Blob<Dtype>* bottom_copy = new Blob<Dtype>(bottom[i]->shape());
                        
                        caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), bottom_copy->mutable_cpu_data());
                        Dtype shift = Dtype(0.001);

                        caffe_add_scalar(count, shift, bottom_copy->mutable_cpu_data());
                                                                  
                        caffe_div (count, diff_.cpu_data(), bottom_copy->cpu_data(), division->mutable_cpu_data());
                        
                        const Dtype alpha_f = sign * top[0]->cpu_diff()[0] *Dtype(2)/  bottom[i]->shape(0);
                        Dtype lambda = Dtype(0.5);
                        Dtype norm = ( sign * lambda * top[0]->cpu_diff()[0]) * Dtype(2)/std::pow(bottom[i]->shape(0),Dtype(2));                       
              
                        caffe_cpu_axpby(
                                count,           // count
                                alpha_f,                           // alpha
                                division->cpu_data(),                // a
                                Dtype(0),                        // beta
                                first_term->mutable_cpu_data()
                                //bottom[i]->mutable_cpu_diff()
                                );
                        
                        
                        
                        caffe_cpu_axpby(
                                count,           // count
                                norm,                           // alpha
                                division->cpu_data(),                // a
                                Dtype(0),                        // beta
                                second_term->mutable_cpu_data()
                                );
                        
                        caffe_sub(count, first_term->cpu_data(), second_term->cpu_data(), bottom[i]->mutable_cpu_diff());
                        
                        
                        delete(first_term);
                        delete(second_term);
                        delete(division);
                        delete(bottom_copy);
                     }
        }
}
#ifdef CPU_ONLY
STUB_GPU(ScaleInvRMSELossLayer);
#endif

INSTANTIATE_CLASS(ScaleInvRMSELossLayer);
REGISTER_LAYER_CLASS(ScaleInvRMSELoss);

}  // namespace caffe
