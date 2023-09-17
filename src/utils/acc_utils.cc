#include <utils/acc_utils.h>
#include <log/log.h>
#include <iostream>
#include <vector>
namespace neutron{
std::ostream& operator<< (std::ostream& out, std::vector<int>& _vec){
  std::vector<int>::const_iterator it = _vec.begin();
  out <<"[ ";
  for(; it != _vec.end(); ++it)
          out << *it<<" ";
  out<<"]";
  return out;
}
std::ostream& operator<< (std::ostream& out, std::vector<float>& _vec){
  std::vector<float>::const_iterator it = _vec.begin();
  out <<"[ ";
  for(; it != _vec.end(); ++it)
          out << *it<<" ";
  out<<"]";
  return out;
}
namespace acc{
    float roc_auc_scores(const torch::Tensor &true_labels,const torch::Tensor &pred_labels){
        float auc=0.0;
        int pos_size=0;
        int neg_size=0;
        CHECK(true_labels.sizes()==pred_labels.sizes())<<"make sure the true labels sizes"<<true_labels.sizes()<<" is equal to pre_lables sizes"<<pred_labels.sizes();
        {
            torch::NoGradGuard no_grad;
            torch::Tensor neg_idx=(true_labels==0).argwhere().to(torch::kInt32);
            torch::Tensor pos_idx=true_labels.argwhere().to(torch::kInt32);
            std::vector<int> neg_idx_vec(neg_idx.data_ptr<int>(),neg_idx.data_ptr<int>()+neg_idx.numel());
            // std::cout<<"neg_idx_vec:"<<neg_idx_vec<<std::endl;
            neg_size=neg_idx_vec.size();
            std::vector<int> pos_idx_vec(pos_idx.data_ptr<int>(),pos_idx.data_ptr<int>()+pos_idx.numel());
            // std::cout<<"pos_idx_vec:"<<pos_idx_vec<<std::endl;
            torch::Tensor pred_labels_=pred_labels.to(torch::kFloat32);
            std::vector<float> pred_labels_vec(pred_labels_.data_ptr<float>(),pred_labels_.data_ptr<float>()+pred_labels_.numel());
            // std::cout<<"pred_lables_vec:"<<pred_labels_vec<<std::endl;
            pos_size=pos_idx_vec.size();
            for(auto pos_idx:pos_idx_vec){
                for(auto neg_idx:neg_idx_vec){
                    if(pred_labels_vec[pos_idx]>pred_labels_vec[neg_idx]){
                        auc+=1;
                    }else if(pred_labels_vec[pos_idx]==pred_labels_vec[neg_idx]){
                        auc+=0.5;
                    }
                }
            }
        }
        return auc/(pos_size*neg_size);
    }
    //average precision score    
    float pr_auc_scores(const torch::Tensor &true_labels,const torch::Tensor &pred_labels){
        float  acc_rst=0;
        {
            torch::NoGradGuard no_grad;

        }
        return acc_rst;

    }
    float average_precision_scores(torch::Tensor &true_labels,const torch::Tensor &pred_labels){
        float acc_rst=0;
        {
            torch::NoGradGuard no_grad;

        }
        return acc_rst;
    }
} //namespace acc
}//namspace neutron