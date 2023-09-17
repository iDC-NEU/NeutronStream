#ifndef NEUTRONSTREAM_ACC_UTILS_H
#define NEUTRONSTREAM_ACC_UTILS_H
#include <torch/torch.h>
namespace neutron
{
namespace acc
{
        float roc_auc_scores(const torch::Tensor &true_labels,const torch::Tensor &pred_labels);
        float pr_auc_scores(const torch::Tensor &true_labels,const torch::Tensor &pred_labels);
        float average_precision_scores(torch::Tensor &true_labels,const torch::Tensor &pred_labels);
} // namespace acc
    
    
} // namespace neutron

#endif