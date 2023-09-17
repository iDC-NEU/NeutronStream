#ifndef NEUTRONSTREAM_GRAPHOP_H
#define NEUTRONSTREAM_GRAPHOP_H

#include <unistd.h>
#include <torch/torch.h>
#include <string>
#include <dygstore/embedding.hpp>
#include <dygstore/dygraph.hpp>
#include <dygstore/interface.hpp>
#include <process/dysubg.h>
namespace neutron
{
  torch::Tensor get_mask_in_neighbors(size_t uid,int64_t num_nodes);
  torch::Tensor get_mask_out_neighbors(size_t uid,int64_t num_nodes);
  torch::Tensor get_mask_neighbors(size_t uid,int64_t num_nodes,const std::string &mode);
  
  torch::Tensor get_mask_in_neighbors(DySubGraph &dysubg,size_t uid,int64_t num_nodes);
  torch::Tensor get_mask_out_neighbors(DySubGraph &dysubg,size_t uid,int64_t num_nodes);
  torch::Tensor get_mask_neighbors(DySubGraph &dysubg,size_t uid,int64_t num_nodes,const std::string &mode);

  torch::Tensor gather_in(size_t uid,int dim=0);
  torch::Tensor gather_out(size_t uid,int dim=0);
  torch::Tensor gather(size_t uid,int dim=0,const std::string &mod="in");
  
  void scatter_in(size_t uid,torch::Tensor &update_emb,int dim=0);
  void scatter_out(size_t uid,torch::Tensor &update_emb,int dim=0);
  void scatter(size_t uid,torch::Tensor &update_emb,int dim=0,const std::string &mod="in");
  

  torch::Tensor gather_in( DySubGraph &dysubg,size_t uid,int dim=0);
  torch::Tensor gather_out(DySubGraph &dysubg,size_t uid,int dim=0);
  torch::Tensor gather(DySubGraph &dysubg,size_t uid,int dim=0,const std::string &mod="in");
  
  void scatter_in(DySubGraph &dysubg,size_t uid,torch::Tensor &update_emb,int dim=0);
  void scatter_out( DySubGraph &dysubg,size_t uid,torch::Tensor &update_emb,int dim=0);
  void scatter(DySubGraph &dysubg,size_t uid,torch::Tensor &update_emb,int dim=0,const std::string &mod="out");
  

  torch::Tensor gather_out_timestamps( DySubGraph &dysubg,size_t uid,const std::string &format="hour");
  torch::Tensor gather_in_timestamps( DySubGraph &dysubg,size_t uid,const std::string &format="hour");
  torch::Tensor gather_timestamps( DySubGraph &dysubg,size_t uid,const std::string &mod="in");
  
  torch::Tensor gather_in_timestamps(size_t uid);
  torch::Tensor gather_out_timestamps(size_t uid);
  torch::Tensor gather_timestamps(size_t uid,const std::string &mod="in");
  
  
} // namespace neutron

#endif