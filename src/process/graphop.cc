#include <process/graphop.h>
#include <dygstore/interface.hpp>
#include <utils/util.h>
namespace neutron{
  static DyNodeEmbedding<torch::Tensor,2>::ptr
  dynode_embedding=DyNodeEmbedding<torch::Tensor,2>::GetInstance();


  torch::Tensor get_mask_in_neighbors(size_t uid,int64_t num_nodes){
    torch::Tensor mask=torch::zeros({num_nodes}).to(torch::kBool);
    std::vector<size_t> u_nies=InNeighbors(uid);
    #pragma omp parallel for
    for(size_t i=0;i<u_nies.size();i++){
      mask[static_cast<int64_t>(u_nies[i])]=true;
    }
    return mask;
  }

  torch::Tensor get_mask_out_neighbors(size_t uid,int64_t num_nodes){
    torch::Tensor mask=torch::zeros({num_nodes}).to(torch::kBool);
    std::vector<size_t> u_nies=OutNeighbors(uid);
    #pragma omp parallel for
    for(size_t i=0;i<u_nies.size();i++){
      mask[static_cast<int64_t>(u_nies[i])]=true;
    }
    return mask;
  }

  torch::Tensor get_mask_neighbors(size_t uid,int64_t num_nodes,const std::string &mode){
    if(mode=="in"){
      return get_mask_in_neighbors(uid,num_nodes);
    }
    else if(mode=="out"){
      return get_mask_out_neighbors(uid,num_nodes);
    }
    CHECK(false)<< "Invalid input mod"<<mode;
    return torch::tensor(std::vector<int>{});
  }


  torch::Tensor get_mask_in_neighbors(DySubGraph &dysubg,size_t uid,int64_t num_nodes){

    torch::Tensor mask=torch::zeros({num_nodes}).to(torch::kBool);
    std::vector<size_t> u_nies=dysubg.in_neighbors(uid);
    #pragma omp parallel for
    for(size_t i=0;i<u_nies.size();i++){
      mask[static_cast<int64_t>(u_nies[i])]=true;
    }
    return mask.to(Session::GetInstance()->GetDevice());
  }

  torch::Tensor get_mask_out_neighbors(DySubGraph &dysubg,size_t uid,int64_t num_nodes){
    torch::Tensor mask=torch::zeros({num_nodes}).to(torch::kBool);
    std::vector<size_t> u_nies=dysubg.out_neighbors(uid);
    #pragma omp parallel for
    for(size_t i=0;i<u_nies.size();i++){
      mask[static_cast<int64_t>(u_nies[i])]=true;
    }
    return mask;
  }

  torch::Tensor get_mask_neighbors(DySubGraph &dysubg,size_t uid,int64_t num_nodes,const std::string &mode){
    if(mode=="in"){
      return get_mask_in_neighbors(dysubg,uid,num_nodes);
    }
    else if(mode=="out"){
      return get_mask_out_neighbors(dysubg,uid,num_nodes);
    }
    CHECK(false)<< "Invalid input mod"<<mode;
    return torch::tensor(std::vector<int>{}); 
  }

  torch::Tensor gather_in(size_t uid,int dim){
    std::vector<size_t> in_neighbors=InNeighbors(uid);
    torch::Tensor rst = index(in_neighbors,dim);
    return rst;
  }
  
  torch::Tensor gather_out(size_t uid,int dim){
    std::vector<size_t> out_neighbors=OutNeighbors(uid);
    torch::Tensor rst= index(out_neighbors,dim);
    return rst;
  }
  
  torch::Tensor gather(size_t uid,int dim,const std::string &mod){
    if(mod=="in"){
      torch::Tensor rst= gather_in(uid,dim);
      return rst;
    }else if(mod=="out"){
      torch::Tensor rst= gather_out(uid,dim);
      return rst;
    }
    CHECK(false)<< "Invalid input mod"<<mod;
    return torch::tensor(std::vector<int>{});
  }
  
  torch::Tensor gather_in( DySubGraph &dysubg,size_t uid,int dim){

    std::vector<size_t> in_neighbors=dysubg.in_neighbors(uid);
    torch::Tensor rst= index(in_neighbors,dim);
    return rst;
  }
  
  torch::Tensor gather_out( DySubGraph &dysubg,size_t uid,int dim){
    std::vector<size_t> out_neighbors=dysubg.out_neighbors(uid);
    torch::Tensor rst= index(out_neighbors,dim);

    return rst;
  }
  
  torch::Tensor gather( DySubGraph &dysubg,size_t uid,int dim,const std::string &mod){
    if(mod=="in"){
      torch::Tensor rst= gather_in(dysubg,uid,dim);
      return rst;
    }else if(mod=="out"){
      torch::Tensor rst= gather_out(dysubg,uid,dim);
      return rst;
    }
    CHECK(false)<< "Invalid input mod"<<mod;
    return torch::tensor(std::vector<int>{});

  }
  
  torch::Tensor timePointVec2Tensor(std::vector<TimePoint> &times,const std::string &format){
    std::vector<float> rst(times.size());
    if(format=="hour"){
      #pragma omp parallel for
      for(size_t i=0;i<times.size();i++){
        rst[i]=times[i].toHour();
      }
    }else{
      CHECK(false)<<" not implemented format"<<format;
    }
    torch::Tensor rst_vec= torch::tensor(rst);
    return rst_vec;
  }
  
  torch::Tensor gather_out_timestamps(DySubGraph &dysubg,size_t uid,const std::string &format){
    std::vector<size_t> out_neighbors=dysubg.out_neighbors(uid);
    std::vector<TimePoint> times=dysubg.get_pre_times(out_neighbors);
    torch::Tensor rst= timePointVec2Tensor(times,format);
    return rst;
  }
  
  torch::Tensor gather_in_timestamps(DySubGraph &dysubg,size_t uid,const std::string &format){
    std::vector<size_t> in_neighbors=dysubg.in_neighbors(uid);
    std::vector<TimePoint> times=dysubg.get_pre_times(in_neighbors);
    torch::Tensor rst= timePointVec2Tensor(times,format);
    return rst;
    
  }

  torch::Tensor gather_in_timestamps(size_t uid,const std::string &format){
    std::vector<size_t> in_neighbors=InNeighbors(uid);
    std::vector<TimePoint> times=GetTimePoints(in_neighbors);
    torch::Tensor rst= timePointVec2Tensor(times,format);
    return rst;
  }
  
  torch::Tensor gather_out_timestamps(size_t uid,const std::string &format){
    std::vector<size_t> out_neighbors=OutNeighbors(uid);
    std::vector<TimePoint> times=GetTimePoints(out_neighbors);
    torch::Tensor rst= timePointVec2Tensor(times,format);
    return rst;
  }
  
  void scatter_in(size_t uid,torch::Tensor &update_emb,int dim){

    std::vector<size_t> in_neighbors=InNeighbors(uid);
    std::vector<torch::Tensor> embs=tensor2vec(update_emb);
    update(in_neighbors,embs,dim);
  }
  
  void scatter_out(size_t uid,torch::Tensor &update_emb,int dim){
    std::vector<size_t> out_neighbors=OutNeighbors(uid);
    std::vector<torch::Tensor> embs=tensor2vec(update_emb);
    update(out_neighbors,embs,dim);
  }
  
  void scatter(size_t uid,torch::Tensor &update_emb,int dim,const std::string &mod){
    if(mod=="in"){
      scatter_in(uid,update_emb,dim);
    }else if(mod=="out"){
      scatter_out(uid,update_emb,dim);
    }
  }

  void scatter_in(DySubGraph &dysubg,size_t uid,torch::Tensor &update_emb,int dim){
    std::vector<size_t> in_neighbors=dysubg.in_neighbors(uid);
    std::vector<torch::Tensor> embs=tensor2vec(update_emb);
    update(in_neighbors,embs,dim);
  }
  
  void scatter_out( DySubGraph &dysubg,size_t uid,torch::Tensor &update_emb,int dim){
    std::vector<size_t> out_neighbors=dysubg.out_neighbors(uid);
    std::vector<torch::Tensor> embs=tensor2vec(update_emb);
    update(out_neighbors,embs,dim);
  }
  
  void scatter(DySubGraph &dysubg,size_t uid,torch::Tensor &update_emb,int dim,const std::string &mod){
    if(mod=="in"){
      scatter_in(dysubg,uid,update_emb,dim);
    }else if(mod=="out"){
      scatter_out(dysubg,uid,update_emb,dim);
    }
  }
} // namespace neutron
