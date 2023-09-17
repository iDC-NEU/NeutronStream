#ifndef NEUTRONSTRAM_DYGSTORE_INTERFACE_HPP
#define NEUTRONSTRAM_DYGSTORE_INTERFACE_HPP
#include <dygstore/dygraph.hpp>
#include <dygstore/timebar.h>
#include <dygstore/embedding.hpp>
#include <dygstore/interface.hpp>
#include <future>
#include <session.h>
namespace neutron
{
  template<typename EdgeDataType=EdgeData,typename EMBType=torch::Tensor,size_t SAVET=2>
  void clear(){
    auto dyg=DiDyGraph<EdgeDataType>::GetInstance();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto initial_dyg=InitialDiDyGraph<EdgeDataType>::GetInstance();
    auto initial_dyemb=InitialEmb<torch::Tensor>::GetInstance();
    dyg->clear();
    embptr->clear();
    initial_dyg->clear();
    initial_dyemb->clear();
  }
  template<typename EdgeDataType=EdgeData>
  std::vector<t_edge<EdgeDataType>> QueryInEdge(size_t src_id,size_t dst_id){
    auto dygptr=DiDyGraph<EdgeDataType>::GetInstanceThreadSafe();
    std::vector<t_edge<EdgeDataType>> rstEdge;
    auto src_rst_future=Session::GetInstance()->getThreadPool()->submit([&]{
      return dygptr->InEdges(src_id);
    });
    auto dst_rst_future=Session::GetInstance()->getThreadPool()->submit([&]{
      return dygptr->InEdges(dst_id);
    });
    auto src_rst=src_rst_future.get();
    rstEdge.insert(rstEdge.end(),src_rst.begin(),src_rst.end());
    auto dst_rst=dst_rst_future.get();
    rstEdge.insert(rstEdge.end(),dst_rst.begin(),dst_rst.end());
    return rstEdge;
  }

  template<typename EdgeDataType=EdgeData>
  std::vector<t_edge<EdgeDataType>> QueryOutEdge(size_t src_id,size_t dst_id){
     auto dygptr=DiDyGraph<EdgeDataType>::GetInstanceThreadSafe();
    std::vector<t_edge<EdgeDataType>> rstEdge;
    auto src_rst_future=Session::GetInstance()->getThreadPool()->submit([&]{
      return dygptr->OutEdges(src_id);
    });
    auto dst_rst_future=Session::GetInstance()->getThreadPool()->submit([&]{
      return dygptr->OutEdges(dst_id);
    });
    auto src_rst=src_rst_future.get();
    rstEdge.insert(rstEdge.end(),src_rst.begin(),src_rst.end());
    auto dst_rst=dst_rst_future.get();
    rstEdge.insert(rstEdge.end(),dst_rst.begin(),dst_rst.end());
    return rstEdge;
  }

  template<typename EdgeDataType=EdgeData>
  std::vector<t_edge<EdgeDataType>> QueryAllEdge(size_t src_id,size_t dst_id){
    std::vector<t_edge<EdgeDataType>> rstEdge;
     auto in_edge_future=Session::GetInstance()->getThreadPool()->submit([&]{
      return QueryInEdge<EdgeDataType>(src_id,dst_id);
    });
    auto out_edge_future=Session::GetInstance()->getThreadPool()->submit([&]{
      return QueryOutEdge<EdgeDataType>(src_id,dst_id);
    });
    auto in_edges=in_edge_future.get();
    rstEdge.insert(rstEdge.end(),in_edges.begin(),in_edges.end());
    auto out_edges=out_edge_future.get();
    rstEdge.insert(rstEdge.end(),out_edges.begin(),out_edges.end());
    return rstEdge;
  }

  template<typename EdgeDataType=EdgeData>
  std::vector<t_edge<EdgeDataType>> QueryEdge(size_t src_id,size_t dst_id){
    if(Session::GetInstance()->GetQueryMode()==InEdgeMode){
      return QueryInEdge<EdgeDataType>(src_id,dst_id);
    }else if(Session::GetInstance()->GetQueryMode()==OutEdgeMode){
      return QueryOutEdge<EdgeDataType>(src_id,dst_id);
    }else if(Session::GetInstance()->GetQueryMode()==AllEdgeMode){
      return QueryAllEdge<EdgeDataType>(src_id,dst_id);
    }else{
      throw std::logic_error(" Query Mode Error !!! ");
    }
  }

  template<typename EdgeDataType=EdgeData,typename EMBType=torch::Tensor,size_t SAVET=2>
  void BackToInitial(){
    auto dyg=DiDyGraph<EdgeDataType>::GetInstance();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto initial_dyg=InitialDiDyGraph<EdgeDataType>::GetInstance();
    dyg->BackToInitial(initial_dyg->numEdges,initial_dyg->numNodes,initial_dyg->adjList);
    TimeBar::GetInstance()->Init(initial_dyg->numNodes,initial_dyg->first_time);
    embptr->BackToInitial();
  
  }

  template<typename EdgeDataType=EdgeData>
  void InitialVertexes(size_t num_nodes,TimePoint first_time=0){
    TimeBar::GetInstance()->Init(num_nodes,first_time);
    auto dyg=DiDyGraph<EdgeDataType>::GetInstance();
    dyg->InitVertexes(num_nodes);
  }

  template<typename EdgeDataType=EdgeData>
  void InitialEdges(const std::vector<std::vector<size_t>> &edges){
    DiDyGraph<EdgeDataType>::GetInstance()->InitEdges(edges);
  }

  template<typename EdgeDataType=EdgeData>
  void Initial(size_t num_nodes,std::vector<std::vector<size_t>> &edges,TimePoint first_time=0){
    auto initial_dyg=InitialDiDyGraph<EdgeDataType>::GetInstance();
    initial_dyg->Initial(num_nodes,edges,first_time);
    InitialVertexes<EdgeDataType>(num_nodes,first_time);
    InitialEdges<EdgeDataType>(edges);

  }
  
  template<typename EdgeDataType=EdgeData>
  void AddEdge(size_t src_id,size_t dst_id){
    auto dyg=DiDyGraph<EdgeDataType>::GetInstance();
    dyg->AddEdge(src_id,dst_id);
  }

  template<typename EdgeDataType=EdgeData>
  void AddVertex(size_t n_id){
    DiDyGraph<EdgeDataType>::GetInstance()->AddVertex(n_id);
  }

  template<typename EdgeDataType=EdgeData>
  void AddVertexes(size_t num_nodes){
    DiDyGraph<EdgeDataType>::GetInstance()->AddVertexes(num_nodes);
  }

  template<typename EdgeDataType=EdgeData>
  bool HasEdge(size_t src_id,size_t dst_id){
    return DiDyGraph<EdgeDataType>::GetInstance()->HasEdge(src_id,dst_id);
  }

  template<typename EdgeDataType=EdgeData>
  size_t NumNodes(){
    return DiDyGraph<EdgeDataType>::GetInstance()->NumNodes();
  }

  template<typename EdgeDataType=EdgeData>
  size_t NumEdges(){
    return DiDyGraph<EdgeDataType>::GetInstance()->NumEdges();
  }

  template<typename EdgeDataType=EdgeData>
  std::vector<size_t> InNeighbors(size_t uid){
    double query_time=cpuSecond();
    auto dygptr=DiDyGraph<EdgeDataType>::GetInstanceThreadSafe();
    auto rst= dygptr->InNeighbors(uid);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
    return rst;
  }

  template<typename EdgeDataType=EdgeData>
  std::vector<size_t> OutNeighbors(size_t uid){
    double query_time=cpuSecond();
    auto dygptr=DiDyGraph<EdgeDataType>::GetInstanceThreadSafe();
    auto rst= dygptr->OutNeighbors(uid);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
    return rst;
  }

  template<typename EdgeDataType=EdgeData>
  size_t InDegree(size_t uid){
    auto dygptr=DiDyGraph<EdgeDataType>::GetInstanceThreadSafe();
    return dygptr->InDegree(uid);
  }

  template<typename EdgeDataType=EdgeData>
  size_t OutDegree(size_t uid){
    auto dygptr=DiDyGraph<EdgeDataType>::GetInstanceThreadSafe();
    return dygptr->OutDegree(uid);
  }

  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  EMBType index(size_t uid,int dim=0){
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    double query_time =cpuSecond();
    EMBType rst= embptr->index(uid,dim);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
    return rst;
  }

  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  EMBType index(size_t uid,int vid,int dim){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    EMBType rst=embptr->index(uid,vid,dim);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
    return rst;
  }

  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  EMBType index(std::vector<size_t> &uids,int dim=0){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto rst_vec=embptr->index(uids,dim);
    auto rst= torch::stack(rst_vec,0).reshape({static_cast<int64_t>(rst_vec.size()),-1});
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
    return rst;
  }
  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  std::vector<EMBType> samples(const std::vector<size_t> &uids,int dim=0){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto rst= embptr->index(uids,dim);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
    return rst;
  }

  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  EMBType index(const std::vector<size_t> &uids,int dim=0){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto rst_vec= embptr->index(uids,dim);
    auto rst= vec2tensor(rst_vec);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
    return rst;

  }
  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  EMBType index_all(int dim=0){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto rst= embptr->index_all(dim);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
    return rst;
  }


  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  void update(size_t uid,EMBType &embed,int dim=0){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    embptr->update(uid,embed,dim);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
  }

  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  void update(size_t uid,const EMBType &embed,int dim=0){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    embptr->update(uid,embed,dim);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
  }

  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  void update(std::vector<size_t> &uids,std::vector<EMBType> &embedding,int dim=0){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe(); 
    embptr->update(uids,embedding,dim);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
  }
  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  void update(const std::vector<size_t> &uids,const std::vector<EMBType> &embedding,int dim=0){
    double query_time = cpuSecond();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe(); 
    embptr->update(uids,embedding,dim);
    Session::GetInstance()->addTimer("query",cpuSecond()-query_time);
  }

  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  void InitEMB(EMBType& emb,int hidden_dim,int dim_id=0){
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    int size_1=emb.size(1);
    int n_nodes=emb.size(0);
    auto device=Session::GetInstance()->GetDevice();
    std::cout<<"emb device:"<<emb.device()<<std::endl;
    emb=emb.to(device);
    std::cout<<"emb device:"<<emb.device()<<std::endl;
    if(size_1==hidden_dim){
      embptr->Init(emb,dim_id);
    }
    //补零
    else if(size_1<hidden_dim){
      torch::Tensor zero_padding=torch::zeros({static_cast<int64_t>(n_nodes),static_cast<int64_t>(hidden_dim-size_1)}).to(device);
      emb=torch::cat({emb,zero_padding},1);
      // std::cout<<emb.sizes()<<std::endl;
      embptr->Init(emb,dim_id);
    }else{
      CHECK(false)<<"hidden_dim"<< hidden_dim <<" must > "<<size_1<<" "<<emb.sizes();
    }
  }

  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  void InitEMBs(EMBType& emb,int hidden_dim,int dims){
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    
    auto device=Session::GetInstance()->GetDevice();
    int size_1=emb.size(1);
    int n_nodes=emb.size(0);
    if(size_1<hidden_dim){
      torch::Tensor zero_padding=torch::zeros({static_cast<int64_t>(n_nodes),static_cast<int64_t>(hidden_dim-size_1)});
      emb=torch::cat({emb,zero_padding},1);
      // std::cout<<emb.sizes()<<std::endl;
    }
    std::vector<torch::Tensor> embs(dims);
    for(int i=0;i<dims;i++){
      embs[i]=emb.to(device);
    }
    embptr->Inits(embs);
  }
  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  void RandomInitEMB(int64_t num_nodes,int64_t hidden_dim,int dims){
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    std::vector<torch::Tensor> embs(dims);
    auto device = Session::GetInstance()->GetDevice();
    for(int i=0;i<dims;i++){
      embs[i]=torch::randn({num_nodes,hidden_dim}).to(device);
    }
    embptr->Inits(embs);
  }
  template<typename EMBType=torch::Tensor,size_t SAVET=2>
  void merge(){
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    embptr->Merge();
  }
} // namespace neutron

#endif