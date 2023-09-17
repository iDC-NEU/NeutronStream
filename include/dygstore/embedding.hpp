#ifndef NEUTRONSTREAM_EMBEDDING_HPP
#define NEUTRONSTREAM_EMBEDDING_HPP

#include <torch/torch.h>
#include <utils/locker.h>
#include <memory>
#include <vector>
#include <utils/hash_bitmap.h>
#include <atomic>
#include <session.h>

namespace neutron{
  template<typename EMBType>
  struct InitialEmb{
    typedef std::shared_ptr<InitialEmb> ptr;
    std::vector<EMBType> initial_embs;
    size_t num_nodes;
    int64_t hidden_dim;
    torch::Device device{torch::kCPU};
    static InitialEmb::ptr InitialEmbPtr;
    static InitialEmb::ptr GetInstance();

    void Initial(size_t _num_nodes,int64_t _hidden_dim,
    torch::Device& _device,std::vector<EMBType>& _initial_embs){
      num_nodes = _num_nodes;
      hidden_dim = _hidden_dim;
      device = _device;
      initial_embs.resize(_initial_embs.size());
      for(size_t i=0;i<_initial_embs.size();i++){
        initial_embs[i] = _initial_embs[i].clone().to(device);
      }
    }
    void clear(){
      initial_embs.clear();
      initial_embs.shrink_to_fit();
      num_nodes = 0;
      hidden_dim = 0;
    }
  };

  template<typename EMBType>
  typename InitialEmb<EMBType>::ptr InitialEmb<EMBType>::InitialEmbPtr=nullptr;

  template<typename EMBType> 
  typename InitialEmb<EMBType>::ptr InitialEmb<EMBType>::GetInstance(){
    static std::mutex mtx;
    if(InitialEmbPtr==nullptr){
      std::unique_lock<std::mutex> locker(mtx);
      if(InitialEmbPtr==nullptr){
        InitialEmbPtr.reset(new(std::nothrow)InitialEmb());
      }
    }
    return InitialEmbPtr;
  }

  template<typename EMBType,size_t SAVET=2>
  class DyNodeEmbedding{
  public:
    typedef std::shared_ptr<DyNodeEmbedding> ptr;
    typedef RWMutex RWMutexType;
    typedef std::vector<CountMap> CountMapType;
    typedef std::vector<std::vector<std::unordered_map<size_t,EMBType>>> UpdateEMBType;

    static DyNodeEmbedding::ptr GetInstance();
    static DyNodeEmbedding::ptr GetInstanceThreadSafe();
    void Inits(std::vector<EMBType> &embs){
      m_device=Session::GetInstance()->GetDevice();
      std::cout<<"m_device: "<<m_device<<std::endl;
      size_t dims=embs.size();
      m_embedding.resize(embs.size());
      
      std::copy(embs.begin(),embs.end(),m_embedding.begin());
      m_num_nodes=m_embedding[0].size(0);
      m_hidden_dim=m_embedding[0].size(1);
      m_update_count.resize(dims);
      m_update_emb.resize(dims);
      for(size_t i=0;i<dims;i++){
        m_update_count[i].resize(m_num_nodes);
        m_embedding[i].set_requires_grad(false);
        m_embedding[i]=m_embedding[i].to(m_device);
      }

      {
        torch::NoGradGuard no_grad;
        InitialEmb<EMBType>::GetInstance()->Initial(
          m_num_nodes,m_hidden_dim,m_device,m_embedding
        );
      }
      
    }
    void Init(EMBType& emb,int dim_id=0){
      //进行扩容
      m_device=Session::GetInstance()->GetDevice();
      std::cout<<"m_device: "<<m_device<<std::endl;
      if(static_cast<size_t>(dim_id)<=m_embedding.size()){
        m_embedding.resize(dim_id+1);
        m_update_count.resize(dim_id+1);
        m_update_emb.resize(dim_id+1);
      }
      m_embedding[dim_id] = emb.to(m_device);
      std::cout<<"m_embedding ["<<dim_id<<"] device:"<<m_embedding[dim_id].device()<<std::endl;
      m_embedding[dim_id].set_requires_grad(false);
      m_num_nodes=emb.size(0);
      m_hidden_dim=emb.size(1);
      m_update_count[dim_id].resize(m_num_nodes);
      // std::cout<<m_num_nodes <<" "<<m_hidden_dim<<std::endl;
      {
        torch::NoGradGuard no_grad;
        InitialEmb<EMBType>::GetInstance()->Initial(
          m_num_nodes,m_hidden_dim,m_device,m_embedding
        );
      }
    }
    EMBType index(size_t uid,int dim=0){
      CHECK(uid<m_num_nodes)<< uid<<" out of range";
      RWMutexType::ReadLock lock(m_rw_mtx);
      size_t update_times=get_version_id(dim,uid);
      // update_times=0;
      if(update_times!=0){
          // return AllEmbedding[update_times-1][uid].clone();
          return index_update(update_times-1,uid,dim);
      }else {
        return index_emb(uid,dim);
      }
    }
    EMBType index(size_t uid,int vid,int dim){
      CHECK(uid<m_num_nodes)<< uid<<" out of range";
      RWMutexType::ReadLock lock(m_rw_mtx);
      size_t update_times=get_version_id(dim,uid);
      // update_times=0;
      CHECK(static_cast<size_t>(vid)<update_times)
      <<"vid error ! "<< vid<< " current version is "<< update_times;
      if(update_times!=0){
          // return AllEmbedding[update_times-1][uid].clone();
          return index_update(update_times-1,uid,dim);
      }else{
        return index_emb(uid,dim);
      }
    }

    std::vector<EMBType> index(std::vector<size_t> &uids,int dim=0){
      std::vector<EMBType> rst(uids.size());
      #pragma omp parallel for
      for(size_t i=0;i<uids.size();i++){
        rst[i]=index(uids[i],dim);
      }
      return rst;
    }
    EMBType index_all(int dim=0){
      if(is_merge.load()==false){
        return m_embedding[dim]; 
      }
      std::vector<EMBType> rst(this->m_num_nodes);
      #pragma omp parallel for
      for(size_t i=0;i<this->m_num_nodes;i++){
        rst[i]=index(i,dim);
      }
      return torch::stack(rst,0);
    }
    std::vector<EMBType> index(const std::vector<size_t> &uids,int dim=0){
      std::vector<EMBType> rst(uids.size());
      #pragma omp parallel for
      for(size_t i=0;i<uids.size();i++){
        rst[i]=index(uids[i],dim);
      }
      return rst;
    }

    void update(size_t uid, EMBType& embedding,int dim=0){
      if(is_merge.load()==false){
        is_merge.store(true);
      }
      RWMutexType::WriteLock lock(m_rw_mtx);
      size_t update_times=get_version_id(dim,uid);
      update_emb(update_times,uid,dim,embedding);
    }
    void update(size_t uid, const EMBType& embedding,int dim=0){
      if(is_merge.load()==false){
        is_merge.store(true);
      }
      RWMutexType::WriteLock lock(m_rw_mtx);
      size_t update_times=get_version_id(dim,uid);
      update_emb(update_times,uid,dim,embedding);
    }
    
    void update(std::vector<size_t> &uids, std::vector<EMBType> &embedding,int dim=0){
      if(is_merge.load()==false){
        is_merge.store(true);
      }
      RWMutexType::WriteLock lock(m_rw_mtx);
      #pragma omp parallel for
      for(size_t i=0;i<uids.size();i++){
        size_t update_times=get_version_id(dim,uids[i]);
        update_emb(update_times,uids[i],dim,embedding[i]);
      }
    }
    void update(const std::vector<size_t> &uids, const std::vector<EMBType> &embedding,int dim=0){
      if(is_merge.load()==false){
        is_merge.store(true);
      }
      RWMutexType::WriteLock lock(m_rw_mtx);
      CHECK_EQ(uids.size(),embedding.size());
      #pragma omp parallel for
      for(size_t i=0;i<uids.size();i++){
        size_t update_times=get_version_id(dim,uids[i]);
        update_emb(update_times,uids[i],dim,embedding[i]);
      }
    }
    void BackToInitial(){
      m_num_nodes=InitialEmb<EMBType>::GetInstance()->num_nodes;
      m_hidden_dim=InitialEmb<EMBType>::GetInstance()->hidden_dim;
      m_device=InitialEmb<EMBType>::GetInstance()->device;
      {
        torch::NoGradGuard no_grad;
        auto _initial_embs=InitialEmb<EMBType>::GetInstance()->initial_embs;
        for(size_t i=0;i<_initial_embs.size();i++){
          m_embedding[i] = _initial_embs[i].clone();
        }
      }
      for(size_t i=0;i<m_update_count.size();i++){
        m_update_count[i].clear();
        m_update_emb[i].clear();
        m_update_emb[i].shrink_to_fit();
      }
      if(is_merge.load()==true){
        is_merge.store(false);
      }
    }
    void SaveWindowInitial(){
      m_window_initial.reset(new HistoryState);
      m_window_initial->his_num_nodes=m_num_nodes;
      m_window_initial->his_emb.resize(m_embedding.size());
      std::copy(m_embedding.begin(),m_embedding.end(),m_window_initial->his_emb.begin());
      m_window_initial->his_update_count.resize(m_update_count.size());
      std::copy(m_update_count.begin(),m_update_count.end(),m_window_initial->his_update_count.begin());
    }
    void ResetWindowInitial(){
      if(m_window_initial ==nullptr){
        std::cout<<" not save window initial!"<<std::endl;
      }else{
        m_num_nodes = m_window_initial->his_num_nodes;
        m_embedding.resize(m_window_initial->his_emb.size());
        std::copy(m_window_initial->his_emb.begin(),m_window_initial->his_emb.end(),m_embedding.begin());
        m_update_count.resize(m_window_initial->his_update_count.size());
        std::copy(m_window_initial->his_update_count.begin(),m_window_initial->his_update_count.end(),m_update_count.begin());
        for(size_t dim_id=0;dim_id<m_embedding.size();dim_id++){
          m_update_emb[dim_id].clear();
          m_update_emb[dim_id].shrink_to_fit();
          m_update_count[dim_id].clear();
          m_embedding[dim_id].set_requires_grad(false);
        }
        if(is_merge.load()==true){
          is_merge.store(false);
        }
      }
    }
    void SaveWindowStep(){
      {
        torch::NoGradGuard no_grad;
        if(m_window_step!=nullptr) 
          return;
        m_window_step.reset(new HistoryState);
        m_window_step->his_num_nodes=m_num_nodes;
        m_window_step->his_emb.resize(m_embedding.size());
        std::copy(m_embedding.begin(),m_embedding.end(),m_window_step->his_emb.begin());
        m_window_step->his_update_count.resize(m_update_count.size());
        std::copy(m_update_count.begin(),m_update_count.end(),m_window_step->his_update_count.begin());

        std::vector<EMBType>  tmp(m_num_nodes);
        for(size_t dim_id=0;dim_id<m_window_step->his_emb.size();dim_id++){
          m_window_step->his_emb[dim_id].set_requires_grad(false);
          for(size_t uid=0;uid<m_num_nodes;uid++){
            size_t update_times=get_version_id(m_window_step->his_update_count,dim_id,uid);
            if(update_times!=0){
                m_window_step->his_emb[dim_id][uid]= m_update_emb[dim_id][update_times-1][uid].detach();
                // m_embedding[dim_id][uid].set_data(m_update_emb[dim_id][update_times-1][uid].detach());
            }
          }
          // std::cout<<m_embedding[dim_id].sizes();
        }
      }
        
    }
    void ResetWindowStep(){
      {
        torch::NoGradGuard no_grad;
        if(m_window_step==nullptr) return;
        std::cout<<"---------embedding resetwindowstep-----------"<<std::endl;
        m_num_nodes = m_window_step->his_num_nodes;
        m_embedding.resize(m_window_step->his_emb.size());
        // torch::NoGradGuard no_grad;
        std::copy(m_window_step->his_emb.begin(),m_window_step->his_emb.end(),m_embedding.begin());
        m_update_count.resize(m_window_step->his_update_count.size());
        std::copy(m_window_step->his_update_count.begin(),m_window_step->his_update_count.end(),m_update_count.begin());
        for(size_t dim_id=0;dim_id<m_embedding.size();dim_id++){
          m_update_emb[dim_id].clear();
          m_update_emb[dim_id].shrink_to_fit();
          m_update_count[dim_id].clear();
          m_embedding[dim_id].set_requires_grad(false);
        }
        if(is_merge.load()==true){
          is_merge.store(false);
        }
      }
    

      m_window_step=nullptr;
    }
    void Merge(){
      RWMutexType::WriteLock lock(m_rw_mtx);
      {
        torch::NoGradGuard no_grad;
        std::vector<EMBType>  tmp(m_num_nodes);
        for(size_t dim_id=0;dim_id<m_embedding.size();dim_id++){
          m_embedding[dim_id].set_requires_grad(false);
          for(size_t uid=0;uid<m_num_nodes;uid++){
            size_t update_times=get_version_id(dim_id,uid);
            if(update_times!=0){
                m_embedding[dim_id][uid]= m_update_emb[dim_id][update_times-1][uid].detach();
                // m_embedding[dim_id][uid].set_data(m_update_emb[dim_id][update_times-1][uid].detach());
            }
          }
          // std::cout<<m_embedding[dim_id].sizes();
        }
      }
      for(size_t dim_id=0;dim_id<m_embedding.size();dim_id++){
          m_update_emb[dim_id].clear();
          m_update_emb[dim_id].shrink_to_fit();
          m_update_count[dim_id].clear();
          m_embedding[dim_id].set_requires_grad(false);
      }
      if(is_merge.load()==true){
        is_merge.store(false);
      }

    }
    void clear(){
      m_hidden_dim=0;
      m_num_nodes=0;
      for(size_t dim_id=0;dim_id<m_embedding.size();dim_id++){
        m_update_emb[dim_id].clear();
        m_update_emb[dim_id].shrink_to_fit();
        m_update_count[dim_id].clear();
      }
      m_embedding.clear();
      m_embedding.shrink_to_fit();
      if(is_merge.load()==true){
        is_merge.store(false);
      }
    }
  protected:
    RWMutexType m_rw_mtx;

    // operator for uodate_count
    size_t get_version_id(size_t dim,size_t index){
      return static_cast<size_t>(m_update_count[dim].get_count(index));
    } 
    size_t get_version_id(CountMapType & update_count,size_t dim,size_t index){
      return static_cast<size_t>(update_count[dim].get_count(index));
    } 
    void add_one(size_t dim,size_t index){
      m_update_count[dim].add_one(index);
    }
    
    //operator for update_emb
    EMBType index_update(int vid,int nid,int dim_id){
      // CHECK(static_cast<size_t>(dim_id)<m_update_emb.size())<<"dim_id is "<<dim_id<<" out of range";
      // CHECK(m_update_emb[dim_id][vid].find(nid)!=m_update_emb[dim_id][vid].end())
      // <<"nid:"<<nid<<" not in dim_id:"<<dim_id<<" vid:"<<vid;
      return m_update_emb[dim_id][vid][nid];
      // log<<"index update:";
      // if(rst.requires_grad()){
      //   log<<"rst.requires_grad_(True)"<<std::endl;
      // }else{
      //   log<<"rst.requires_grad_(False)"<<std::endl;
      // }

    }
    EMBType index_emb(int nid,int dim_id=0){
      // CHECK(static_cast<size_t>(dim_id)<m_embedding.size())<<"dim_id is "<<dim_id<<"out of range";
      return m_embedding[dim_id][nid];

    }
    void update_emb(int vid,int nid,int dim_id,const EMBType &emb){
      // CHECK(static_cast<size_t>(dim_id)<m_update_emb.size())<<"dim_id is "<<dim_id<<" out of range";
      // CHECK(emb.size(0)==m_hidden_dim)
      // <<"dim may be "<<m_hidden_dim<<" "<<" and your embedding dim is "<<emb.size(0);
      // //kuorong 
      if(static_cast<size_t>(vid)>=m_update_emb[dim_id].size()){
        #pragma omp critical  
        {
          if(static_cast<size_t>(vid)>=m_update_emb[dim_id].size()){
            if(m_update_emb[dim_id].size()<=20)
              m_update_emb[dim_id].resize(int(m_update_emb[dim_id].size()*2)+1);
            else{
              m_update_emb[dim_id].resize(int(m_update_emb[dim_id].size()*1.5)+1);
            }
          }
        }
      }
      // CHECK(static_cast<size_t>(vid)<m_update_emb[dim_id].size())
      // <<"vid:"<<vid<<" size:"<<m_update_emb[dim_id].size();

      // CHECK(m_update_emb[dim_id][vid].find(nid)==m_update_emb[dim_id][vid].end())
      // <<" maybe inplace error!\n"
      // <<"dim_id:"<<dim_id<<" vid:"<<vid<<" nid:"<<nid;
      #pragma omp critical  
      {
        m_update_emb[dim_id][vid][nid]=emb.to(m_device);
        add_one(dim_id,nid);
      }
      

    }
  public:
    ~DyNodeEmbedding(){

      // log.close();
    }
  private:

    static DyNodeEmbedding::ptr DyNodeEmbeddingPtr;
    DyNodeEmbedding(){
      m_device=Session::GetInstance()->GetDevice();
    }
    int64_t m_hidden_dim{0};
    size_t m_num_nodes{0};
    torch::Device m_device{torch::kCPU};
    std::vector<EMBType> m_embedding;
    CountMapType m_update_count;
    UpdateEMBType m_update_emb;
    std::atomic_bool is_merge{false};
    struct HistoryState{
      size_t his_num_nodes;
      std::vector<EMBType> his_emb;
      CountMapType his_update_count;
    };
    std::shared_ptr<HistoryState> m_window_initial{nullptr};
    std::shared_ptr<HistoryState> m_window_step{nullptr};
  };

  template<typename EMBType,size_t SAVET>
  typename DyNodeEmbedding<EMBType,SAVET>::ptr 
  DyNodeEmbedding<EMBType,SAVET>::DyNodeEmbeddingPtr=nullptr;

  template<typename EMBType,size_t SAVET>
  typename DyNodeEmbedding<EMBType,SAVET>::ptr 
  DyNodeEmbedding<EMBType,SAVET>::GetInstance(){
    if(DyNodeEmbeddingPtr==nullptr){
      DyNodeEmbeddingPtr.reset(new(std::nothrow)DyNodeEmbedding());
    }
    return DyNodeEmbeddingPtr;
  }

  template<typename EMBType,size_t SAVET>
  typename DyNodeEmbedding<EMBType,SAVET>::ptr 
  DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe(){
    static std::mutex mtx;
    if(DyNodeEmbeddingPtr==nullptr){
      std::unique_lock<std::mutex> locker(mtx);
      if(DyNodeEmbeddingPtr==nullptr){
        DyNodeEmbeddingPtr.reset(new(std::nothrow)DyNodeEmbedding());
      }
    }
    return DyNodeEmbeddingPtr;
  }
} // namespace neutron
#endif