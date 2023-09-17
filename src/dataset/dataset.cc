#include "dataset/dataset.h"
#include <dygstore/interface.hpp>
#include <utils/util.h>
#include <log/log.h>
namespace neutron{
    Dataset::Dataset(std::string name,std::string datadir,bool is_temporal):
    dataset_name(std::move(name)),dataset_dir(std::move(datadir)),isTemporal(is_temporal){

    initial_graph_path=dataset_dir+initial_graph_path;
    embedding_path=dataset_dir+embedding_path;
    events_file=dataset_dir+dataset_name+".csv";
    process();
    size_t batch_size=Session::GetInstance()->GetBatchSize();
    size_t block_size=Session::GetInstance()->GetBlockSize();
    batchNum=(evtList.size()+batch_size-1)/batch_size;
    blockNum=(evtList.size()+block_size-1)/block_size;
  }
  Dataset::Dataset(std::string name,std::string datadir,const std::vector<int64_t> &shape,bool is_temporal):
    dataset_name(std::move(name)),dataset_dir(std::move(datadir)),isTemporal(is_temporal){
    shape_=shape;
    initial_graph_path=dataset_dir+initial_graph_path;
    embedding_path=dataset_dir+embedding_path;
    events_file=dataset_dir+dataset_name+".csv";
    process();
    size_t batch_size=Session::GetInstance()->GetBatchSize();
    size_t block_size=Session::GetInstance()->GetBlockSize();
    batchNum=(evtList.size()+batch_size-1)/batch_size;
    blockNum=(evtList.size()+block_size-1)/block_size;
    // --------debug
    std::cout<<"-------------------dataset.cc-------------"<<std::endl;
    std::cout<<"\t\t\t\t batch_size:\t"<<batch_size<<std::endl;
    std::cout<<"\t\t\t\t evtList.size():\t"<<evtList.size()<<std::endl;
    std::cout<<"\t\t\t\t batchNum:\t"<<batchNum<<std::endl;
    std::cout<<"-------------------dataset.cc-------------"<<std::endl;
    //--------debug
  }
  Dataset::Dataset(const std::vector<Event> & evtList){
        this->evtList.resize(evtList.size());
        std::copy(evtList.begin(),evtList.end(),this->evtList.begin());
        size_t batch_size=Session::GetInstance()->GetBatchSize();
        batchNum=(evtList.size()+batch_size-1)/batch_size;
  }
  void Dataset::process() {
    if(file_exist(this->events_file)){
        readEvents(this->events_file);
    }
    else{
        NEUTRON_LOG_INFO(LOG_ROOT())<<"no datafile: "<<this->events_file<<std::endl;
        exit(0);
    }
    if(file_exist(this->initial_graph_path)){
        readEdge(this->initial_graph_path);
        Initial(this->num_vertices,this->edges,first_time);
    }
    else{
        NEUTRON_LOG_INFO(LOG_ROOT())<<"no datafile: "<<this->events_file<<std::endl;
        exit(0);
    }
    if(file_exist(this->embedding_path)){
        readEmbedding(this->embedding_path);
        torch::Tensor emb=vec2tensor(this->embeddings);
        if(shape_.empty()){
        //   DyNodeEmbedding::GetInstance()->InitNodeEmbedding(this->embeddings,Session::GetInstance()->GetHiddenDim());
            InitEMB(emb,Session::GetInstance()->GetHiddenDim());
        }
        else{
            // DyNodeEmbedding::GetInstance()->InitNodeEmbedding(this->embeddings,Session::GetInstance()->GetHiddenDim(),shape_);
            InitEMBs(emb,Session::GetInstance()->GetHiddenDim(),shape_[0]);
        }
    }
    else{
        if(shape_.empty()){
            // DyNodeEmbedding::GetInstance()->RandomInitNodeEmbedding(this->num_vertices,shape_);
            RandomInitEMB(this->num_vertices,Session::GetInstance()->GetHiddenDim(),1);
        }else{
            // DyNodeEmbedding::GetInstance()->RandomInitNodeEmbedding(this->num_vertices,Session::GetInstance()->GetHiddenDim());
            RandomInitEMB(this->num_vertices,Session::GetInstance()->GetHiddenDim(),shape_[0]);
        }
    }
  }

  void Dataset::readEdge(const std::string& path) {
    std::vector<std::vector<std::string>> edges_str=readTxt(path);
    if(isTemporal){
        int num_nodes=std::stoi(edges_str[0][0].substr(1));
        this->num_edges=0;
        this->num_vertices=num_nodes;
    }
    else{
        int numEdges=std::stoi(edges_str[0][1].substr(1));
        int num_nodes=std::stoi(edges_str[0][0].substr(1));

        this->num_edges=numEdges;
        this->num_vertices=num_nodes;
        this->edges.resize(2);
        for(int i=1;i<=numEdges;i++){
            size_t u=std::stoi(edges_str[i][0]);
            size_t v=std::stoi(edges_str[i][1]);
            this->edges[0].push_back(u);
            this->edges[1].push_back(v);
        }
    }
  }

  void Dataset::readEmbedding(const std::string& path) {
    std::vector<std::vector<std::string>> embed_str= readTxt(path);
    for(size_t i=0;i<this->num_vertices;i++){
        std::vector<double> features;
        for(const auto& item:embed_str[i]){
            features.push_back(std::stod(item));
        }
        torch::Tensor tmp=torch::tensor(features);
        this->embeddings.push_back(tmp);
    }
    for(size_t i=this->num_vertices;i<embed_str.size();i++){
        std::vector<double> features;
        for(const auto& item:embed_str[i]){
            features.push_back(std::stod(item));
        }
        torch::Tensor tmp=torch::tensor(features);
        this->add_embeddings.push(tmp);
    }
  }

  void Dataset::readEvents(const std::string &path) {
    std::vector<std::vector<std::string>> strs= readCSV(path);
    if(isTemporal){
        auto startTime=std::stol(strs[1][2]);
        for(size_t i=1;i<strs.size();i++){
            
            size_t sid=std::stoi(strs[i][0]);
            size_t did=std::stoi(strs[i][1]);
            auto timeStamp=std::stol(strs[i][2]);
            auto time=TimePoint(timeStamp-startTime);
            if(i==1){
             first_time=time; 
            }
            evtList.emplace_back(sid,did,time,"AE");
        }
    }
    else if(this->dataset_name=="github"||this->dataset_name=="social"){
        for(size_t i=1;i<strs.size();i++){
            size_t sid=std::stoi(strs[i][0]);
            size_t did=std::stoi(strs[i][1]);
            auto time=TimePoint(strs[i][2]);
            if(i==1){
             first_time=time; 
            }
            evtList.emplace_back(sid,did,time,strs[i][3]);
        }
    }
    else{
        for(size_t i=1;i<strs.size();i++){
            size_t sid=std::stoi(strs[i][0]);
            size_t did=std::stoi(strs[i][1]);
            auto time=TimePoint(std::stol(strs[i][2]));
            if(i==1){
             first_time=time; 
            }
            evtList.emplace_back(sid,did,time,strs[i][3]);
        }
    }
  }

  bool Dataset::judge_Slide(size_t idx,int e){
    size_t batch_size=Session::GetInstance()->GetBatchSize();
    size_t block_size=Session::GetInstance()->GetBlockSize();
    size_t start_pos=batch_size*idx+ e*block_size;
    // int end_pos=start_pos+batch_size;
    //auto begin=this->evtList.begin()+start_pos;
    // auto end=this->evtList.begin()+end_pos;
    if(start_pos>=evtList.size()){
        return false;
    }
    return true;
  }
  std::vector<Event> Dataset::getMissIdxBatch(int e){
    size_t block_size=Session::GetInstance()->GetBlockSize();
    size_t start_pos=0;
    size_t end_pos=e*block_size;
    auto begin=this->evtList.begin()+start_pos;
    auto end=this->evtList.begin()+end_pos;
    if(end_pos>=evtList.size()){
        end=evtList.end();
    }
    std::vector<Event> rst(begin,end);
    return  rst;
  }

  std::vector<Event> Dataset::getSlideIdxBatch(size_t idx,int e) {

    size_t batch_size=Session::GetInstance()->GetBatchSize();
    size_t block_size=Session::GetInstance()->GetBlockSize();
    size_t start_pos=batch_size*idx+ e*block_size;
    size_t end_pos=start_pos+batch_size;
    auto begin=this->evtList.begin()+start_pos;
    auto end=this->evtList.begin()+end_pos;
    if(end_pos>=evtList.size()){
        end=evtList.end();
    }
    std::vector<Event> rst(begin,end);
    return  rst;

  }

  std::vector<Event> Dataset::getIdxBatch(size_t idx) {

    size_t batch_size=Session::GetInstance()->GetBatchSize();

    size_t start_pos=batch_size*idx;
    size_t end_pos=start_pos+batch_size;
    auto begin=this->evtList.begin()+start_pos;
    auto end=this->evtList.begin()+end_pos;
    if(end_pos>=evtList.size()){
        end=evtList.end();
    }
    std::vector<Event> rst(begin,end);
    return  rst;
  }

  std::vector<Event> Dataset::getTrainIdxBatch(size_t idx,size_t train_event_num) {

      int batch_size=static_cast<int>(Session::GetInstance()->GetBatchSize());

      int start_pos=batch_size*static_cast<int>(idx);
      int end_pos=start_pos+batch_size;
      if(end_pos>=static_cast<int>(train_event_num)){
          end_pos=train_event_num;
      }
      auto begin=this->evtList.begin()+start_pos;
      auto end=this->evtList.begin()+end_pos;

      std::vector<Event> rst(begin,end);
      return  rst;

  }

    std::vector<Event> Dataset::getTrainDataSet(size_t train_event_num){
        int end_pos=static_cast<int>(train_event_num);
        auto begin=this->evtList.begin();
        auto end=this->evtList.begin()+end_pos;
        std::vector<Event> rst(begin,end);
        return  rst;
    }
    std::vector<Event> Dataset::getValidDataset(size_t train_event_num){
        size_t start_pos=train_event_num;
        auto begin=this->evtList.begin()+start_pos;
        auto end=this->evtList.end();
        std::vector<Event> rst(begin,end);
        return  rst;
    }


  std::vector<Event> Dataset::getValidIdxBatch(int idx,size_t train_event_num){
      
      size_t batch_size=Session::GetInstance()->GetBatchSize();

      size_t start_pos=batch_size*idx+train_event_num;
      size_t end_pos=start_pos+batch_size;
      auto begin=this->evtList.begin()+start_pos;
      auto end=this->evtList.begin()+end_pos;
      if(end_pos>=evtList.size()){
          end=evtList.end();
      }
      std::vector<Event> rst(begin,end);
      return  rst;
  }

}