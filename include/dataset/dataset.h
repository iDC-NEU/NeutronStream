#ifndef NEUTRONSTREAM_DATASET_H
#define NEUTRONSTREAM_DATASET_H
#include <string>
#include <vector>
#include <torch/torch.h>
#include <dygstore/event.h>
#include <session.h>
#include <utils/mytime.h>
namespace neutron{
  class Dataset {
    private:
      
      std::string dataset_name;
      std::string dataset_dir;
      std::vector<int64_t> shape_;
      bool isTemporal=false;
      TimePoint first_time=0;
      std::string initial_graph_path="graph0.txt";
      size_t num_vertices{0};
      size_t num_edges{0};
      std::string embedding_path="embedding0.txt";
      std::string events_file;
      std::vector<std::vector<size_t>> edges;
      std::vector<torch::Tensor> embeddings;
      std::queue<torch::Tensor> add_embeddings;
      std::vector<Event> evtList;
      size_t batchNum;
      size_t blockNum;
    public:
      Dataset() = default;
      Dataset(std::string name,std::string datadir,bool is_temporal=false);
      const std::vector<Event> & getEventlist() { return evtList; }
      Dataset(std::string name,std::string datadir,const std::vector<int64_t> &shape,bool is_temporal=false);
      Dataset(const std::vector<Event> & evtList);
      void process();
      void readEdge(const std::string& path);
      void readEmbedding(const std::string& path);
      void readEvents(const std::string &path);

      size_t getNumVertices() const {return this->num_vertices;}
      size_t getNumEdges() const {return this->num_edges;}
      size_t getBatchNum() const{return batchNum;}
      size_t getBlockNum() const{return blockNum;}
      size_t getNumEvents() const{return evtList.size();}

      std::vector<Event> getIdxBatch(size_t idx);
      std::vector<Event> getSlideIdxBatch(size_t idx,int e);
      std::vector<Event> getTrainIdxBatch(size_t idx,size_t train_event_num);
      std::vector<Event> getTrainDataSet(size_t train_event_num);
      std::vector<Event> getValidDataset(size_t train_event_num);
      std::vector<Event> getValidIdxBatch(int idx,size_t train_event_num);
      bool judge_Slide(size_t idx,int e);
      std::vector<Event> getMissIdxBatch(int e);

      static  bool file_exist(const std::string &path){
        std::ifstream f(path.c_str());
        return f.good();
      }
  };
} // namespace neutron

#endif