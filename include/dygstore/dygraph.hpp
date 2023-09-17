#ifndef NEUTRONSTREAM_DIDYGRAPH_H
#define NEUTRONSTREAM_DIDYGRAPH_H
#include <memory>
#include <dygstore/timebar.h>
#include <dygstore/adjlist.hpp>
#include <torch/torch.h>
namespace neutron
{
  template<class EdgeDataType>
  struct InitialDiDyGraph{
    typedef std::shared_ptr<InitialDiDyGraph> ptr;
    static InitialDiDyGraph::ptr GetInstance();
    size_t numNodes{};
    size_t numEdges{};
    DiAdjList<EdgeDataType> adjList;
    TimePoint first_time;
    void Initial(size_t num_nodes,std::vector<std::vector<size_t>> &edges,TimePoint time=0){
      numNodes=num_nodes;
      first_time=time;
      adjList.AddVertexes(num_nodes);
      if(edges.empty()){
        return ;
      }
      size_t edge_num =edges[0].size();
      numEdges+=edge_num;
      std::vector<t_edge<EdgeDataType>> baseEdges;
      baseEdges.resize(edge_num);
      #pragma omp parallel for
      for(size_t i=0;i<edge_num;i++){
          size_t src_id=edges[0][i];
          size_t dst_id =edges[1][i];
          baseEdges[i]={src_id,dst_id};
      }
      adjList.CreateBaseEdges(baseEdges);
    }
    void clear(){
      numEdges=0;
      numNodes=0;
      adjList.clear();
      first_time=0;
    }

    private:
    static InitialDiDyGraph::ptr InitialDiDyGraphPtr;
  };


  template<class EdgeDataType>
  typename InitialDiDyGraph<EdgeDataType>::ptr InitialDiDyGraph<EdgeDataType>::InitialDiDyGraphPtr=nullptr;
  template<class EdgeDataType>
  typename InitialDiDyGraph<EdgeDataType>::ptr InitialDiDyGraph<EdgeDataType>::GetInstance(){
    if(InitialDiDyGraphPtr==nullptr){
      InitialDiDyGraphPtr.reset(new(std::nothrow) InitialDiDyGraph());
    }
    return InitialDiDyGraphPtr;
  }


  template<typename EdgeDataType>
  class DyGraph{
    public:
      typedef std::shared_ptr<DyGraph> ptr;
      static DyGraph::ptr getptr(){
        return std::make_shared<DyGraph>();
      }
    private:


  };

  template<typename EdgeDataType>
  class DiDyGraph:public DyGraph<EdgeDataType>{
  public:
    typedef std::shared_ptr<DiDyGraph> ptr;
    static DiDyGraph::ptr GetInstance();
    static DiDyGraph::ptr GetInstanceThreadSafe();
    void BackToInitial(size_t num_edges,size_t num_nodes,const DiAdjList<EdgeDataType> &adjlist){
      numEdges.store(num_edges);
      numNodes.store(num_nodes);
      adjList=adjlist;
    }
    void SaveWindowInitial(){
      m_window_initial.reset(new HistoryState);
      m_window_initial->his_num_nodes=numNodes.load();
      m_window_initial->his_num_edges=numEdges.load();
      m_window_initial->his_adjList=adjList;
    }
    void ResetWindowInitial(){
      if(m_window_initial==nullptr){
        std::cout<<"not save graph topologhy in window initial"<<std::endl;
        return;
      }
      numNodes.store(m_window_initial->his_num_nodes);
      numEdges.store(m_window_initial->his_num_edges);
      adjList=m_window_initial->his_adjList;
    }

    void SaveWindowStep(){
      if(m_window_step != nullptr){
        return ;
      }
      m_window_step.reset(new HistoryState);
      m_window_step->his_num_nodes=numNodes.load();
      m_window_step->his_num_edges=numEdges.load();
      m_window_step->his_adjList=adjList;
    }

    void get_information(){
      std::cout<<"num nodes:\t"<<numNodes.load()<<"\t\t"<<"num_edges:\t"<<numEdges.load()<<std::endl;

    }

    void ResetWindowStep(){
      if(m_window_step == nullptr){
        std::cout<<"not save graph topology in wiondow next step;"<<std::endl;
        return;
      }
      numNodes.store(m_window_step->his_num_nodes);
      numEdges.store(m_window_step->his_num_edges);
      adjList=m_window_step->his_adjList;
      m_window_step=nullptr;
    }

    size_t NumEdges() const {
      return numEdges.load();
    }

    size_t NumNodes() const {
      return numNodes.load();
    }

    std::vector<t_edge<EdgeDataType>> OutEdges(size_t nid) {
        CHECK(HasVertex(nid))<<nid <<" nid out of range";
        return adjList.OutEdges(nid);
    }

    std::vector<t_edge<EdgeDataType>> InEdges(size_t nid)  {
        CHECK(HasVertex(nid))<<nid <<" nid out of range";
        return adjList.InEdges(nid);
    }

    std::vector<t_edge<EdgeDataType>> Edges(size_t nid) {
        CHECK(HasVertex(nid))<<nid <<" nid out of range";
        return adjList.Edges(nid);
    }

    bool HasVertex(size_t nid) const{
        return nid<numNodes.load();
    }

    bool HasEdge(size_t src_id,size_t dst_id){
        if(HasVertex(src_id) && HasVertex(dst_id)){
            return adjList.hasEdge(src_id,dst_id);
        }
        return false;
    }

    size_t Degree(size_t nid) {
        CHECK(HasVertex(nid))<<nid << " out of range";
        return adjList.Degree(nid);
    }

    size_t InDegree(size_t nid){
        CHECK(HasVertex(nid))<<nid << " out of range";
        return adjList.InDegree(nid);
    }

    size_t OutDegree(size_t nid){
        CHECK(HasVertex(nid))<<nid << " out of range";
        return adjList.OutDegree(nid);
    }

    std::vector<size_t> OutNeighbors(size_t nid){
        CHECK(HasVertex(nid))<<nid <<" src_id out of range";
        return adjList.OutNeighbors(nid);
    }

    std::vector<size_t> InNeighbors(size_t nid){
        CHECK(HasVertex(nid))<<nid <<" src_id out of range";
        return adjList.InNeighbors(nid);
    }

    void AddEdge(size_t src_id,size_t dst_id){
      if(!HasEdge(src_id,dst_id)){
        numEdges++;
        adjList.AddEdge(src_id,dst_id);
//            std::cout<<"add Edge ("<<src_id <<" , "<< dst_id<< ")"<<std::endl;
      }
    }

    void AddEdge(size_t src_id,size_t dst_id,const EdgeDataType &edgedata){
      if(!HasEdge(src_id,dst_id)){
          numEdges++;
          adjList.AddEdge(src_id,dst_id,edgedata);
      }
    }

    void AddEdges(std::vector<std::vector<size_t>> &edges,std::vector<EdgeDataType> &edgedatas){
      if(edges.empty()) return ;
      size_t edge_num =edges[0].size();
      numEdges+=edge_num;
      CHECK(edgedatas.size()!=edge_num)
      <<"edgeDatas size is not equal to edge num edgeData size="
      <<edgedatas.size()<<" edges size:"<<edges.size();
      std::vector<t_edge<EdgeDataType>> baseEdges;
      baseEdges.resize(edge_num);
      #pragma omp parallel for
      for(size_t i=0;i<edge_num;i++){
          size_t src_id=edges[0][i];
          size_t dst_id =edges[1][i];
          EdgeDataType edgedata =edgedatas[i];
          CHECK(!HasEdge(src_id,dst_id))<<"SRC_ID:"<<src_id<<" DST_ID:"<<dst_id;
          baseEdges[i]={src_id,dst_id,edgedata};
      }
      adjList.CreateBaseEdges(baseEdges);
    }

    void AddEdges(const std::vector<std::vector<size_t>> &edges){
      if(edges.empty()) return;
      size_t edge_num =edges[0].size();
      numEdges+=edge_num;
      std::vector<t_edge<EdgeDataType>> baseEdges;
      baseEdges.resize(edge_num);
      #pragma omp parallel for
      for(size_t i=0;i<edge_num;i++){
          size_t src_id=edges[0][i];
          size_t dst_id =edges[1][i];
          if(HasEdge(src_id,dst_id)){ continue;}
          CHECK(!HasEdge(src_id,dst_id))<<"SRC_ID:"<<src_id<<" DST_ID:"<<dst_id;
          baseEdges[i]={src_id,dst_id};
      }
      adjList.CreateBaseEdges(baseEdges);
    }

    void AddVertex(size_t nid){
      numNodes++;
      adjList.AddVertexes(1);
    }

    void InitVertexes(size_t num_nodes){
      numNodes.store(num_nodes);
      adjList.AddVertexes(num_nodes);
      
    }

    void InitEdges(const std::vector<std::vector<size_t>> &edges,const std::vector<EdgeDataType>&edgeData){
      if(edges.empty()) return;
      this->AddEdges(edges,edgeData);
    }

    void InitEdges(const std::vector<std::vector<size_t>> &edges){
      if(edges.empty()) return;
      this->AddEdges(edges);

    }
    void clear(){
      numNodes.store(0);
      numEdges.store(0);
      adjList.clear();
      InitialDiDyGraph<EdgeDataType>::GetInstance()->clear();
    }

  private:
    DiDyGraph(){};
    static DiDyGraph::ptr DiDyGraphPtr;
    static std::mutex mtx;
    std::atomic_size_t numNodes;
    std::atomic_size_t numEdges;
    DiAdjList<EdgeDataType> adjList;

    struct HistoryState{
      size_t his_num_nodes;
      size_t his_num_edges;
      DiAdjList<EdgeDataType> his_adjList;
    };
    std::shared_ptr<HistoryState> m_window_initial;
    std::shared_ptr<HistoryState> m_window_step;
  };
  

  template<typename EdgeDataType>
  typename DiDyGraph<EdgeDataType>::ptr DiDyGraph<EdgeDataType>::DiDyGraphPtr=nullptr;

  template<class EdgeDataType>
  typename DiDyGraph<EdgeDataType>::ptr DiDyGraph<EdgeDataType>::GetInstance(){
    if(DiDyGraphPtr==nullptr){
      DiDyGraphPtr.reset(new(std::nothrow)DiDyGraph());
    }
    return DiDyGraphPtr;
  }

  template<class EdgeDataType>
  std::mutex DiDyGraph<EdgeDataType>::mtx;
  template<class EdgeDataType>
  typename DiDyGraph<EdgeDataType>::ptr DiDyGraph<EdgeDataType>::GetInstanceThreadSafe(){
    if(DiDyGraphPtr==nullptr){
      std::unique_lock<std::mutex> locker(mtx);
      if(DiDyGraphPtr==nullptr){
        DiDyGraphPtr.reset(new (std::nothrow) DiDyGraph());
      }
    }
    return DiDyGraphPtr;
  }
} // namespace neutron

#endif