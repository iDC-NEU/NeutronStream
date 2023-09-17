#ifndef NEUTRONSTREAM_ADJLIST_H
#define NEUTRONSTREAM_ADJLIST_H
#include <memory>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <utils/mytime.h>
#include <utils/thread_utils.hpp>
#include <dtype.h>
#include <session.h>
#include <iostream>
namespace neutron{
  struct EdgeData{
      float value{1.0f};
      std::string to_string(){
        return std::to_string(value);
      }
  };


  template<typename EdgeDataType>
  struct t_edge{
    size_t src{};
    size_t dst{};
    EdgeDataType edge_data;
    TimePoint time;
    t_edge()=default;
    t_edge(size_t src_, size_t dst_) : src(src_), dst(dst_),time(0) {}
    t_edge(size_t src_, size_t dst_,const EdgeDataType &val_):
    src(src_), dst(dst_),edge_data(val_),time(0) {}
    t_edge(size_t src_, size_t dst_,const EdgeDataType &val_,const TimePoint &time_):
    src(src_), dst(dst_),edge_data(val_),time(time_) {}
    t_edge(const t_edge &other){
        src=other.src;
        dst=other.dst;
        edge_data=other.edge_data;
        time=other.time;
    }
    t_edge &operator=(const t_edge & other){
        if(this==&other){
            return *this;
        }
        src=other.src;
        dst=other.dst;
        edge_data=other.edge_data;
        time=other.time;
        return *this;
    }

    friend std::ostream & operator<<(std::ostream& os, const t_edge& edge){
      os<<"( "<<edge.src<<","<<edge.dst<<" )";
      return os;
    }

  };
  template<typename EdgeDaTaType,int InitialSize=20>
  class dance_t_edge{
    public:
      dance_t_edge() {
        ids.reserve(InitialSize);
        edge_datas.reserve(InitialSize);
        offset.reserve(InitialSize);
      };
      dance_t_edge(const dance_t_edge &other){
        ids=other.ids;
        edge_datas=other.edge_datas;
        offset.clear();
        set_map();
      }
      dance_t_edge &operator= (const dance_t_edge&other){
        if(this==&other){
            return *this;
        }
        ids.resize(other.ids.size());
        edge_datas.resize(other.edge_datas.size());

        std::copy(std::begin(other.ids),std::end(other.ids),std::begin(ids));
        std::copy(std::begin(other.edge_datas),std::end(other.edge_datas),std::begin(edge_datas));
        offset.clear();
        set_map();
        return *this;
      }
      explicit dance_t_edge(std::vector<size_t> &n_ids){
        ids.insert(ids.end(),n_ids.begin(),n_ids.end());
        edge_datas.resize(n_ids.size());
        set_map();
      }
      explicit dance_t_edge(std::vector<size_t> &n_ids, std::vector<EdgeDaTaType> &edgeData){
        ids.insert(ids.end(),n_ids.begin(),n_ids.end());
        edge_datas.insert(edge_datas.end(),edgeData.begin(),edgeData.end());
        set_map();
      } 
      bool isEdge(size_t nid){
        return offset.count(nid)!=0;
      }
      bool is_exist(std::vector<size_t> &nids){
        for(auto &nid : nids){
            if(offset.count(nid)!=0){
                std::cout<<nid<<" has exist! insert error!"<<std::endl;
                return true;
            }
        }
        return false;
      }
      bool is_exist(size_t nid){
            if(offset.count(nid)!=0){
                std::cout<<nid<<" has exist! insert error!"<<std::endl;
                return true;
            }
            return false;
        }
      void set_map(){
          for(size_t i=0;i<ids.size();i++){
              offset[ids[i]]=i;
          }
      }
      void add_neighbors(std::vector<size_t> &nids){
          if(is_exist(nids)) return;
          size_t ids_size=ids.size();
          for(auto &nid : nids){
              offset[nid]=ids_size++;
          }
          ids.insert(ids.end(),nids.begin(),nids.end());
          edge_datas.resize(edge_datas.size()+nids.size());
      }
      void add_neighbors(std::vector<size_t> &nids,std::vector<EdgeDaTaType> &edgedatas){
          if(is_exist(nids)) return;
          size_t ids_size=ids.size();
          for(auto &nid : nids){
              offset[nid]=ids_size++;
          }
          ids.insert(ids.end(),nids.begin(),nids.end());
          edge_datas.insert(edge_datas.end(),edgedatas.begin(),edgedatas.end());
      }

      void add_neighbor(size_t n_id){
          if(is_exist(n_id)) return;
          offset[n_id]=ids.size();
          ids.emplace_back(n_id);
          edge_datas.emplace_back();
      }
      void add_neighbor(size_t n_id,const EdgeDaTaType &edgedata){
          if(is_exist(n_id)) return;
          ids.emplace_back(n_id);
          edge_datas.emplace_back(edgedata);
      }
      size_t size() const{
          return ids.size();
      }
      std::vector<EdgeDaTaType> & EdgeDatas(){
        return edge_datas;
      }
      std::vector<size_t> & Ids(){
        return ids;
      }

    private:
      std::unordered_map<size_t,size_t> offset;
      std::vector<size_t> ids;
      std::vector<EdgeDaTaType> edge_datas;
  };


  template<typename EdgeDaTaType>
  class AdjTableNodes{
    public:
      typedef std::shared_ptr<AdjTableNodes> ptr; 
      AdjTableNodes()=default;
      AdjTableNodes(const AdjTableNodes & other){
          InNeighbors=other.InNeighbors;
          OutNeighbors=other.OutNeighbors;
      }
      AdjTableNodes &operator=(const AdjTableNodes &other){
          if(this == &other){
              return *this;
          }
          InNeighbors=other.InNeighbors;
          OutNeighbors=other.OutNeighbors;
          return *this;
      }
      
      void addOutEdge(size_t dst_id){
          OutNeighbors.add_neighbor(dst_id);
      }
      void addInEdge(size_t src_id){
          InNeighbors.add_neighbor(src_id);
      }

      void addOutEdge(size_t dst_id,const EdgeDaTaType& edgeData){
          OutNeighbors.add_neighbor(dst_id,edgeData);
      }
      void addInEdge(size_t src_id,const EdgeDaTaType &edgeData){
          InNeighbors.add_neighbor(src_id,edgeData);
      }

      void addInEdges(std::vector<size_t> &src_ids,std::vector<EdgeDaTaType> &edgeDatas){
          InNeighbors.add_neighbors(src_ids,edgeDatas);
      }
      void addInEdges(std::vector<size_t> & src_ids){
          InNeighbors.add_neighbors(src_ids);
      };
      void addOutEdges(std::vector<size_t> &dst_ids,std::vector<EdgeDaTaType> &edgeDatas){
          OutNeighbors.add_neighbors(dst_ids,edgeDatas);
      }

      void addOutEdges(std::vector<size_t> &dst_ids){
          OutNeighbors.add_neighbors(dst_ids);
      }
      bool hasInEdge(size_t src){
          return InNeighbors.isEdge(src);
      }
      bool hasOutEdge(size_t dst){
          return OutNeighbors.isEdge(dst);
      }
      bool hasEdge(size_t nid){
          return hasInEdge(nid)|| hasOutEdge(nid);
      }

      std::vector<EdgeDaTaType> inEdgeData(){
          return InNeighbors.EdgeDatas();
      }
      std::vector<size_t> in_neighbors(){
          return InNeighbors.Ids();
      }
      std::vector<EdgeDaTaType> outEdgeData(){
          return OutNeighbors.EdgeDatas();
      }
      std::vector<size_t> out_neighbors(){
          return OutNeighbors.Ids();
      }
      size_t num_in_neighbors() const{
          return InNeighbors.size();
      }
      size_t num_out_neighbors() const{
          return OutNeighbors.size();
      }

    private:
      dance_t_edge<EdgeDaTaType> InNeighbors;
      dance_t_edge<EdgeDaTaType> OutNeighbors;
  };
  
  
  template<typename EdgeDataType=EdgeData>
  class DiAdjList{
    public:
      typedef std::shared_ptr<DiAdjList> ptr;
      DiAdjList()=default;
      DiAdjList(const DiAdjList &other){
        Vertex_Lists.resize(other.Vertex_Lists.size());
        std::copy(std::begin(other.Vertex_Lists),std::end(other.Vertex_Lists),std::begin(Vertex_Lists));
      }
      DiAdjList &operator=(const DiAdjList &other){
        if(this==&other){
            return *this;
        }
        Vertex_Lists.resize(other.Vertex_Lists.size());
        std::copy(std::begin(other.Vertex_Lists),std::end(other.Vertex_Lists),std::begin(Vertex_Lists));
        return *this;
      }
      
      void AddVertexes( size_t num_Vertexs){
        Vertex_Lists.resize(Vertex_Lists.size()+num_Vertexs);
      }
      void CreateBaseEdges(const std::vector<t_edge<EdgeDataType>> &edges){
        std::set<size_t> srcsSet;
        std::set<size_t> dstsSet;
        std::map< size_t ,std::pair<std::vector<size_t>,std::vector<EdgeDataType>>> src_VecEdges;
        std::map< size_t,std::pair<std::vector<size_t>,std::vector<EdgeDataType>>>  dst_VecEdges;
        size_t numEdges=edges.size();
        for ( size_t i = 0; i < numEdges; ++i) {
            auto src =edges[i].src;
            auto dst =edges[i].dst;
            srcsSet.insert(src);
            dstsSet.insert(dst);
            src_VecEdges[src].first.emplace_back(dst);
            src_VecEdges[src].second.emplace_back(edges[i].edge_data);
            dst_VecEdges[dst].first.emplace_back(src);
            dst_VecEdges[dst].second.emplace_back(edges[i].edge_data);
        }
        //parallel
        std::vector<size_t> parallelSrcsSet(srcsSet.begin(),srcsSet.end());
        std::vector<size_t> parallelDstsSet(dstsSet.begin(),dstsSet.end());
        auto rst_out=Session::GetInstance()->getThreadPool()->submit([&]{
          //Out
          #pragma omp parallel for
          for(size_t i=0;i<parallelSrcsSet.size();i++){
              Vertex_Lists[parallelSrcsSet[i]].addOutEdges(src_VecEdges[parallelSrcsSet[i]].first,src_VecEdges[parallelSrcsSet[i]].second);
          }
        });
       auto rst_in= Session::GetInstance()->getThreadPool()->submit([&]{
          //In
          #pragma omp parallel for
          for(size_t i=0;i<parallelDstsSet.size();i++){
              Vertex_Lists[parallelDstsSet[i]].addInEdges(dst_VecEdges[parallelDstsSet[i]].first,dst_VecEdges[parallelDstsSet[i]].second);
          }
        });
        rst_out.get();
        rst_in.get();
        
      }
      void AddEdge(t_edge<EdgeDataType> &edge){
        auto rst_in=Session::GetInstance()->getThreadPool()->submit([&]{
          Vertex_Lists[edge.src].addOutEdge(edge.dst,edge.edge_data);
        });
        auto rst_out=Session::GetInstance()->getThreadPool()->submit([&]{
          Vertex_Lists[edge.dst].addInEdge(edge.src,edge.edge_data);
        });
        rst_in.get();
        rst_out.get();
      }
      void AddEdge(size_t src_id,size_t dst_id){
        // auto rst_in=Session::GetInstance()->getThreadPool()->submit([&]{
        //   Vertex_Lists[src_id].addOutEdge(dst_id);
        // });
        // auto rst_out=Session::GetInstance()->getThreadPool()->submit([&]{
        //   Vertex_Lists[dst_id].addInEdge(src_id);
        // });
        // rst_in.get();
        // rst_out.get();
        std::thread in_thread([&]{
          Vertex_Lists[src_id].addOutEdge(dst_id);  
        });
        std::thread out_thread([&]{
          Vertex_Lists[dst_id].addInEdge(src_id);
        });
        in_thread.join();
        out_thread.join();
      }
      void AddEdge(size_t src_id,size_t dst_id,const EdgeDataType& edgedata){
        auto rst_in=Session::GetInstance()->getThreadPool()->submit([&]{
          Vertex_Lists[src_id].addOutEdge(dst_id,edgedata);
        });
        auto rst_out=Session::GetInstance()->getThreadPool()->submit([&]{
          Vertex_Lists[dst_id].addInEdge(src_id,edgedata);
        });
        rst_in.get();
        rst_out.get();
      }
      void AddEdges(std::vector<t_edge<EdgeDataType>> &edges){
        CreateBaseEdges(edges);
      }
      std::vector<t_edge<EdgeDataType>> OutEdges(size_t n_id){
        std::vector<t_edge<EdgeDataType>> rstEdge;
        std::vector<size_t> neighbors =Vertex_Lists[n_id].out_neighbors();
        std::vector<EdgeDataType> nei_datas=Vertex_Lists[n_id].outEdgeData();
        rstEdge.resize(neighbors.size());
        #pragma omp parallel for
        for(size_t i=0;i<neighbors.size();i++){
            rstEdge[i]={n_id,neighbors[i],nei_datas[i]};
        }
        return rstEdge;
      }
      std::vector<t_edge<EdgeDataType>> InEdges(size_t n_id){
        std::vector<t_edge<EdgeDataType>> rstEdge;
        std::vector<size_t> src_neighbors =Vertex_Lists[n_id].in_neighbors();
        std::vector<EdgeDataType> src_nei_datas=Vertex_Lists[n_id].inEdgeData();
        rstEdge.resize(src_neighbors.size());
        #pragma omp parallel for
        for(size_t i=0;i<src_neighbors.size();i++){
            rstEdge[i]={src_neighbors[i],n_id,src_nei_datas[i]};
        }
        return rstEdge;
      }
      std::vector<t_edge<EdgeDataType>> Edges(size_t n_id){
        std::vector<t_edge<EdgeDataType>> rstEdges;
        auto rst_in_future=Session::GetInstance()->getThreadPool()->submit([&]{
          return InEdges(n_id);
        });
        auto rst_out_future=Session::GetInstance()->getThreadPool()->submit([&]{
          return OutEdges(n_id);
        });
        auto rst_in=rst_in_future.get();
        rstEdges.insert(rstEdges.end(),rst_in.begin(),rst_in.end());
        auto rst_out=rst_out_future.get();
        rstEdges.insert(rstEdges.end(),rst_out.begin(),rst_out.end());
        return rstEdges;
      }

      std::vector<size_t> OutNeighbors(size_t n_id){
        return Vertex_Lists[n_id].out_neighbors();
      }
      std::vector<size_t> InNeighbors(size_t nid){
        return Vertex_Lists[nid].in_neighbors();
      }
      bool hasEdge( size_t sid, size_t did){
        
        if(sid>Vertex_Lists.size() || did >Vertex_Lists.size()){
            return false;
        }
        return Vertex_Lists[sid].hasOutEdge(did) && Vertex_Lists[did].hasInEdge(sid);
      }
      size_t Degree(size_t uid){
        return InDegree(uid)+OutDegree(uid);
      }
      size_t InDegree(size_t uid){
        return Vertex_Lists[uid].num_in_neighbors();
      }
      size_t OutDegree(size_t uid){
        return Vertex_Lists[uid].num_out_neighbors();
      }
      void clear(){
        Vertex_Lists.clear();
        Vertex_Lists.shrink_to_fit();
      }
    private:
        std::vector<AdjTableNodes<EdgeDataType>> Vertex_Lists;
  };
  
}
#endif