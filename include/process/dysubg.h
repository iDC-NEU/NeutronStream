#ifndef NEUTRONSTREM_DYSUBG_HPP
#define NEUTRONSTREM_DYSUBG_HPP
#include <dygstore/event.h>
#include <vector>
#include <utils/thread_safe/thread_safe_set.h>
#include <utils/thread_safe/thread_safe_map.h>
#include <dygstore/adjlist.hpp>
#include <unordered_map>
#include <utils/mytime.h>

namespace neutron
{
  //只保留局部的拓扑结构
  class DySubGraph{
  public:
  
    typedef thread_safe::map<size_t,std::vector<size_t>> AdjList;
    typedef thread_safe::set<size_t> NodeSet;
    typedef std::shared_ptr<DySubGraph> ptr;
    
    DySubGraph()=default;
    virtual ~DySubGraph() {}
    DySubGraph(DySubGraph& other){
      event=other.event;
      numEdges=other.numEdges;
      updata_vertices.resize(other.updata_vertices.size());
      std::copy(other.updata_vertices.begin(),other.updata_vertices.end(),updata_vertices.begin());
      vertices.resize(other.vertices.size());
      std::copy(other.vertices.begin(),other.vertices.end(),vertices.begin());
      timeStamps=other.timeStamps;
      adj_in_list=other.adj_in_list;
      adj_out_list=other.adj_out_list;
    }
    DySubGraph& operator=( DySubGraph& other){
      if(this==&other){
        return *this;
      }
      event=other.event;
      numEdges=other.numEdges;
      updata_vertices.resize(other.updata_vertices.size());
      std::copy(other.updata_vertices.begin(),other.updata_vertices.end(),updata_vertices.begin());
      vertices.resize(other.vertices.size());
      std::copy(other.vertices.begin(),other.vertices.end(),vertices.begin());
      timeStamps=other.timeStamps;
      adj_in_list=other.adj_in_list;
      adj_out_list=other.adj_out_list;
      return *this;
    }
    DySubGraph(const Event &event,int hop=1);
    Event get_event() const {return event;}
    size_t get_event_src() const{return event.src_id;};
    size_t get_event_dst() const{return event.dst_id;};
    TimePoint get_time_point() const {return event.time_point;}

    void setUpdateSet(const std::vector<size_t> &u_set){
      std::unordered_set<size_t> updateSet;
      updateSet.insert(updata_vertices.begin(),updata_vertices.end());
      updateSet.insert(u_set.begin(),u_set.end());
      updata_vertices=std::vector<size_t>(updateSet.begin(),updateSet.end());
    }
    
    std::vector<size_t> in_neighbors(size_t u_id) {
      std::vector<size_t> rst;
      if(adj_in_list.find(u_id)!=adj_in_list.end()){
          return adj_in_list[u_id];
      }
      return rst;
    }
    std::vector<size_t> out_neighbors(size_t u_id){
      std::vector<size_t> rst;
      if(adj_out_list.find(u_id)!=adj_out_list.end()){
          return adj_out_list[u_id];
      }
      return rst;
    }
    std::vector<size_t> get_update_vertex() const {
        return updata_vertices;
    }

    std::vector<size_t> get_v_set() const {
        return vertices;
    }
    size_t in_degree(size_t uid)  {
        if(adj_in_list.find(uid)!=adj_in_list.end()){
            return adj_in_list[uid].size();
        }
        return 0;
    }
    size_t out_degree(size_t uid){
        if(adj_out_list.find(uid)!=adj_out_list.end()){
            return adj_out_list[uid].size();
        }
        return 0;
    }
    std::vector<TimePoint> get_pre_times(std::vector<size_t> &uids){
      std::vector<TimePoint> rst(uids.size());
      for(size_t i=0;i<uids.size();i++){
        rst[i]=timeStamps[uids[i]];
      }
      return rst;
    }
    TimePoint get_pre_time(size_t uid){
      CHECK(timeStamps.find(uid) != timeStamps.end())<< uid <<"not in dysubg";
      return timeStamps[uid];
    }

  private:
    Event event;
    size_t numEdges{0};
    std::vector<size_t> updata_vertices;
    std::vector<size_t> vertices;
    std::unordered_map<size_t,TimePoint> timeStamps;
    DySubGraph::AdjList adj_in_list;
    DySubGraph::AdjList adj_out_list;
private:
    void make_subgraph(int hop,std::vector<size_t>& frontiers,
    DySubGraph::AdjList &adjlist,NodeSet &vertex_sets,bool is_in_mode);

  };
  
} // namespace neutron

#endif