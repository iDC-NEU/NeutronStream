#include <process/dysubg.h>
#include <process/graphop.h>
#include <session.h>
#include <omp.h>

namespace neutron{

  DySubGraph::DySubGraph(const Event &event,int hop){
      this->event = event;
      updata_vertices={event.src_id,event.dst_id};
      DySubGraph::NodeSet vertexes_set(updata_vertices.begin(),updata_vertices.end());
      std::vector<size_t> frontier1({event.src_id});
      make_subgraph(hop,frontier1,adj_in_list,vertexes_set,true);

      std::vector<size_t> frontier2({event.src_id});
      make_subgraph(hop,frontier2,adj_out_list,vertexes_set,false);

      std::vector<size_t> frontier3({event.dst_id});
      make_subgraph(hop,frontier3,adj_in_list,vertexes_set,true);

      std::vector<size_t> frontier4({event.dst_id});
      make_subgraph(hop,frontier4,adj_out_list,vertexes_set,false);

      vertices.resize(vertexes_set.size());
      std::copy(vertexes_set.begin(),vertexes_set.end(),vertices.begin());
      std::vector<TimePoint> times= GetTimePoints(vertices);
      for(size_t i=0;i<times.size();i++){
        timeStamps[vertices[i]] = times[i];
      }
  }

  void DySubGraph::make_subgraph(int hop,std::vector<size_t>& frontiers,
    DySubGraph::AdjList &adjlist,NodeSet &vertex_sets,bool is_in_mode){
      if(hop<=0) return;
      std::vector<std::vector<size_t>> all_neighbors;
      if(hop!=1)
        all_neighbors.resize(frontiers.size());
      if(is_in_mode){
            #pragma omp parallel for
            for(size_t i=0;i<frontiers.size();i++){
                if(!(adjlist.find(frontiers[i])!=adjlist.end())){
                    auto neighbors=InNeighbors(frontiers[i]);
                    if(hop!=1)
                      all_neighbors[i]=neighbors;
                    std::sort(neighbors.begin(),neighbors.end());
                    #pragma omp critical
                    {

                      adjlist[frontiers[i]]=neighbors;
                      vertex_sets.insert(neighbors.begin(),neighbors.end());
                    }
                    
                }
            }
        }
        else{
            #pragma omp parallel for
            for(size_t i=0;i<frontiers.size();i++){
                if(!(adjlist.find(frontiers[i])!=adjlist.end())){
                    auto neighbors=OutNeighbors(frontiers[i]);
                    if(hop!=1)
                      all_neighbors[i]=neighbors;
                    std::sort(neighbors.begin(),neighbors.end());
                    #pragma omp critical
                    {
                      adjlist[frontiers[i]]=neighbors;
                      vertex_sets.insert(neighbors.begin(),neighbors.end());
                    }
                }
            }
        }
        hop--;
        if(hop<=0) return ;
        std::vector<size_t> next_frontiers;
        for(auto & item:all_neighbors){
            next_frontiers.insert(next_frontiers.end(),item.begin(),item.end());
        }
      
      make_subgraph(hop,next_frontiers,adjlist,vertex_sets,is_in_mode);
  }
};