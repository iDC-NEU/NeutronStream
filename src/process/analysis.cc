#include<process/engine/analysis.hpp>
#include <set>

#include <algorithm>
namespace neutron{

    void DependedGraph::graph2layers(){
        std::vector<size_t> rst;
        std::vector<size_t> vecs=GraphPtr[startId].get_deps();
        int nums=vecs.size();
        rst.resize(nums);
        for(int i=0;i<nums;i++){
            finished[vecs[i]-1-startId]=true;
            rst[i]=vecs[i]-1;
        }
        exe_layers.push_back(rst);
        while(!rst.empty()){
            std::set<size_t> next_layers;
            for(auto id:rst){
                std::vector<size_t> id_deps=GraphPtr[id+1].get_deps();
                next_layers.insert(id_deps.begin(),id_deps.end());
            }
            std::vector<size_t> next_rst;
            for(auto id:next_layers){
                std::vector<size_t> id_deped=GraphPtr[id].get_deped();
                if(!finished[id-1-startId] && std::all_of(id_deped.begin(),id_deped.end(),[&](size_t id){
                    return finished[id-1-startId];
                })){
                    next_rst.push_back(id-1);
                }
            }
            if(next_rst.empty()) return ;
            for(auto &fid:next_rst){
                finished[fid-startId]=true;
            }
            exe_layers.push_back(next_rst);
            rst=next_rst;
        }
    }

    std::vector<size_t> DependedGraph::awake_next_events(size_t eid) {
        int awake_id=static_cast<int>(eid+1);
        std::vector<size_t> rst;
        #pragma omp critical
        {
//                std::cout<<"awake id:"<<awake_id<<std::endl;
          for(auto &id:GraphPtr[awake_id].get_deps()){
              std::vector<size_t> id_deped=GraphPtr[id].get_deped();
              if(!finished[id-1-startId] && std::all_of(id_deped.begin(),id_deped.end(),[this](size_t did){
                  return finished[did-1-startId];
              })){
                  rst.push_back(id-1);
                  finished[id-1-startId]=true;
              }
          }
        }
        return rst;
    }
}