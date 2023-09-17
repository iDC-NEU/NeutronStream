#ifndef NEUTRONSTREAM_ANALYSIS_HPP
#define NEUTRONSTREAM_ANALYSIS_HPP
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <memory>
namespace neutron{
class DepGraphNode{
private:
    std::vector<size_t> deps;
    std::vector<size_t> deped;
    bool isSuperNode{false};

public:
    DepGraphNode()=default;
    ~DepGraphNode()=default;
    DepGraphNode(const DepGraphNode & depG){
        deps=depG.deps;
        deped=depG.deped;
        isSuperNode=depG.isSuperNode;
    }
    explicit DepGraphNode(bool is_spn):isSuperNode(is_spn){
    };
    bool isSuper() const{return  isSuperNode;}
    std::vector<size_t> get_deps() const {return deps;};
    std::vector<size_t> get_deped() const {return deped;}
    size_t deps_size () const {return deps.size();}
    size_t deped_size() const {return deped.size();}
    void add_deps(size_t id){deps.emplace_back(id);}
    void add_deped(size_t id){deped.emplace_back(id);}
};

class DependedGraph{
private:
    typedef std::unordered_map<size_t,DepGraphNode> DepGraphType;
    DepGraphType GraphPtr;
    size_t numEvent=0;
    std::vector<bool> finished;
    std::vector<size_t> layers;
    std::vector<std::vector<size_t>> exe_layers;
    std::atomic_size_t layer_id{0};
    std::size_t startId=0;
public:
    size_t numEvents() const {
        return numEvent;
    }
    size_t get_layers_num() const{
        return exe_layers.size();
    }
    explicit DependedGraph(int num_evts,size_t start_id):startId(start_id){
        numEvent=num_evts;
        GraphPtr[0]=DepGraphNode(true);
        layers.push_back(startId);
        GraphPtr.reserve(num_evts+1);
        finished.resize(numEvent);
    };
    DependedGraph(const DependedGraph &depG){
        startId=depG.startId;
        exe_layers=depG.exe_layers;
    }

    std::vector<size_t> awake_next_events(size_t eid);

    void graph2layers();

    std::vector<size_t> get_sources(){
        layer_id++;
        return exe_layers[0];
    }
    std::vector<size_t> awale_next_layers(){
        size_t cur_id=layer_id.load();
        if(cur_id<exe_layers.size()){
            layer_id++;
            return exe_layers[cur_id];
        }
        return {};
    }

    void addDep(size_t ei,size_t ej){
        GraphPtr[ei].add_deps(ej);
        GraphPtr[ej].add_deped(ei);
    }

    template<typename DySubG>
    static bool isDepended(const DySubG &esg1,const DySubG &esg2){
        if(esg1.get_event().time_point<esg2.get_event().time_point){
            auto uset=esg1.get_update_vertex();
            auto vset=esg2.get_v_set();
            std::unordered_set<size_t> intersection(uset.begin(),uset.end());
            for(auto &num:vset){
                if(intersection.find(num)!=intersection.end()) return true;
            }
            return false;
        }
        return false;
    }
    template<typename DySubG>
    static bool isDepended(std::shared_ptr<DySubG> &esg1,std::shared_ptr<DySubG> &esg2){
        if(esg1->get_event().time_point<esg2->get_event().time_point){
            auto uset=esg1->get_update_vertex();
            auto vset=esg2->get_v_set();
            std::unordered_set<size_t> intersection(uset.begin(),uset.end());
            for(auto &num:vset){
                if(intersection.find(num)!=intersection.end()) return true;
            }
            return false;
        }
        return false;
    }
    template<typename DySubG>
    void analysis(std::vector<std::shared_ptr<DySubG>> esgList,size_t check_id){
        check_id=check_id+1;
        std::vector<bool> visit(check_id-startId, false);
        bool flag= false;
        std::queue<size_t> taskQ;
        for(auto itr=layers.begin();itr!=layers.end();){
            if(visit[*itr-startId]) continue;
            visit[*itr-startId]=true;
//                std::cout<<"analysis : "<< *itr-1 << " "<<check_id-1<<std::endl;
            if(GraphPtr[*itr].isSuper() || isDepended(esgList[*itr-1],esgList[check_id-1])){
                addDep(*itr,check_id);
                itr=layers.erase(itr);
                flag= true;
            }
            else{
                if(!flag){
                    for(auto id:GraphPtr[*itr].get_deped()){
                        if(id!=startId) taskQ.push(id);
                    }
                }
                itr++;
            }
        }
        while(!flag &&!taskQ.empty()){
            size_t size=taskQ.size();
            while(size--){
                size_t curId=taskQ.front();
                taskQ.pop();
                if(visit[curId-startId]) continue;
                visit[curId-startId]= true;
//                    std::cout<<"analysis : "<< curId-1 << " "<<check_id-1<<std::endl;
                if(isDepended(esgList[curId-1],esgList[check_id-1])){
                    addDep(curId,check_id);
                    flag= true;
                }
                else{
                    if(!flag){
                        for(auto id:GraphPtr[curId].get_deped()){
                            if(id!=startId) taskQ.push(id);
                        }
                    }
                }
            }
        }
        if(!flag) addDep(startId,check_id);
        layers.push_back(check_id);
    }

    template<typename DySubG>
    void analysis(const std::vector<DySubG> &esgList,size_t check_id){
        check_id=check_id+1;
        std::vector<bool> visit(check_id-startId, false);
        bool flag= false;
        std::queue<size_t> taskQ;
        for(auto itr=layers.begin();itr!=layers.end();){
            if(visit[*itr-startId]) continue;
            visit[*itr-startId]=true;
//                std::cout<<"analysis : "<< *itr-1 << " "<<check_id-1<<std::endl;
            if(GraphPtr[*itr].isSuper() || isDepended(esgList[*itr-1],esgList[check_id-1])){
                addDep(*itr,check_id);
                itr=layers.erase(itr);
                flag= true;
            }
            else{
                if(!flag){
                    for(auto id:GraphPtr[*itr].get_deped()){
                        if(id!=startId) taskQ.push(id);
                    }
                }
                itr++;
            }
        }
        while(!flag &&!taskQ.empty()){
            size_t size=taskQ.size();
            while(size--){
                size_t curId=taskQ.front();
                taskQ.pop();
                if(visit[curId-startId]) continue;
                visit[curId-startId]= true;
//                    std::cout<<"analysis : "<< curId-1 << " "<<check_id-1<<std::endl;
                if(isDepended(esgList[curId-1],esgList[check_id-1])){
                    addDep(curId,check_id);
                    flag= true;
                }
                else{
                    if(!flag){
                        for(auto id:GraphPtr[curId].get_deped()){
                            if(id!=startId) taskQ.push(id);
                        }
                    }
                }
            }
        }
        if(!flag) addDep(startId,check_id);
        layers.push_back(check_id);
    }
};
} // namespace neutron
#endif