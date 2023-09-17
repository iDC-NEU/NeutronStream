#include <memory>
#include <session.h>
#include <process/dysubg.h>
#include <process/engine/analysis.hpp>
#include <dataset/dataset.h>
#include <dygstore/interface.hpp>
#include <queue>
#include <dataset/eventbuffer.h>
#include <model/dygnn.h>

namespace neutron{
    static std::vector<float> windows_rate;
    void window_analysis(std::vector<Event> &evtList){
        static size_t cutNum = 50;
        size_t num_layers = 0 ;
        size_t num_events = evtList.size();
        
        std::vector<std::shared_ptr<DySubGraph>> subgslist(num_events);
        std::queue<DependedGraph> depGtask;
        size_t numsCutsStage=0;
        if(num_events%cutNum==0){
            numsCutsStage=num_events/cutNum;
        }
        else{
            numsCutsStage=num_events/cutNum+1;
        }
        for(size_t i=0;i<numsCutsStage;i++) {
            int start_id = i * cutNum;
            DependedGraph depG(std::min(static_cast<int>(num_events),
                                            static_cast<int>(start_id + cutNum)) - start_id,
                                   start_id);
            for (size_t e_ids = start_id; e_ids < num_events && e_ids < (i + 1) * cutNum; e_ids++) {
                auto evt =evtList[e_ids];
                if(Session::GetInstance()->GetIsTemporal()){
                    // std::cout << "is Temporal" << std::endl;
                    AddEdge(evt.src_id, evt.dst_id);
                    AddEdge(evt.dst_id, evt.src_id);
                }
                else if(evt.eventType==AE){
                    AddEdge(evt.src_id, evt.dst_id);
                    AddEdge(evt.dst_id, evt.src_id);
                }
                subgslist[e_ids]=std::make_shared<DySubGraph>(evt);
                depG.analysis<DySubGraph>(subgslist, e_ids);
            }
            depG.graph2layers();
            depGtask.push(depG);
        }
        while (!depGtask.empty()){
            auto depG = depGtask.front();
            depGtask.pop();
            num_layers += depG.get_layers_num();
        }
        float rate = static_cast<float>(num_events)/static_cast<float>(num_layers);
        windows_rate.push_back(rate);
    }

    void window_analysis_dygnn(std::vector<Event> &evtList){
        static size_t cutNum = 50;
        size_t num_layers = 0 ;
        size_t num_events = evtList.size();
        
        std::vector<std::shared_ptr<DyGNNSubG>> subgslist(num_events);
        std::queue<DependedGraph> depGtask;
        size_t numsCutsStage=0;
        if(num_events%cutNum==0){
            numsCutsStage=num_events/cutNum;
        }
        else{
            numsCutsStage=num_events/cutNum+1;
        }
        for(size_t i=0;i<numsCutsStage;i++) {
            int start_id = i * cutNum;
            DependedGraph depG(std::min(static_cast<int>(num_events),
                                            static_cast<int>(start_id + cutNum)) - start_id,
                                   start_id);
            for (size_t e_ids = start_id; e_ids < num_events && e_ids < (i + 1) * cutNum; e_ids++) {
                auto evt =evtList[e_ids];
            
                AddEdge(evt.src_id, evt.dst_id);
                subgslist[e_ids]=std::make_shared<DyGNNSubG>(evt);
                depG.analysis<DyGNNSubG>(subgslist, e_ids);
            }
            depG.graph2layers();
            depGtask.push(depG);
        }
        while (!depGtask.empty()){
            auto depG = depGtask.front();
            depGtask.pop();
            num_layers += depG.get_layers_num();
        }
        float rate = static_cast<float>(num_events)/static_cast<float>(num_layers);
        windows_rate.push_back(rate);
    }


    void data_analysis(){
        auto sessionptr=Session::GetInstance();
        std::string name = sessionptr->GetDataSet();
        std::cout << "dataset_name:" << name << std::endl;
        
        std::string datadir = "./data/" + name + "/";
        std::cout << "datadir:" << datadir << std::endl;

        bool is_temporal=Session::GetInstance()->GetIsTemporal();

        Dataset dataset(name, datadir,is_temporal);
        auto evtList = dataset.getEventlist();
        size_t numEvents=evtList.size();


        auto evtlist =dataset.getEventlist();
        int step=sessionptr->GetStep();
        EventBuffer::ptr slide_window = nullptr;
        std::string slide_mode = sessionptr->GetSlideMode();
        if(slide_mode == "slide"){
        size_t win_size=sessionptr->GetWindowSize();
        std::cout << "window size:"<<win_size << std::endl;
        if(step==-1){
            slide_window.reset(new SlideWindow(evtlist,win_size,static_cast<int>(win_size)));
        }else{
            slide_window.reset(new SlideWindow(evtlist,win_size,step));
        }
        }else if(slide_mode == "vary"){
        size_t up = sessionptr->GetUp();
        size_t down = sessionptr->GetDown();
        float stepRate = sessionptr->GetStepRate();
        std::cout << "step rate"<<stepRate<<std::endl;
        if(stepRate<0){
            slide_window.reset(new VaryWindow(evtlist,down,up,step));
        }else{
            slide_window.reset(new VaryWindowRate(evtlist,down,up,step,stepRate));
        }
            
        }else if(slide_mode == "increase"){
        size_t increase_size = sessionptr->GetIncreaseSize();
        slide_window.reset(new IncreaseWindow(evtlist,0.1,increase_size));
        }
        // 滑动窗口 
        //记录开始时间
        std::vector<Event> evts= slide_window->nextWindowData();
        while(!evts.empty()){
            evts=slide_window->nextWindowData();
            size_t split=static_cast<size_t>(evts.size()*0.8);
            std::vector<Event> train_evts(evts.begin(), evts.begin()+split);
            std::vector<Event> test_evts(evts.begin()+split,evts.end());
            Dataset train_dataset(train_evts);
            Dataset test_dataset(test_evts);
            size_t train_batch_num =train_dataset.getBatchNum();
            for(size_t bidx=0;bidx<train_batch_num;bidx++){
                std::vector<Event> data=dataset.getIdxBatch(bidx);
                window_analysis_dygnn(data);
            }
        }
        std::cout<<"avg score:"<<std::accumulate(windows_rate.begin(), windows_rate.end(),0.0f)/windows_rate.size()<<std::endl;
    }
}

int main(int argc, char **argv){
    std::cout << " dataset analysis"<<std::endl;
    neutron::Session::Init(argc,argv);
    neutron::data_analysis();
}