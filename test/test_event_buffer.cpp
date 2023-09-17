#include <dataset/eventbuffer.h>
#include <dataset/dataset.h>
#include <thread>

namespace neutron{

    void test_slide(){
        int EnventCount =1100;
        std::vector<Event> evtlists;
        for(int i=0; i<EnventCount;i++){
            evtlists.emplace_back(i,i,1000000000,AE);
        }
        // std::string name = "social";
        // std::string datadir = "./data/" + name + "/";
        // Dataset dataset(name,datadir);

        std::shared_ptr<EventBuffer> buffer(new SlideWindow(evtlists,300,100));

        std::vector<Event> data= buffer->nextWindowData();
        
        while(!data.empty()){

            for(auto &evt:data){
                std::cout<<evt.toString()<<std::endl;
            }
            data=buffer->nextWindowData();
            sleep(2);
        }
    }
    void test_vary_window(){
        int EnventCount =3100;
        std::vector<Event> evtlists;
        for(int i=0; i<EnventCount;i++){
            evtlists.emplace_back(i,i,1000000000,AE);
        }

        // std::string name = "social";
        // std::string datadir = "./data/" + name + "/";
        // Dataset dataset(name,datadir);
        std::shared_ptr<EventBuffer> buffer(new VaryWindow(evtlists,400,1000,200));

        std::vector<Event> data= buffer->nextWindowData();
        
        while(!data.empty()){

            for(auto &evt:data){
                std::cout<<evt.toString()<<std::endl;
            }
            
            data=buffer->nextWindowData();
            sleep(2);
        }
    }
    void test_increase_window(){
        int EnventCount =3100;
        std::vector<Event> evtlists;
        for(int i=0; i<EnventCount;i++){
            evtlists.emplace_back(i,i,1000000000,AE);
        }
        std::shared_ptr<EventBuffer> buffer(new IncreaseWindow(evtlists,0.1,200));
        std::vector<Event> data= buffer->nextWindowData();
        while(!data.empty()){

            for(auto &evt:data){
                std::cout<<evt.toString()<<std::endl;
            }
            data=buffer->nextWindowData();
            sleep(2);
        }
    }
    
}
int main(){
    // neutron::test_mode();
    // neutron::test_init();
    neutron::test_slide();
    neutron::test_vary_window();
    neutron::test_increase_window();
}