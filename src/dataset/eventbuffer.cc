#include <dataset/eventbuffer.h>
#include <session.h>
namespace neutron{
    
    EventBuffer::EventBuffer(const std::vector<Event> &evts):start_pos(0){
        buffer.resize(evts.size());
        std::copy(evts.begin(), evts.end(), buffer.begin());
    }

    IncreaseWindow::IncreaseWindow(const std::vector<Event> &evt_list,double _init_rate,size_t _increase_size):
    EventBuffer(evt_list),init_rate(_init_rate),increase_size(_increase_size){
        cur_length = static_cast<size_t>(evt_list.size()*init_rate);
    }

    std::vector<Event> IncreaseWindow::nextWindowData(){
        double window_time = cpuSecond();
        if(cur_length> buffer.size()){
            return std::vector<Event>();
        }
        size_t cur_size =cur_length;
        cur_length += increase_size;
        std::cout<<"---------------------Increase Window mode,init_rate:"<<
        init_rate<<" ("<< start_pos << "-" << cur_size<<")/"<<GetBufferSize()<<"------------"<<std::endl;
        auto rst= std::vector<Event>(buffer.begin(),buffer.begin()+cur_size);
        Session::GetInstance()->addTimer("window_time:",cpuSecond()-window_time);
        return rst;

    }
    
    //SlideWindow 构造函数
    // _step 表示滑动距离
    // _window_size 表示 窗口大小
    SlideWindow::SlideWindow(const std::vector<Event> &evt_list,size_t _window_size,int _step):
    EventBuffer(evt_list),window_size(_window_size),step(_step){
    }
    // 得到下一个窗口的数据
    std::vector<Event> SlideWindow::nextWindowData(){
        double window_time = cpuSecond();
        if(is_end()){
            return std::vector<Event>();
        }
        std::vector<Event>::iterator begin=buffer.begin()+start_pos;
        std::vector<Event>::iterator end;
        size_t end_pos=0;
        if(start_pos+window_size>=GetBufferSize()){
            end=buffer.end();
            end_pos=GetBufferSize();
            std::cout<<"---------------------slide window mode,step:"<< step <<" train data: (" 
            << start_pos << "-"<<end_pos<<")/" << GetBufferSize()
            <<"---------------------"<<std::endl;
            start_pos=end_pos;  //add 
        }
        else{
            end=begin+window_size;
            end_pos=start_pos+window_size;
            std::cout<<"---------------------slide window mode,step:"<< step <<" train data: (" 
            << start_pos << "-"<<end_pos<<")/" << GetBufferSize()
            <<"---------------------"<<std::endl;
            start_pos += step;
        }
        //slide window
        
        
        auto rst= std::vector<Event>(begin,end);
        Session::GetInstance()->addTimer("window_time:",cpuSecond()-window_time);
        return rst;
    }

    //_down 表示 下限
    // _up 表示 上限 
    VaryWindow::VaryWindow(const std::vector<Event> &evt_list,size_t _down,size_t _up,int _step)
    :EventBuffer(evt_list),down(_down),up(_up),step(_step){}
    

    std::vector<Event> VaryWindow::nextWindowData(){
        double window_time = cpuSecond();
        if(is_end()){
            return std::vector<Event>();
        }
        std::unordered_set<size_t> nodes;
        std::vector<Event> vary_windows;
        size_t src_start_pos = start_pos;
        for(;src_start_pos<buffer.size();src_start_pos++){
            auto evt=buffer[src_start_pos];
            if(vary_windows.size()>up){
                break;
            }else if(vary_windows.size()<down){
                vary_windows.push_back(evt);
                nodes.insert(evt.src_id);
                nodes.insert(evt.dst_id);
            }else if((nodes.find(evt.src_id)!=nodes.end()) || (nodes.find(evt.dst_id)!=nodes.end())){
                vary_windows.push_back(evt);
            }else{
                break;
            }
        }
        
        std::cout<<"---------------------vary window mode,up:"<< up
        <<",down:"<<down <<" train data: (" << start_pos << "-"<< src_start_pos<<")/"<< buffer.size() 
        <<"---------------------"<<std::endl;
        if(step>0){
            start_pos += step;
        }else{
            start_pos = src_start_pos;
        }
        Session::GetInstance()->addTimer("window_time:",cpuSecond()-window_time);

        return vary_windows;
    }

    VaryWindowRate::VaryWindowRate(const std::vector<Event> &evt_list,size_t _down,size_t _up,int _step,float _rate_step)
    :EventBuffer(evt_list),down(_down),up(_up),step(_step),rate_step(_rate_step){}
    
    std::vector<Event> VaryWindowRate::nextWindowData(){
        double window_time = cpuSecond();
        if(is_end()){
            return std::vector<Event>();
        }
        std::unordered_set<size_t> nodes;
        std::vector<Event> vary_windows;
        size_t dst_start_pos= start_pos;
        size_t src_start_pos = start_pos;
        for(;dst_start_pos<buffer.size();dst_start_pos++){
            auto evt=buffer[dst_start_pos];
            if(vary_windows.size()>up){
                break;
            }else if(vary_windows.size()<down){
                vary_windows.push_back(evt);
                nodes.insert(evt.src_id);
                nodes.insert(evt.dst_id);
            }else if((nodes.find(evt.src_id)!=nodes.end()) || (nodes.find(evt.dst_id)!=nodes.end())){
                vary_windows.push_back(evt);
            }else{
                break;
            }
        }
        if(step>0){
            size_t cur_len = dst_start_pos-start_pos;
            step = static_cast<int>(cur_len*rate_step);
            Session::GetInstance()->SetStep(step);
            start_pos += step;

        }else{
            start_pos = dst_start_pos;
        }
        if(dst_start_pos>=buffer.size()-1){
            start_pos = dst_start_pos + 1;
        }
        std::cout<<"---------------------vary window mode,up:"<< up
        <<",down:"<<down<<"  stepRate:"<<rate_step <<" next step:"<<step <<" train data: (" << src_start_pos << "-"<< dst_start_pos<<")/"<< buffer.size() 
        <<"---------------------"<<std::endl;
        Session::GetInstance()->addTimer("window_time:",cpuSecond()-window_time);

        return vary_windows;
    }
}