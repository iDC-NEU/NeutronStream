#ifndef NEUTRONSTREAM_EVENTBUFFER_H
#define NEUTRONSTREAM_EVENTBUFFER_H
#include <dygstore/event.h>
#include <vector>
#include <string>
#include <mutex>
#include <type_traits>
#include <unordered_set>
namespace neutron{

    class EventBuffer{
        public:
        typedef std::shared_ptr<EventBuffer> ptr;
        public:
            EventBuffer(const std::vector<Event> &evts);
            void InitEventBuffer(const std::vector<Event> &evts);
            bool is_end(){ return start_pos >=buffer.size();} 
            size_t GetBufferSize() { return buffer.size();}
            virtual std::vector<Event> nextWindowData() = 0;
            virtual ~EventBuffer(){}
        public:
            size_t start_pos;
            std::vector<Event> buffer;
            
    };

    class IncreaseWindow : public EventBuffer{
        public:
            IncreaseWindow(const std::vector<Event> &evt_list,double _init_rate=0.1,size_t increase_size=200);
            std::vector<Event> nextWindowData() override;
        private:
            double init_rate;
            size_t increase_size;
            size_t cur_length;
    };
    
    class SlideWindow : public EventBuffer{
        public:
            SlideWindow(const std::vector<Event> &evt_list,size_t _window_size,int _step);
            std::vector<Event> nextWindowData() override;
        private:
            size_t window_size; 
            int step;
    };


    class VaryWindow:public EventBuffer{
        public:
            VaryWindow(const std::vector<Event> &evt_list,size_t _down=200,size_t _up=1000,int step=-1);
            std::vector<Event> nextWindowData() override;
        private:
            size_t down;
            size_t up;
            int step;
    };

    class VaryWindowRate:public EventBuffer{
        public:
            VaryWindowRate(const std::vector<Event> &evt_list,size_t _down=200,size_t _up=1000,int _step=-1,float _rate_step=0.5);
            std::vector<Event> nextWindowData() override;
        private:
            size_t down;
            size_t up;
            int step;
            float rate_step;
        
    };
}
#endif