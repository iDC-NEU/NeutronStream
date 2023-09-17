
#ifndef NEUTRONSTREAM_EVENT_H
#define NEUTRONSTREAM_EVENT_H
#include <memory>
#include <unordered_map>
#include <string>
#include <utils/mytime.h>
namespace neutron{
  
  enum EventType {
      Communication=0,
      AE,
      AI,
      AO,
      DE,
      DI,
      DO,
      OTHER
  };
class SysEventType{
  private:
      static std::shared_ptr<SysEventType> SysEventTypePtr;
  public:
      static std::shared_ptr<SysEventType> GetInstance();
      static std::unordered_map<std::string,EventType> hashEventType;
      static EventType strToEventType(std::string str){
          if(hashEventType.find(str)!=hashEventType.end()){
              return hashEventType[str];
          }
          return OTHER;
      }
      static std::string evtType2Str(EventType type){
          switch (type) {
              case Communication:return "Communication";break;
              case AE: return "AddEdge";break;
              case AI: return "AddInNeighbor";break;
              case AO: return "AddOutNeighbor";break;
              case DE: return "DeleteEdge";break;
              case DI: return "DeleteInNeighbor";break;
              case DO: return "DeleteOutNeighbor";break;
              default:break;
          }
          return "Other";
      }
};
class Event{
    public:
        size_t src_id;
        size_t dst_id;
        TimePoint time_point;
        EventType eventType;
    Event()=default;
    Event(const Event &evt){
        src_id=evt.src_id;
        dst_id=evt.dst_id;
        time_point=evt.time_point;
        eventType=evt.eventType;
    }
    Event(size_t _src_id,size_t _dst_id,const TimePoint &time,EventType evtType)
        :src_id(_src_id),dst_id(_dst_id),time_point(time),eventType(evtType){}
    Event(size_t _src_id,size_t _dst_id,const TimePoint &time,const std::string &evtStr)
    :src_id(_src_id),dst_id(_dst_id),time_point(time){
        eventType= SysEventType::strToEventType(evtStr);
    };
    bool operator ==(const Event &evt)const{
        return evt.src_id==src_id &&evt.dst_id==dst_id;
    }
    
    Event& operator =(const Event &other){
        if(this == &other){
            return *this;
        }
        src_id=other.src_id;
        dst_id=other.dst_id;
        time_point=other.time_point;
        eventType=other.eventType;
        return *this;
    }

    bool operator()(const Event &evt) const {
        return *this==evt;
    }

    bool operator< (const Event &evt)const{
        return src_id<evt.src_id || (src_id==evt.src_id &&dst_id<evt.dst_id);
    }
    bool operator> (const Event &evt)const{
        return src_id>evt.src_id || (src_id==evt.src_id &&dst_id>evt.dst_id);
    }
    std::string toString() {
        return std::to_string(src_id)+","+std::to_string(dst_id)+","+time_point.toString()+","+ SysEventType::evtType2Str(eventType);
    }

};
    // std::ostream& operator<<(std::ostream &out,Event evt){
    //     out<<evt.toString();
    //     return out;
    // }
    struct EventHashFun{
        size_t operator()(const Event &evt) const{
            size_t src_hash_code=evt.src_id<<8;
            size_t dst_hash_code=evt.dst_id<<4;
            return std::hash<size_t>()(src_hash_code)+std::hash<size_t>()(dst_hash_code);
        }
    };
}

#endif