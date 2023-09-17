#include <dygstore/event.h>
namespace neutron{
    std::shared_ptr<SysEventType> SysEventTypePtr=nullptr;
    std::shared_ptr<SysEventType> GetInstance(){
        if(SysEventTypePtr==nullptr){
            SysEventTypePtr.reset(new (std::nothrow)SysEventType());
        }
        return  SysEventTypePtr;
    }
    std::unordered_map<std::string,EventType> SysEventType::hashEventType={
            {"Communication",Communication},
            {"AE",AE},
            {"AI",AI},
            {"AO",AO},
            {"DE",DE},
            {"DI",DI},
            {"DO",DO}
    };
} // namespace name
