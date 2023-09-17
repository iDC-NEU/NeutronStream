#ifndef NEUTRON_SAVE_STATE_H
#define NEUTRON_SAVE_STATE_H
#include <dygstore/dygraph.hpp>
#include <dygstore/embedding.hpp>
#include <session.h>
namespace neutron
{
  template<typename EdgeDataType=EdgeData,typename EMBType=torch::Tensor,size_t SAVET=2>
  void save_window_initial(){
    if(Session::GetInstance()->GetSlideMode()=="other"){
      return;
    }
    // std::cout<< "save_window_initial" <<std::endl;
    auto dyg=DiDyGraph<EdgeDataType>::GetInstance();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto timebarptr=TimeBar::GetInstance();
    dyg->SaveWindowInitial();
    embptr->SaveWindowInitial();
    timebarptr->SaveWindowInitial();
    std::cout<< "save_window_initial" <<std::endl;
  }
  template<typename EdgeDataType=EdgeData,typename EMBType=torch::Tensor,size_t SAVET=2>
  void reset_window_initial(){
    if(Session::GetInstance()->GetSlideMode()=="other"){
      return;
    }
    auto dyg=DiDyGraph<EdgeDataType>::GetInstance();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto timebarptr=TimeBar::GetInstance();
    dyg->ResetWindowInitial();
    embptr->ResetWindowInitial();
    timebarptr->ResetWindowInitial();
    std::cout<< "reset_window_initial" <<std::endl;
  }
  template<typename EdgeDataType=EdgeData,typename EMBType=torch::Tensor,size_t SAVET=2>
  void save_window_step(){
    if(Session::GetInstance()->GetSlideMode()=="other"){
      return;
    }
    if(Session::GetInstance()->GetStep()<0 || !Session::GetInstance()->GetIsLastEpoch()){
      return;
    }
    Session::GetInstance()->AddCurStep();
    if(Session::GetInstance()->IsCurStep()){
      auto dyg=DiDyGraph<EdgeDataType>::GetInstance();
      auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
      auto timebarptr=TimeBar::GetInstance();
      dyg->SaveWindowStep();
      embptr->SaveWindowStep();
      timebarptr->SaveWindowStep();
      std::cout<< "save_window_step" <<std::endl;
    }
    
  }
  
  template<typename EdgeDataType=EdgeData,typename EMBType=torch::Tensor,size_t SAVET=2>
  void reset_window_step(){
    if(Session::GetInstance()->GetSlideMode()=="other"){
      return;
    }
    if(Session::GetInstance()->GetStep()<0){
      return ;
    }
    auto dyg=DiDyGraph<EdgeDataType>::GetInstance();
    auto embptr=DyNodeEmbedding<EMBType,SAVET>::GetInstanceThreadSafe();
    auto timebarptr=TimeBar::GetInstance();
    // dyg->SaveWindowStep();
    dyg->ResetWindowStep();
    embptr->SaveWindowStep();
    timebarptr->SaveWindowStep();
    dyg->get_information();
    
    // embptr->ResetWindowStep();
    // timebarptr->ResetWindowStep();
    Session::GetInstance()->ClearCurStep();
    Session::GetInstance()->SetIsLastEpoch(false);
    std::cout<< "reset_window_step" <<std::endl;
  }
  
} // namespace neutron

#endif