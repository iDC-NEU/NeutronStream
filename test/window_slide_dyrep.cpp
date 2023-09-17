#include <iostream>
#include <log/log.h>
#include <NumCpp.hpp>
#include <dataset/dataset.h>
#include <dygstore/interface.hpp>
#include <dataset/eventbuffer.h>
#include <dygstore/save_state.hpp>
#include <model/dyrep.h>

namespace neutron{
std::tuple<float,float> test(DyRep &model,Dataset &dataset){
    model.eval();
    torch::NoGradGuard no_grad;
    size_t test_batch_num= dataset.getBatchNum();
    std::vector<float> losses;
    for(size_t bidx=0; bidx<test_batch_num;bidx++){
      std::vector<Event> data=dataset.getIdxBatch(bidx);
      auto  output = model.forward(data);
      torch::Tensor loss1= std::get<0>(output);
      torch::Tensor loss2 = std::get<1>(output);
      torch::Tensor loss = (loss1 + loss2) / static_cast<int>(data.size());
      losses.push_back(loss.item().toFloat());
    }
    float loss_sum = std::accumulate(losses.begin(),losses.end(),0.0f);
    return {loss_sum/losses.size(),0.8};
}

void window_slide_train(){
    if(Session::GetInstance()->GetSlideMode()=="other"){
        std::cout << "no set slide_mode:{slide,vary,increase}"<<std::endl;
        return ;
    }
    auto sessionptr=Session::GetInstance();
    torch::manual_seed(0);
    torch::cuda::manual_seed(0);
    nc::random::seed(0);
    size_t Epochs = sessionptr->GetEpoch();
    std::string name = sessionptr->GetDataSet();
    int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();

    std::cout << "dataset_name:" << name << std::endl;
    std::cout <<"Epochs:"<<Epochs<<std::endl;
    std::cout<<"Hidden_dim:"<<HIDDEN_DIM<<std::endl;
    std::string datadir = "./data/" + name + "/";
    std::cout << "datadir:" << datadir << std::endl;
    bool is_temporal=Session::GetInstance()->GetIsTemporal();
    Dataset dataset(name, datadir,is_temporal);
    std::cout<<"---prepare dataset finished ---"<<std::endl;


    size_t NUM_NODES = dataset.getNumVertices();
    auto device = sessionptr->GetDevice();
    std::cout<<"device:"<<device << std::endl;

    auto model = DyRep(NUM_NODES,
                        HIDDEN_DIM);

    double lr = 0.0002;
    torch::optim::AdamOptions adamOptions(lr);
    adamOptions.betas(std::make_tuple(0.5, 0.999));
//        std::cout<<"model.parameters():"<<model.parameters()<<std::endl;
    torch::optim::Adam optimizer(model.parameters(), adamOptions);
    torch::optim::StepLR stepLr(optimizer, 10, 0.5);

    model.to(device);

            //define window
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
    bool is_Slide_Mode = false;
    size_t splittmp = 0;
    if(slide_mode == "slide"){
      splittmp = sessionptr->GetWindowSize()-step;
      is_Slide_Mode = true;
    }
    size_t split = 0;
    while(!evts.empty()){
      if(is_Slide_Mode){
        // split= evts.size() > splittmp ? static_cast<size_t>(sessionptr->GetWindowSize()-step) : static_cast<size_t>(evts.size()*0.8);
        split = evts.size() > 200 ? 200 : static_cast<size_t>(evts.size()*0.8);
      }else{
        split=static_cast<size_t>(evts.size()*0.8);
      }
      std::vector<Event> train_evts(evts.begin(), evts.begin()+split);
      std::vector<Event> test_evts(evts.begin()+split,evts.end());
      Dataset train_dataset(train_evts);
      Dataset test_dataset(test_evts);
      if(slide_mode!="increase"){
        reset_window_step();
      }
      float max_auc = 0;
      save_window_initial();
      size_t train_batch_num =train_dataset.getBatchNum();
      for(size_t epoch=0;epoch<Epochs;epoch++){
        model.train();
        if(epoch==Epochs-1){
          Session::GetInstance()->SetIsLastEpoch(true);
        }
        std::vector<float> accs;
        double train_start =cpuSecond();
        for(size_t bidx=0;bidx<train_batch_num;bidx++){
          optimizer.zero_grad();
          std::cout<<"Train: batch "<<bidx+1<<"/"<<train_batch_num<<" ";
          std::vector<Event> data=train_dataset.getIdxBatch(bidx);
          auto output = model.forward(data);
          torch::Tensor loss1 = std::get<0>(output);
          torch::Tensor loss2 = std::get<1>(output);
          torch::Tensor loss = (loss1 + loss2) / static_cast<int>(data.size());
          std::cout<<"loss:"<<loss.item().toFloat()<<" "<<std::endl;
          loss.backward();
          torch::nn::utils::clip_grad_value_(model.parameters(),100);
          optimizer.step();
          merge();
      }

      sessionptr->addTimer("train",cpuSecond()-train_start);

      auto rst=test(model,test_dataset);
      float test_loss = std::get<0>(rst);
      float test_auc = std::get<1>(rst);
      max_auc = std::max(max_auc,test_auc);
      std::cout<<"Test: test loss:"<<test_loss<<" test_auc:"<<test_auc<<std::endl;
      reset_window_initial();
    }
    evts=slide_window->nextWindowData();
  }
  sessionptr->printAllTimers();
}

}
int main(int argc ,char *argv[]){
    std::cout<<"window slide train"<<std::endl;
    // check torch version
    // std::cout<<"Torch versionL"<<std::endl<<TORCH_VERSION_MAJOR<<"."<<TORCH_VERSION_MINOR<<"."<<TORCH_VERSION_PATCH<<std::endl;

    neutron::Session::Init(argc,argv);
    neutron::window_slide_train();
    return 0;
}
