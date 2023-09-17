#include <model/debug/bug_dygnn.h>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <session.h>
#include <NumCpp.hpp>
#include <dataset/dataset.h>
#include <utils/mytime.h>
#include <dygstore/interface.hpp>
#include <utils/acc_utils.h>
namespace neutron{
  void train(){
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

    Dataset dataset(name, datadir,{5,HIDDEN_DIM},true);

    size_t batch_num = dataset.getBatchNum();
    size_t NUM_NODES = dataset.getNumVertices();
    auto device = sessionptr->GetDevice();
    std::cout<<"device:"<<device << std::endl;

// 开启正向传播时的梯度探测
    // torch::autograd::AnomalyMode::set_enabled(false);
    
    debug::DyGNN model(NUM_NODES,HIDDEN_DIM,HIDDEN_DIM);
    model.to(device);
    model.train();
    torch::optim::AdamOptions adamOptions(0.001);
    adamOptions.set_lr(0.0001);
    adamOptions.weight_decay(0.001);
    for(auto &p:model.parameters()){
      p.set_requires_grad(true);
    }
    auto optimizer = torch::optim::Adam(model.parameters(),adamOptions);
    
    for(size_t epoch=0;epoch<Epochs;epoch++){
      sessionptr->SetCurEpoch(epoch);
      Timing epoch_time("epoch_"+std::to_string(epoch));
      std::cout<<"epoch:"<<epoch<<std::endl;

      for(size_t bidx=0;bidx<batch_num;bidx++){
        std::cout<<"batch "<<bidx+1<<"/"<<batch_num<<" ";
        sessionptr->SetCurBatchIdx(bidx);
        Timing batch_time("batch");
        std::vector<Event> data=dataset.getIdxBatch(bidx);
        double ft=cpuSecond();
        // printf("fuck t6b\n");
        auto output=model.forward(data);
        torch::Tensor loss =std::get<0>(output);
        std::cout<<"loss:"<<loss.item().toFloat()<<" ";
        sessionptr->addTimer("forward",cpuSecond()-ft);
        double bt=cpuSecond();
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        sessionptr->addTimer("backward",cpuSecond()-bt);
        batch_time.end();
        merge();
      }
      
      epoch_time.end();
      BackToInitial();
      sessionptr->printAllTimers(true);
    }
  }
    
} // namespace neutron
int main(int argc, char *argv[])
{
    std::cout<<"debug dygnn"<<std::endl;
    neutron::Session::Init(argc,argv);
    neutron::train();
    return 0;
}
