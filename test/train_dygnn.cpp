#include <iostream>
#include <torch/torch.h>
#include <string>
#include <session.h>
#include <NumCpp.hpp>
#include <dataset/dataset.h>
#include <model/dygnn.h>
#include <utils/mytime.h>
#include <dygstore/interface.hpp>
#include <utils/acc_utils.h>
namespace F = torch::nn::functional;
namespace neutron{
  torch::Tensor get_valid_loss(const std::vector<Event> &valid_datas){
    std::vector<size_t> head_list;
    std::vector<size_t> tail_list;
    for(size_t i=0; i<valid_datas.size(); i++){
      head_list.push_back(valid_datas[i].src_id);
      tail_list.push_back(valid_datas[i].dst_id);
    }
    std::vector<int> label_vec(head_list.size(), 1);
    torch::Tensor head_tensors =F::normalize(index(head_list,4));
    
    torch::Tensor tail_tensors =F::normalize(index(tail_list,4));

    torch::Tensor scores = torch::bmm(head_tensors.view({static_cast<int64_t>(head_list.size()), 1, head_tensors.size(1)}),
                       tail_tensors.view({static_cast<int64_t>(tail_list.size()), head_tensors.size(1), 1})).view({static_cast<int64_t>(head_list.size())});
    torch::Tensor labels =torch::tensor(label_vec).to(torch::kFloat);
    auto loss = torch::binary_cross_entropy_with_logits(scores, labels);
    return loss;
  }

  void train(){
    auto sessionptr=Session::GetInstance();
    torch::manual_seed(0);
    torch::cuda::manual_seed(0);
    nc::random::seed(0);
    size_t Epochs = sessionptr->GetEpoch();
    std::string name = sessionptr->GetDataSet();
    int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
    double trainRatio = Session::GetInstance()->GetTrainRatio();
    // double validRatio = Session::GetInstance()->GetValidRatio();

    std::cout << "dataset_name:" << name << std::endl;
    std::cout <<"Epochs:"<<Epochs<<std::endl;
    std::cout<<"Hidden_dim:"<<HIDDEN_DIM<<std::endl;
    std::string datadir = "./data/" + name + "/";
    std::cout << "datadir:" << datadir << std::endl;

    Dataset dataset(name, datadir,{5,HIDDEN_DIM},true);
    size_t batch_num = dataset.getBatchNum();

    size_t eventSize = dataset.getNumEvents();
    size_t train_eventSize = eventSize*trainRatio;

    // size_t train_batch_num = (train_eventSize+Session::GetInstance()->GetBatchSize()-1)
    //                           /Session::GetInstance()->GetBatchSize();
    // size_t valid_batch_num = (valid_eventSize+Session::GetInstance()->GetBatchSize()-1)
    //                           /Session::GetInstance()->GetBatchSize();    


    size_t NUM_NODES = dataset.getNumVertices();
    auto device = sessionptr->GetDevice();
    std::cout<<"device:"<<device << std::endl;

// 开启正向传播时的梯度探测
    // torch::autograd::AnomalyMode::set_enabled(false);
    
    DyGNN model = DyGNN(NUM_NODES,HIDDEN_DIM,HIDDEN_DIM);
    model.to(device);
    
    torch::optim::AdamOptions adamOptions(0.001);
    adamOptions.set_lr(0.001);
    adamOptions.weight_decay(0.001);
    for(auto &p:model.parameters()){
      p.set_requires_grad(true);
    }
    auto optimizer = torch::optim::Adam(model.parameters(),adamOptions);
    
    for(size_t epoch=0;epoch<Epochs;epoch++){
      sessionptr->SetCurEpoch(epoch);
      Timing epoch_time("epoch_"+std::to_string(epoch));
      std::cout<<"epoch:"<<epoch<<std::endl;
      model.train();
      for(size_t bidx=0;bidx<batch_num;bidx++){
        std::cout<<"batch "<<bidx+1<<"/"<<batch_num<<" ";
        sessionptr->SetCurBatchIdx(bidx);
        Timing batch_time("batch");
        std::vector<Event> data=dataset.getIdxBatch(bidx);
        // std::vector<Event> data=dataset.getTrainIdxBatch(bidx,train_eventSize);
        double ft=cpuSecond();
        // printf("fuck t6b\n");
        auto output = model.forward(data);
        torch::Tensor loss=std::get<0>(output);
        
        std::cout<<"loss:"<<loss.item().toFloat()<<" ";
        sessionptr->addTimer("forward",cpuSecond()-ft);
        double bt=cpuSecond();
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        sessionptr->addTimer("backward",cpuSecond()-bt);
        batch_time.end();
        // BackToInitial();
        merge();
      }
      epoch_time.end();
      sessionptr->printAllTimers(true);
      // model.eval();
      // for(size_t valid_idx=0;valid_idx<valid_batch_num;valid_idx++){
      //     std::vector<Event> 
      //     validdata = dataset.getValidIdxBatch(valid_idx,train_eventSize);
      //     torch::Tensor loss=model.forward(validdata);
      //     std::cout<<"valid batch "<<valid_idx+1<<"/"<<valid_batch_num<<" loss:"<<loss.item().toFloat()<<" "<<std::endl;
      //     //EmbeddingInitial();
      // }

      // torch::Tensor valid_loss=get_valid_loss(dataset.getValidDataset(train_eventSize));
      // std::cout<<"valid loss:"<<valid_loss.item().toFloat()<<std::endl;
      BackToInitial();
    }
    
  }
}

int main(int argc,char *argv[]){
    std::cout<<"train dygnn"<<std::endl;
    neutron::Session::Init(argc,argv);
    neutron::train();
    return 0;
}