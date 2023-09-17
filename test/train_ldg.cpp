#include <model/ldg.h>
#include <log/log.h>
#include <NumCpp.hpp>
#include <dataset/dataset.h>
#include <dygstore/interface.hpp>
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

        bool is_temporal=Session::GetInstance()->GetIsTemporal();
        Dataset dataset(name, datadir,is_temporal);
        size_t batch_num = dataset.getBatchNum();
        size_t NUM_NODES = dataset.getNumVertices();
        auto device = sessionptr->GetDevice();
        std::cout<<"device:"<<device << std::endl;

        auto model = LDG(NUM_NODES,
                            HIDDEN_DIM);

        double lr = 0.0002;
        torch::optim::AdamOptions adamOptions(lr);
        adamOptions.betas(std::make_tuple(0.5, 0.999));
    //        std::cout<<"model.parameters():"<<model.parameters()<<std::endl;
        torch::optim::Adam optimizer(model.parameters(), adamOptions);
        torch::optim::StepLR stepLr(optimizer, 10, 0.5);

        model.to(device);

        for(size_t epoch =0;epoch<Epochs;epoch++){
            Timing epoch_time("epoch_"+std::to_string(epoch));
            sessionptr->SetCurEpoch(epoch);
            for (size_t batch_idx = 0; batch_idx < batch_num; batch_idx++) {
                model.train();
                optimizer.zero_grad();
                std::cout << "batch " << batch_idx + 1 << "/" << batch_num << " ";
                sessionptr->SetCurBatchIdx(batch_idx);
                Timing batch_time("batch");
                std::vector<Event> data = dataset.getIdxBatch(batch_idx);
                double tf=cpuSecond();
                auto output = model.forward(data);
                torch::Tensor loss1 = std::get<0>(output);
                torch::Tensor loss2 = std::get<1>(output);
                torch::Tensor loss = (loss1 + loss2) / static_cast<int>(data.size());
                Session::GetInstance()->addTimer("forward",cpuSecond()-tf);
                std::cout<<"loss:"<<loss.item().toFloat()<<" ";
                
                double bt=cpuSecond();
                loss.backward();
                torch::nn::utils::clip_grad_value_(model.parameters(),100);
                optimizer.step();
                sessionptr->addTimer("backward",cpuSecond()-bt);
                batch_time.end();
                merge();
            }
            epoch_time.end();
            sessionptr->printAllTimers(true);
        }
        
    }
}// neutron


int main(int argc ,char *argv[]){
    NEUTRON_LOG_INFO(LOG_ROOT())<<"train ldg"<<std::endl;
    neutron::Session::Init(argc,argv);
    neutron::train();
    return 0;
}