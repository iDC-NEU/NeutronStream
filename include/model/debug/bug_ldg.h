#ifndef NEUTRONSTREAM_BUG_LDG_H
#define NEUTRONSTREAM_BUG_LDG_H
#include <process/sampler.hpp>
#include <dygstore/event.h>
#include<torch/torch.h>
#include <process/dysubg.h>

using namespace torch::indexing;
namespace neutron{
    namespace debug{    
    class LDGSubg:public DySubGraph{
        public:
            LDGSubg(const Event &evt):DySubGraph(evt){}
            // torch::Tensor gather(size_t uid,int64_t hidden,const torch::Tensor &S);
    };
    class LDGSampler:public SampleFCB<int>{
        public:
            typedef std::shared_ptr<LDGSampler> ptr;
        private:
            size_t num_nodes;
			size_t num_surv_samples;
			std::vector<torch::Tensor> src_embeddings;
			std::vector<torch::Tensor> dst_embeddings;
			std::vector<torch::Tensor> non_embeddings1;
			std::vector<torch::Tensor> non_embeddings2;
        public:
            LDGSampler(size_t _num_nodes,size_t _num_surv_samples):
			num_nodes(_num_nodes),num_surv_samples(_num_surv_samples){
			};
            int sample(const std::shared_ptr<DySubGraph> dysubg) override;
			std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> getLabels();
			void clear();
    };
    class LDG:public torch::nn::Module{
    private:
        torch::nn::Linear W_h= nullptr;
        torch::nn::Linear W_struct= nullptr;
        torch::nn::Linear W_rec= nullptr;
        torch::nn::Linear W_t= nullptr;
        torch::nn::Bilinear Omega0= nullptr;
        torch::nn::Bilinear Omega1= nullptr;
        torch::Tensor psi ;
        torch::Tensor S;
        torch::nn::Linear linear= nullptr;
        size_t num_assocail_types;
        size_t num_nodes;
        int64_t out_dim;
        int64_t n_hidden;
        int n_types=2;
        int num_surv_samples=5;
        std::string exe_mode;
        torch::Device device = torch::Device(c10::DeviceType::CPU);
        LDGSampler::ptr sampler;
    public:

        LDG(size_t num_nodes, int64_t hidden_dim,
            int num_assocail_types=1,
            int N_Surv_Samples=5);

        void init_weight();

        void generate_S();
        
        torch::Tensor gather(LDGSubg &dyg,size_t nid);

        void update_S(size_t src_id,size_t dst_id,int src_degree,torch::Tensor lambda,int k);

        std::tuple<torch::Tensor,torch::Tensor> forward(std::vector<Event> &event);

        torch::Tensor intensity_rate_lambda_batch(torch::Tensor &u_embedding,torch::Tensor &v_embedding,const std::vector<int> &k);

    };

} // namespace debug    
} // namespace neutron

#endif

