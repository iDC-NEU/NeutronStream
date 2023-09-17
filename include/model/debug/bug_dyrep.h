#ifndef NEUTRONSTREAM_BUG_DYREP_H
#define NEUTRONSTREAM_BUG_DYREP_H
#include <torch/torch.h>
#include <process/dysubg.h>
#include <dygstore/event.h>
#include <process/sampler.hpp>

namespace neutron{
    namespace debug{    
    class DyRepSubg:public DySubGraph{
	public:
		DyRepSubg(const Event &event): DySubGraph(event) {}
		// torch::Tensor gather(size_t uid,int64_t hidden,const torch::Tensor &S);
	};

	class DyRepSampler : public SampleFCB<int>{
		public:
			typedef std::shared_ptr<DyRepSampler> ptr;
		private:
			size_t num_nodes;
			size_t num_surv_samples;
			std::vector<torch::Tensor> src_embeddings;
			std::vector<torch::Tensor> dst_embeddings;
			std::vector<torch::Tensor> non_embeddings1;
			std::vector<torch::Tensor> non_embeddings2;

		public:
			DyRepSampler(size_t _num_nodes,size_t _num_surv_samples):
			num_nodes(_num_nodes),num_surv_samples(_num_surv_samples){
			};
			
			int sample(const std::shared_ptr<DySubGraph> dysubg) override;
			std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> getLabels();
			void clear();
	};
	class DyRep:public torch::nn::Module{
		private:
			torch::nn::Linear W_h= nullptr;
			torch::nn::Linear W_struct= nullptr;
			torch::nn::Linear W_rec= nullptr;
			torch::nn::Linear W_t= nullptr;
			torch::nn::Linear Omega0= nullptr;
			torch::nn::Linear Omega1= nullptr;
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
			DyRepSampler::ptr sampler;

			std::vector<int64_t> times_key;
			torch::Tensor Lambda_dict;
			const int N=5000;
			int lambda_count=0;
			bool lambda_loop=false;

		public:
			DyRep(size_t num_nodes, 
						int64_t hidden_dim,
						int num_assocail_types=1,
						int N_Surv_Samples=5);
			void init_weight();
			void generate_S();
			torch::Tensor gather(DyRepSubg &subg,size_t nid);
      void update_S(size_t src_id,size_t dst_id,int src_degree,torch::Tensor lambda,int k);

			torch::Tensor intensity_rate_lambda_batch(torch::Tensor &u_embedding,torch::Tensor &v_embedding,const std::vector<int> &k);
			torch::Tensor compute_cond_density(size_t src_id,size_t dst_id,const torch::Tensor &TimeBar);
			std::tuple<torch::Tensor,torch::Tensor> forward(std::vector<Event> &events);
			std::tuple<torch::Tensor,torch::Tensor> predict(std::vector<Event> &events);
			torch::Tensor compute_lambda_pred(Event &evt);
		
	};

} // namespace debug    
} // namespace neutron

#endif


