#include<model/debug/bug_dygnn.h>
#include <session.h>
#include <process/engine/interface.hpp>
#include <process/graphop.h>
#include <dygstore/save_state.hpp>
#include <utils/acc_utils.h>
namespace neutron{
    namespace debug{
        int DyGNNSampler::sample(const std::shared_ptr<DySubGraph> dysubg){
                Event e= dysubg->get_event();
                size_t src_id=e.src_id;
                size_t dst_id=e.dst_id;
		torch::Tensor src_emb=index(src_id,4);
		output_rep_head.push_back(src_emb);
		torch::Tensor dst_emb=index(dst_id,4);
		output_rep_tail.push_back(dst_emb);
		std::vector<size_t> tail_neg_list_snodes;
		std::vector<size_t> head_neg_list_snodes;
		tail_neg_list_snodes.reserve(NUM_NEGATIVE);
		head_neg_list_snodes.reserve(NUM_NEGATIVE);
		std::vector<size_t> head_node_head_neighbors;
		std::vector<size_t> tail_node_head_neighbors;
		std::vector<size_t> head_node_tail_neighbors;
		std::vector<size_t> tail_node_tail_neighbors;
		all_head_nodes.insert(src_id);
		all_tail_nodes.insert(dst_id);
		if(dysubg->in_degree(src_id)){
                        head_node_head_neighbors = dysubg->in_neighbors(src_id);
                        all_head_nodes.insert(head_node_head_neighbors.begin(),head_node_head_neighbors.end());
		}
		if(dysubg->in_degree(dst_id)){
                        tail_node_head_neighbors = dysubg->in_neighbors(dst_id);
                        all_head_nodes.insert(tail_node_head_neighbors.begin(),tail_node_head_neighbors.end());
		}
		if(dysubg->out_degree(src_id)){
                        head_node_tail_neighbors= dysubg->out_neighbors(src_id);
                        all_tail_nodes.insert(head_node_tail_neighbors.begin(),head_node_tail_neighbors.end());
		}
		if(dysubg->out_degree(dst_id)){
                        tail_node_tail_neighbors= dysubg->out_neighbors(dst_id);
                        all_tail_nodes.insert(tail_node_tail_neighbors.begin(),tail_node_tail_neighbors.end());
		}
		nc::NdArray<size_t> tail_neg_samples;
		nc::NdArray<size_t> head_neg_samples;
                std::vector<size_t> tmp_tail_candidates;
		std::vector<size_t> tmp_head_candidates;
		std::vector<size_t> tail_candidates;
		std::vector<size_t> head_candidates;
		// std::vector<size_t> all_tail_nodes_vec(all_tail_nodes.begin(),all_tail_nodes.end());
		// std::vector<size_t> all_head_nodes_vec(all_head_nodes.begin(),all_head_nodes.end());
                set_diff(all_tail_nodes, {src_id, dst_id}, tmp_tail_candidates);
		set_diff(tmp_tail_candidates, head_node_tail_neighbors, tail_candidates);
                set_diff(all_head_nodes, {src_id, dst_id}, tmp_head_candidates);
		set_diff(tmp_head_candidates, tail_node_head_neighbors, head_candidates);
		//printf("fuck n4\n");
		if(tail_candidates.empty()){
			tail_neg_samples = nc::random::choice(nc::arange<size_t>(NUM_NODES),NUM_NEGATIVE);
		}else{
			tail_neg_samples=nc::random::choice(nc::NdArray<size_t>(tail_candidates.begin(),tail_candidates.end()),NUM_NEGATIVE);
		}
		if(head_candidates.empty()){
			head_neg_samples=nc::random::choice(nc::arange<size_t>(NUM_NODES),NUM_NEGATIVE);
		}
		else{
			head_neg_samples=nc::random::choice(nc::NdArray<size_t >(head_candidates.begin(),head_candidates.end()),NUM_NEGATIVE);
		}
		//printf("fuck n5\n");
		for(auto &nid:tail_neg_samples) {
			tail_neg_list_snodes.emplace_back(nid);
		}
		for(auto &nid:head_neg_samples){
			head_neg_list_snodes.emplace_back(nid);
		}
		std::vector<torch::Tensor> head_neg_list_single=samples(head_neg_list_snodes,4);
		std::vector<torch::Tensor> tail_neg_list_single=samples(tail_neg_list_snodes,4);
		for(unsigned int i = 0;i < head_neg_list_single.size();i++){
			head_neg_list.push_back(head_neg_list_single[i]);
		}
		for(unsigned int i = 0;i < tail_neg_list_single.size();i++){
			tail_neg_list.push_back(tail_neg_list_single[i]);
		}
		// std::cout<<"sample "<<src_id<<" "<<dst_id<<" finished"<<std::endl;
		return 1;
	}
    
   
    void DyGNNSampler::clear(){

        std::set<size_t>().swap(all_head_nodes);

        std::set<size_t>().swap(all_tail_nodes);
 
        std::vector<torch::Tensor>().swap(output_rep_head);

        std::vector<torch::Tensor>().swap(output_rep_tail);

        std::vector<torch::Tensor>().swap(tail_neg_list);

        std::vector<torch::Tensor>().swap(head_neg_list);
    }


  DyGNN::DyGNN(size_t num_nodes, 
  int64_t embedding_dims, 
  int64_t edge_output_size, 
  std::string act,
  size_t num_negative,
  float drop) {
    std::cout << "DyGNN contruster" << std::endl;
    this->NUM_NODES = num_nodes;
    this->NUM_NEGATIVE = num_negative;
    this->act_str = act;
    this->embedding_dims = embedding_dims;
    exe_mode = Session::GetInstance()->GetExeMode();

    device = Session::GetInstance()->GetDevice();
    decayer = new Decayer();
    decayer->to(device);
    
    edge_updater_head = new EdgeUpdater(embedding_dims, edge_output_size);
    edge_updater_head->to(device);
    edge_updater_tail = new EdgeUpdater(embedding_dims, edge_output_size);
    edge_updater_tail->to(device);
    node_updater_head = new TLSTM(edge_output_size, embedding_dims);
    node_updater_head->to(device);
    node_updater_tail = new TLSTM(edge_output_size, embedding_dims);
    node_updater_tail->to(device);
    combiner = new Combiner(embedding_dims, embedding_dims, std::move(act));
    combiner->to(device);
    attention = new Attention(embedding_dims);
    attention->to(device);
    tran_head_edge_head = register_module("tran_head_edge_head",
                                          torch::nn::Linear(edge_output_size, embedding_dims));
	tran_head_edge_head->options.bias(true);
    tran_head_edge_tail = register_module("tran_head_edge_tail",
                                          torch::nn::Linear(edge_output_size, embedding_dims));
	tran_head_edge_tail->options.bias(true);
    tran_tail_edge_head = register_module("tran_tail_edge_head",
                                          torch::nn::Linear(edge_output_size, embedding_dims));
	tran_tail_edge_head->options.bias(true);
    tran_tail_edge_tail = register_module("tran_tail_edge_tail",
                                          torch::nn::Linear(edge_output_size, embedding_dims));
	tran_tail_edge_tail->options.bias(true);


	transfer2head = register_module("transfer2head",
                                            torch::nn::Linear(embedding_dims, embedding_dims));
    transfer2head->options.bias(false);
    transfer2tail = register_module("transfer2tail",
                                      torch::nn::Linear(embedding_dims, embedding_dims));
    transfer2tail->options.bias(false);
	
	transfer2head->to(device);
    transfer2tail->to(device);

	bceWithLogitsLoss = register_module("bceWithLogitsLoss",torch::nn::BCEWithLogitsLoss())  ;                                   
    dropout = register_module("dropout", torch::nn::Dropout(drop));
    sampler=std::make_shared<DyGNNSampler>(num_nodes,num_negative,embedding_dims);
  }
  std::tuple<torch::Tensor,float> DyGNN::forward(std::vector<Event> &events) {

    
    std::function<void(Event &)> update_graph = [&](Event &event) {
          AddEdge(event.src_id, event.dst_id);
    };
    std::function<torch::Tensor(DyGNNSubG &)> time_encoder = [&](DyGNNSubG &dySubGraph) -> torch::Tensor {
        TimePoint src_pre_time=dySubGraph.get_pre_time(dySubGraph.get_event_src());
        TimePoint dst_pre_time=dySubGraph.get_pre_time(dySubGraph.get_event_dst());
        TimePoint cur_time=dySubGraph.get_time_point();
        auto head_delta_t = torch::tensor(time_delta_hour(cur_time, src_pre_time));
        auto tail_delta_t = torch::tensor(time_delta_hour(cur_time, dst_pre_time));
        auto transed_head_delta_t = decayer->forward(head_delta_t);
        auto transed_tail_delta_t = decayer->forward(tail_delta_t);
        return torch::cat({transed_head_delta_t.view({1}), transed_tail_delta_t.view({1})});
    };
    
    std::function<void(DyGNNSubG &)> update_fun = [&](DyGNNSubG &dySubGraph) {
      Event evt = dySubGraph.get_event();
      torch::Tensor current_t = torch::tensor(evt.time_point.toHour()).to(device);
      //获取 
//0 - cell-head
//1 - cell-tail
//2 - hidden-head
//3 - hidden-tail
//4 - node_representation
      torch::Tensor head_node_cell_head = index(evt.src_id,0);

      torch::Tensor head_node_hidden_head=index(evt.src_id,2);

      torch::Tensor head_node_hidden_tail=index(evt.src_id,3);

      torch::Tensor head_node_rep = index(evt.src_id,4);

      torch::Tensor tail_node_cell_tail =index(evt.dst_id,1);

      torch::Tensor tail_node_hidden_tail = index(evt.dst_id,3);

      torch::Tensor tail_node_hidden_head=index(evt.dst_id,2);

      torch::Tensor tail_node_rep = index(evt.dst_id,4); 

      torch::Tensor time_exp = time_encoder(dySubGraph);

      torch::Tensor transed_head_delta_t = time_exp[0];
      torch::Tensor transed_tail_delta_t = time_exp[1];

      torch::Tensor edge_info_head = this->edge_updater_head->forward(head_node_rep, tail_node_rep);

      torch::Tensor edge_info_tail = this->edge_updater_tail->forward(head_node_rep, tail_node_rep);

      torch::Tensor updated_head_node_cell_head,updated_head_node_hidden_head;

      std::tie(updated_head_node_cell_head,updated_head_node_hidden_head)
      =this->node_updater_head->forward(
          edge_info_head, 
          head_node_cell_head, 
          head_node_hidden_head, 
          transed_head_delta_t);

      torch::Tensor updated_tail_node_cell_tail,updated_tail_node_hidden_tail;
      
      std::tie(updated_tail_node_cell_tail,updated_tail_node_hidden_tail)
      = this->node_updater_tail->forward(
              edge_info_tail, 
              tail_node_cell_tail, 
              tail_node_hidden_tail, 
              transed_tail_delta_t);

      torch::Tensor update_head_node_rep = combiner->forward(
              updated_head_node_hidden_head, head_node_hidden_tail);

      torch::Tensor update_tail_node_rep = combiner->forward(
              tail_node_hidden_head, updated_tail_node_hidden_tail); 
              
//0 - cell-head
//1 - cell-tail
//2 - hidden-head
//3 - hidden-tail
//4 - node_representation
        update(evt.src_id,updated_head_node_cell_head,0);
        update(evt.src_id,updated_head_node_hidden_head,2);
        update(evt.src_id,update_head_node_rep,4);

        update(evt.dst_id,updated_tail_node_cell_tail,1);
        update(evt.dst_id,updated_tail_node_hidden_tail,3);
        update(evt.dst_id,update_tail_node_rep,4);

        this->propagation(dySubGraph, evt.src_id, update_head_node_rep, current_t, edge_info_head, "head");

        this->propagation(dySubGraph, evt.dst_id, update_tail_node_rep, current_t, edge_info_tail, "tail");

    };
    
    sampler->clear();
	
    for(size_t i=0;i<events.size();i++){
			double t_update_graph=cpuSecond();
			update_graph(events[i]);
			Session::GetInstance()->addTimer("update graph",cpuSecond()-t_update_graph);

			double t_generate =cpuSecond();
			DySubGraph::ptr subg(new DyGNNSubG(events[i]));
			Session::GetInstance()->addTimer("generate graph",cpuSecond()-t_generate);
			double t_update_emb=cpuSecond();
			update_fun(*(dynamic_cast<DyGNNSubG*>(subg.get())));
			Session::GetInstance()->addTimer("update emb",cpuSecond()-t_update_emb);
			double t_sample = cpuSecond();
			sampler->sample(subg);
			Session::GetInstance()->addTimer("sample",cpuSecond()-t_sample);
			if(Session::GetInstance()->GetSlideMode()!="other"){
				save_window_step();
			}
		
	}
    
	torch::Tensor output_rep_head_tensor =torch::cat(sampler->output_rep_head).view({-1, this->embedding_dims});
	torch::Tensor output_rep_tail_tensor =torch::cat(sampler->output_rep_tail).view({-1, this->embedding_dims});
	torch::Tensor head_neg_tensors=torch::cat(sampler->head_neg_list).view({-1, this->embedding_dims});
	torch::Tensor tail_neg_tensors=torch::cat(sampler->tail_neg_list).view({-1, this->embedding_dims});

	output_rep_head_tensor = this->dropout->forward(this->transfer2head->forward(output_rep_head_tensor));
	output_rep_tail_tensor = this->dropout->forward(this->transfer2tail->forward(output_rep_tail_tensor));
	head_neg_tensors = this->dropout->forward(this->transfer2head->forward(head_neg_tensors));
	tail_neg_tensors = this->dropout->forward(this->transfer2tail->forward(tail_neg_tensors));

	torch::Tensor head_pos_tensors =output_rep_head_tensor.clone().repeat(
		{1,static_cast<long>(NUM_NEGATIVE)});
	torch::Tensor tail_pos_tensors=output_rep_tail_tensor.clone().repeat(
		{1,static_cast<long>(NUM_NEGATIVE)});
	
	int64_t num_pp = output_rep_head_tensor.size(0);

	torch::Tensor labels_p = torch::tensor(std::vector<int>(num_pp, 1)).to(torch::kFloat).to(device).requires_grad_(
			true);
	torch::Tensor labels_n = torch::tensor(std::vector<int>(num_pp * NUM_NEGATIVE * 2, 0)).to(torch::kFloat).to(
			device).requires_grad_(true);
	
	torch::Tensor labels = torch::cat({labels_p, labels_n}).to(torch::kFloat);

	torch::Tensor scores_p = torch::bmm(output_rep_head_tensor.view({num_pp, 1, embedding_dims}),
										output_rep_tail_tensor.view({num_pp, embedding_dims, 1}));

	torch::Tensor scores_n_1 = torch::bmm(
			head_neg_tensors.view({static_cast<long>(num_pp * NUM_NEGATIVE), 1, embedding_dims}),
			tail_pos_tensors.view({static_cast<long>(num_pp * NUM_NEGATIVE), embedding_dims, 1}));

	torch::Tensor scores_n_2 = torch::bmm(
			head_pos_tensors.view({static_cast<long>(num_pp * NUM_NEGATIVE), 1, embedding_dims}),
			tail_neg_tensors.view({static_cast<long>(num_pp * NUM_NEGATIVE), embedding_dims, 1}));

	torch::Tensor scores = torch::cat({scores_p, scores_n_1, scores_n_2}).view(
			{static_cast<long>(num_pp * (1 + 2 * NUM_NEGATIVE))});

  torch::Tensor losses = torch::binary_cross_entropy_with_logits(scores, labels);
	float auc=acc::roc_auc_scores(labels,scores);
	return {losses,auc};
  }
  void DyGNN::propagation(DyGNNSubG &dySubGraph,
                          size_t node_index,
                          const torch::Tensor &node_rep,
                          const torch::Tensor &current,
                          const torch::Tensor &edge_info,
                          const std::string &node_type) {
		size_t in_degree_t = dySubGraph.in_degree(node_index);
		if (in_degree_t) {
        
		torch::Tensor head_timestamps_tensor = 
		gather_in_timestamps(dySubGraph,node_index).view({-1, 1}).to(device);

		torch::Tensor head_nei_edge_info;
		if (node_type == "head") {
			head_nei_edge_info=this->tran_head_edge_head->forward(edge_info);
		} else {

			head_nei_edge_info=this->tran_tail_edge_head->forward(edge_info);
		}

		torch::Tensor head_delta_ts =
				current.repeat({static_cast<int64_t>(in_degree_t), 1}) - head_timestamps_tensor;

		torch::Tensor trans_head_delta_ts = this->decayer->forward(head_delta_ts);

		torch::Tensor head_nei_cell=gather_in(dySubGraph,node_index,0);

		torch::Tensor tran_head_nei_edge_info =
		head_nei_edge_info.repeat({static_cast<int64_t>(in_degree_t), 1}) * trans_head_delta_ts;

		torch::Tensor node_reps = node_rep.repeat({static_cast<int64_t>(in_degree_t)}).view(
				{-1, this->embedding_dims});

		torch::Tensor head_nei_reps =gather_in(dySubGraph,node_index,4);

		torch::Tensor att_score_head = this->attention->forward(node_reps, head_nei_reps);

		head_nei_cell = head_nei_cell + tran_head_nei_edge_info * att_score_head;

		torch::Tensor head_nei_hidden = this->act(head_nei_cell);

		torch::Tensor head_nei_tail_hidden =gather_in(dySubGraph,node_index,3);

		torch::Tensor head_nei_rep = combiner->forward(head_nei_hidden, head_nei_tail_hidden);
//0 - cell-head
//1 - cell-tail
//2 - hidden-head
//3 - hidden-tail
//4 - node_representation
		scatter_in(dySubGraph,node_index, head_nei_cell,0);
		scatter_in(dySubGraph,node_index, head_nei_hidden,2);
		scatter_in(dySubGraph,node_index, head_nei_rep,4);

      	}
      	size_t out_degree_t = dySubGraph.out_degree(node_index);
		if (out_degree_t) {

		torch::Tensor tail_timestamps_tensor = gather_out_timestamps(dySubGraph,node_index).view({-1, 1}).to(device);

		torch::Tensor tail_nei_edge_info;
		if (node_type == "head") {
				tail_nei_edge_info = this->tran_head_edge_tail->forward(edge_info);
		} else {
				tail_nei_edge_info = this->tran_tail_edge_tail->forward(edge_info);
		}

		torch::Tensor tail_delta_ts =
						current.repeat({static_cast<int64_t>(out_degree_t), 1}) - tail_timestamps_tensor;

		torch::Tensor trans_tail_delta_ts = this->decayer->forward(tail_delta_ts);

		torch::Tensor tail_nei_cell = gather_out(dySubGraph,node_index,1);

		torch::Tensor tran_tail_nei_edge_info =
                tail_nei_edge_info.repeat({static_cast<int64_t>(out_degree_t), 1}) * trans_tail_delta_ts;

		torch::Tensor node_reps = node_rep.repeat({static_cast<int64_t>(out_degree_t), 1}).view(
						{-1, this->embedding_dims});


		torch::Tensor tail_nei_reps = gather_out(dySubGraph,node_index,4);

		torch::Tensor att_score_tail = this->attention->forward(node_reps, tail_nei_reps);

		tail_nei_cell = tail_nei_cell + tran_tail_nei_edge_info * att_score_tail;

		torch::Tensor tail_nei_hidden = this->act(tail_nei_cell);

		torch::Tensor tail_nei_head_hidden = gather_out(dySubGraph,node_index,2);

		torch::Tensor tail_nei_rep = this->combiner->forward(tail_nei_head_hidden, tail_nei_hidden);
//0 - cell-head
//1 - cell-tail
//2 - hidden-head
//3 - hidden-tail
//4 - node_representation
		scatter_out(dySubGraph,node_index,tail_nei_cell,1);
		scatter_out(dySubGraph,node_index,tail_nei_hidden,3);
		scatter_out(dySubGraph,node_index,tail_nei_rep,4);
    }
}      
    } // namespace debug
    
    
} // namespace neutron
