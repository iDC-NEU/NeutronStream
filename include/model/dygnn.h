#ifndef NEUTRONSTREAM_DYGNN_H
#define NEUTRONSTREAM_DYGNN_H
#include <process/sampler.hpp>
#include<unordered_set>
#include <torch/torch.h>
#include <dygstore/event.h>
#include <process/dysubg.h>
#include <process/engine/interface.hpp>
#include <dygstore/interface.hpp>
#include <NumCpp.hpp>
namespace neutron{
    class DyGNN;

    class DyGNNSubG:public DySubGraph{
    public:
        DyGNNSubG()=default;
        DyGNNSubG(const Event &evt): DySubGraph(evt,1){
            std::vector<size_t> src_out= out_neighbors(evt.src_id);
            std::vector<size_t> dst_out= out_neighbors(evt.dst_id);
            std::vector<size_t> src_in = in_neighbors(evt.src_id);
            std::vector<size_t> dst_in = in_neighbors(evt.dst_id);
            src_out.insert(src_out.end(),dst_out.begin(),dst_out.end());
            src_out.insert(src_out.end(),src_in.begin(),src_in.end());
            src_out.insert(src_out.end(),dst_in.begin(),dst_in.end());
            setUpdateSet(src_out);
        };
    };

  class DyGNNSampler:public SampleFCB<int>{
    friend class DyGNN;
    private:
        std::set<size_t> all_head_nodes;
        std::set<size_t> all_tail_nodes;
        std::vector<torch::Tensor> output_rep_head;
        std::vector<torch::Tensor> output_rep_tail;
        std::vector<torch::Tensor> tail_neg_list;
        std::vector<torch::Tensor> head_neg_list;
        size_t NUM_NODES;
        size_t NUM_NEGATIVE;
        int64_t EMBEDDING_DIM;
        torch::Device device=torch::kCPU;
    public:
    DyGNNSampler(size_t _num_nodes,size_t _num_negative,int64_t _embedding_dims):
    SampleFCB<int>(),
    NUM_NODES(_num_nodes),
    NUM_NEGATIVE(_num_negative),
    EMBEDDING_DIM(_embedding_dims){
        device=Session::GetInstance()->GetDevice();
    }
    int sample(const std::shared_ptr<DySubGraph> dysubg) override;
    void clear();


  };

//0 - cell-head
//1 - cell-tail
//2 - hidden-head
//3 - hidden-tail
//4 - node_representation

class Decayer : public torch::nn::Module{
private:
    torch::nn::Linear lin= nullptr;
    float w;
    std::string decay_method;
public:
    explicit Decayer(float w=2,std::string decay_mothod="log"){
        lin= register_module("lin",torch::nn::Linear(1,1));
        lin->options.bias(false);
        this->w=w;
        this->decay_method=std::move(decay_mothod);
    }

    torch::Tensor forward(const torch::Tensor& delta_t){
        if(this->decay_method=="exp")
            return torch::exp(-this->w * delta_t);
        if(this->decay_method=="log")
            return 1.0/torch::log(2.7183+this->w*delta_t);
        if(this->decay_method=="rev")
            return 1.0/(1+this->w*delta_t);
        else
            return torch::exp(-this->w * delta_t);
    }

};
class EdgeUpdater : public torch::nn::Module{
private:
    torch::nn::Linear h2o= nullptr;
    torch::nn::Linear l2o= nullptr;
    std::string act;

public:
    EdgeUpdater(int64_t node_input_size,int64_t output_size,std::string act="tanh",int64_t relation_input_size=0,bool bias=true){
        h2o= register_module("h2o",torch::nn::Linear(node_input_size,output_size));
        l2o= register_module("l2o",torch::nn::Linear(node_input_size,output_size));
        this->act=std::move(act);
    }
    torch::Tensor forward(const torch::Tensor& head_node, const torch::Tensor& tail_node){
        torch::Tensor tmp1=this->h2o->forward(head_node);
        torch::Tensor tmp2=this->l2o->forward(tail_node);
        torch::Tensor edge_output=tmp1+tmp2;
        torch::Tensor rst;
        if(act=="tanh")
            rst= torch::tanh(edge_output);
        else if(act=="sigmoid")
            rst= torch::sigmoid(edge_output);
        else
            rst= torch::relu(edge_output);
        return rst;
    }
};
class TLSTM:public torch::nn::Module{
private:
    torch::nn::Linear i2h= nullptr;
    torch::nn::Linear h2h= nullptr;
    torch::nn::Linear lin= nullptr;
    torch::nn::Sequential c2s= nullptr;
    torch::nn::Sigmoid sigmoid= nullptr;
public:
    TLSTM(int64_t input_size,int64_t hidden_size){
    i2h= register_module("i2h",torch::nn::Linear(input_size,4*hidden_size));
    i2h->options.bias(true);
    h2h= register_module("h2h",torch::nn::Linear(input_size,4*hidden_size));
    h2h->options.bias(true);
    c2s= register_module("c2s",torch::nn::Sequential());
    lin = register_module("lin",torch::nn::Linear(hidden_size,hidden_size));
    lin->options.bias(true);
    c2s->push_back(register_module("lin1",lin));
    c2s->push_back(register_module("tanh",torch::nn::Tanh()));

    sigmoid = register_module("sigmoid",torch::nn::Sigmoid());
}
std::tuple<torch::Tensor,torch::Tensor>
    forward(const torch::Tensor& input,const torch::Tensor &cell,const torch::Tensor &hidden,const torch::Tensor& transed_delta_t){
        torch::Tensor cell_short=c2s->forward(cell);
        torch::Tensor cell_new=cell - cell_short + cell_short * transed_delta_t;
        torch::Tensor gates=this->i2h->forward(input)+this->h2h->forward(hidden);
        auto gate_vec= gates.chunk(4,0);
        torch::Tensor in_gate =this->sigmoid->forward(gate_vec[0]);
        torch::Tensor for_gate =this->sigmoid->forward(gate_vec[1]);
        torch::Tensor cell_gate = torch::tanh(gate_vec[2]);
        torch::Tensor out_gate = this->sigmoid->forward(gate_vec[3]);
        torch::Tensor cell_output = for_gate*cell_new +in_gate*cell_gate;
        torch::Tensor hidden_output = out_gate*torch::tanh(cell_output);
        return std::make_tuple(cell_output,hidden_output);
    }
};

class Combiner :public torch::nn::Module{
private:
    torch::nn::Linear h2o  =nullptr;
    torch::nn::Linear l2o = nullptr;
    std::string act;
public:
    Combiner(int64_t input_size,int64_t output_size,std::string act){
        h2o= register_module("h2o",torch::nn::Linear(input_size,output_size));
        h2o->options.bias(true);
        l2o= register_module("l2o",torch::nn::Linear(input_size,output_size));
        l2o->options.bias(true);
        this->act=std::move(act);
    }
    torch::Tensor forward(const torch::Tensor& head_info,const torch::Tensor& tail_info){
        torch::Tensor head_info_rst=this->h2o->forward(head_info);
        torch::Tensor tail_info_rst=this->l2o->forward(tail_info);
        torch::Tensor node_output=head_info_rst+tail_info_rst;

        if(this->act == "tanh")
            return torch::tanh(node_output);
        else if(this->act == "sigmoid")
            return torch::sigmoid(node_output);
        else
            return torch::relu(node_output);
    }
};
class Attention:public torch::nn::Module{
private:
    torch::nn::Bilinear bi_linear = nullptr;

public:
    explicit Attention(int64_t embedding_dims){
        bi_linear= register_module("bi_linear",torch::nn::Bilinear(embedding_dims,embedding_dims,1));
    }
    torch::Tensor forward(const torch::Tensor& node1,const torch::Tensor& node2){
        return torch::softmax(this->bi_linear->forward(node1,node2).view({-1,1}),0);
    }

};
class DyGNN : public torch::nn::Module{
private:
    size_t NUM_NODES;
    size_t NUM_NEGATIVE;
    int64_t embedding_dims;
    std::string exe_mode;
    std::string act_str;
    torch::Device device=torch::kCPU;
    Decayer *decayer;
    EdgeUpdater *edge_updater_head;
    EdgeUpdater *edge_updater_tail;
    TLSTM * node_updater_head;
    TLSTM * node_updater_tail;
    Combiner * combiner;

    Attention *attention;
    torch::nn::Linear tran_head_edge_head =nullptr;
    torch::nn::Linear tran_head_edge_tail =nullptr;
    torch::nn::Linear tran_tail_edge_head =nullptr;
    torch::nn::Linear tran_tail_edge_tail =nullptr;

    torch::nn::Linear transfer2head =nullptr;
    torch::nn::Linear transfer2tail =nullptr;
    
    torch::nn::Dropout dropout = nullptr;
    torch::nn::BCEWithLogitsLoss bceWithLogitsLoss = nullptr;
    std::shared_ptr<DyGNNSampler> sampler;

public:

    DyGNN(size_t num_nodes,
    int64_t embedding_dims,
    int64_t edge_output_size,std::string 
    act="tanh",
    size_t num_negative=5,
    float drop=0);
    std::tuple<torch::Tensor,float> forward(std::vector<Event> &events);

    void propagation(DyGNNSubG &dySubGraph,
                     size_t node_index, 
                     const torch::Tensor &node_rep, 
                     const torch::Tensor& current, 
                     const torch::Tensor& edge_info, 
                     const std::string& node_type);
    
    torch::Tensor act(const torch::Tensor& tensor_T){
        if(act_str=="tanh")
            return torch::tanh(tensor_T);
        else if (act_str =="sigmoid")
            return torch::sigmoid(tensor_T);
        else if(act_str =="relu")
            return torch::relu(tensor_T);
        CHECK(false)<<act_str<<" not implemented yet";
        return torch::tensor(std::vector<int>{});
    }

};
}
#endif