#include <model/ldg.h>
#include <NumCpp.hpp>
#include <utils/mytime.h>
#include <process/graphop.h>
#include <process/engine/interface.hpp>
namespace neutron
{
    // torch::Tensor LDGSubg::gather(size_t nid,int64_t hidden,const torch::Tensor &S){
    //     torch::Tensor h_struct=torch::zeros({hidden}).to(S.device());
    //     if(in_degree(nid)>0){
    //         torch::Tensor agg=gather_in(*this,nid);

    //         torch::Tensor mask=get_mask_in_neighbors(*this,nid,S.size(0));

    //         torch::Tensor att=S[static_cast<int64_t>(nid)].masked_select(mask);
    //         // CHECK(!att.isnan().any().item().toBool())<<"att: "<<att;
    //         // CHECK(!att.isinf().any().item().toBool())<<"att: "<<att;
    //         // NEUTRON_LOG_INFO(LOG_ROOT())<<"att:\n"<<att<<std::endl;
    //         torch::Tensor q_ui=torch::exp(att).view({-1,1});
    //         torch::Tensor q_u=(q_ui/(torch::sum(q_ui)+1e-7));

    //         // CHECK(!q_u.isnan().any().item().toBool())<<"q_u:"<<q_u<<" q_ui:"<<q_ui
    //         // <<"att :"<<att<<" S:"<<S;
    //         h_struct=std::get<0>(torch::max(torch::sigmoid(q_u*agg),0));
    //     }
    //     return h_struct;
    // }

    int LDGSampler::sample(const std::shared_ptr<DySubGraph> dysubg)
    {
        Event e = dysubg->get_event();
        size_t src_id = e.src_id;
        size_t dst_id = e.dst_id;
        nc::NdArray<int> rand_samples =
            nc::random::choice(nc::deleteIndices<int>(nc::arange<int>(num_nodes),
                                                      nc::NdArray({static_cast<nc::uint32>(src_id),
                                                                   static_cast<nc::uint32>(dst_id)})),
                               num_surv_samples * 2, false);
        auto src_dst_emb = samples({src_id, dst_id});
        src_embeddings.push_back(src_dst_emb[0]);
        dst_embeddings.push_back(src_dst_emb[1]);
        std::vector<size_t> non_embedding1_snode;
        std::vector<size_t> non_embedding2_snode;
        for (size_t i = 0; i < num_surv_samples; i++)
        {
            for (int k = 0; k < 2; k++)
            {
                non_embedding1_snode.emplace_back(src_id);
                non_embedding1_snode.emplace_back(rand_samples.at(num_surv_samples + i));
                non_embedding2_snode.emplace_back(dst_id);
                non_embedding2_snode.emplace_back(rand_samples.at(i));
            }
        }
        auto non1_snode = samples(non_embedding1_snode);
        auto non2_snode = samples(non_embedding2_snode);
        non_embeddings1.insert(non_embeddings1.end(), non1_snode.begin(), non1_snode.end());
        non_embeddings2.insert(non_embeddings2.end(), non2_snode.begin(), non2_snode.end());
        return 1;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LDGSampler::getLabels()
    {
        torch::Tensor embedding_1_cat = torch::stack(src_embeddings, 0);
        torch::Tensor embedding_2_cat = torch::stack(dst_embeddings, 0);
        torch::Tensor non_embedding_1_cat = torch::stack(non_embeddings1, 0);
        torch::Tensor non_embedding_2_cat = torch::stack(non_embeddings2, 0);
        return {embedding_1_cat, embedding_2_cat, non_embedding_1_cat, non_embedding_2_cat};
    }

    void LDGSampler::clear()
    {
        src_embeddings.clear();
        dst_embeddings.clear();
        non_embeddings1.clear();
        non_embeddings2.clear();
    }

    LDG::LDG(size_t num_nodes, int64_t hidden_dim, int num_assocail_types, int N_Surv_Samples)
    {
        this->num_nodes = num_nodes;
        this->n_hidden = hidden_dim;
        this->num_assocail_types = num_assocail_types;
        this->num_surv_samples = N_Surv_Samples;
        device = Session::GetInstance()->GetDevice();

        W_h = register_module("W_h", torch::nn::Linear(hidden_dim, hidden_dim));
        W_struct = register_module("W_struct", torch::nn::Linear(n_hidden * this->num_assocail_types, n_hidden));
        W_rec = register_module("W_rec", torch::nn::Linear(hidden_dim, hidden_dim));
        W_t = register_module("W_t", torch::nn::Linear(4, hidden_dim));
        int64_t d1 = this->n_hidden;
        int64_t d2 = this->n_hidden;
        Omega0 = register_module("Omega0", torch::nn::Bilinear(d1, d1, 1));
        Omega1 = register_module("Omega1", torch::nn::Bilinear(d2, d2, 1));
        psi = register_parameter("psi", 0.5 * torch::ones(n_types));
        generate_S();
        sampler = std::make_shared<LDGSampler>(num_nodes, num_surv_samples);
        init_weight();
    }

    void LDG::init_weight()
    {

        torch::nn::init::xavier_normal_(this->W_h->weight.data());
        torch::nn::init::xavier_normal_(this->W_struct->weight.data());
        torch::nn::init::xavier_normal_(this->W_rec->weight.data());
        torch::nn::init::xavier_normal_(this->W_t->weight.data());
        torch::nn::init::xavier_normal_(this->Omega0->weight.data());
        torch::nn::init::xavier_normal_(this->Omega1->weight.data());
    }

    void LDG::generate_S()
    {
        S = torch::zeros({static_cast<int64_t>(num_nodes), static_cast<int64_t>(num_nodes)}).to(torch::kFloat32).to(device);
        for (size_t i = 0; i < num_nodes; i++)
        {
            std::vector<size_t> neighbors = OutNeighbors(i);
            size_t out_degree = neighbors.size();
            
            if (out_degree == 0)
                continue;
            // std::cout<<i<<" "<<out_degree<<std::endl;
            for (size_t j = 0; j < neighbors.size(); j++){
                S[i][neighbors[j]] = 1.0f / out_degree;
            }
        }
    }
    torch::Tensor LDG::gather(LDGSubg &dysubg, size_t nid)
    {
        torch::Tensor h_struct = torch::zeros({this->n_hidden}).to(S.device());
        if (dysubg.in_degree(nid) > 0)
        {
            torch::Tensor agg = this->W_h->forward(gather_in(dysubg, nid));
            torch::Tensor mask = get_mask_in_neighbors(dysubg, nid, S.size(0));

            torch::Tensor att = S[static_cast<int64_t>(nid)].masked_select(mask);
            CHECK(!att.isnan().any().item().toBool()) << "att: " << att;
            CHECK(!att.isinf().any().item().toBool()) << "att: " << att;
            // std::cout<<"att:\n"<<att<<std::endl;
            // NEUTRON_LOG_INFO(LOG_ROOT())<<"att:\n"<<att<<std::endl;
            torch::Tensor q_ui = torch::exp(att).view({-1, 1});
            torch::Tensor q_u = (q_ui / (torch::sum(q_ui) + 1e-7));

            CHECK(!q_u.isnan().any().item().toBool()) << "q_u:" << q_u << " q_ui:" << q_ui
                                                      << "att :" << att << " S:" << S;
            h_struct = std::get<0>(torch::max(torch::sigmoid(q_u * agg), 0));
        }
        return h_struct;
    }

    void LDG::update_S(size_t src_id, size_t dst_id, int degree, torch::Tensor lambda, int k)
    {
        if (!HasEdge(src_id, dst_id))
            return;
        float b = (degree==0) ? 1 : 1.0f / (degree + 1e-7);
        if (k == 0)
        {
            torch::Tensor y = S.index({static_cast<int64_t>(src_id)});
            y[dst_id] = lambda.item().toFloat() + b;
            y /= (torch::sum(y) + 1e-7);
            S.index_put_({static_cast<int64_t>(src_id)}, y);
        }
        else if (k == 1)
        {
            auto y = S.index({static_cast<int64_t>(src_id)});
            float b_prime =(degree==0) ? 0 : 1.0f / (degree + 1e-7);
            float x = b_prime - b;
            y[dst_id] = lambda.item().toFloat() + b;
            for (auto &nid : OutNeighbors(src_id))
            {
                if (nid != dst_id)
                    y[nid] = y[nid] - x;
            }
            y /= (torch::sum(y) + 1e-7);
            S.index_put_({static_cast<int64_t>(src_id)}, y);
        }
    } 
    std::tuple<torch::Tensor, torch::Tensor> LDG::forward(std::vector<Event> &events)
    {
        std::vector<std::vector<size_t>> vertex_degree;
        std::vector<int> event_type;
        event_type.reserve(events.size());

        std::function<void(Event &)> update_graph = [&](Event &event)
        {
            if (event.eventType == AE)
            {
                AddEdge(event.src_id, event.dst_id);
                AddEdge(event.dst_id, event.src_id);
                event_type.push_back(0);
            }
            else if (event.eventType == Communication)
            {
                event_type.push_back(1);
            }
            // std::cout<<"evt ("<<event.src_id<<" "<<event.dst_id<<") od:"<<OutDegree(event.src_id)<<" "<<OutDegree(event.dst_id)<<std::endl;
            vertex_degree.push_back({OutDegree(event.src_id), OutDegree(event.dst_id)});
        };

        std::function<torch::Tensor(LDGSubg &)> time_encoders = [&](LDGSubg &dySubGraph) -> torch::Tensor
        {
            static torch::Tensor time_mm = torch::tensor({0, 0, 0, 0});
            static torch::Tensor time_sd = torch::tensor({50, 7, 15, 15});
            TimePoint src_pre_time = dySubGraph.get_pre_time(dySubGraph.get_event_src());
            TimePoint dst_pre_time = dySubGraph.get_pre_time(dySubGraph.get_event_dst());
            TimePoint cur_time = dySubGraph.get_time_point();
            std::vector<int> time_span1 = time_delta_dhms(cur_time, src_pre_time);
            std::vector<int> time_span2 = time_delta_dhms(cur_time, dst_pre_time);
            torch::Tensor tmp = torch::cat({torch::tensor(time_span1),
                                            torch::tensor(time_span2)},
                                           0)
                                    .view({2, 4});
            torch::Tensor h_t = (tmp - time_mm) / time_sd;
            return h_t;
        };

        std::function<void(LDGSubg &)> update_fun = [&](LDGSubg &dysubg)
        {
            torch::Tensor h_t = time_encoders(dysubg).to(device);
            Event evt = dysubg.get_event();
            torch::Tensor h_update = index({evt.src_id, evt.dst_id});
            torch::Tensor h2 = this->W_rec->forward(h_update.view({2, n_hidden}));
            torch::Tensor h3 = this->W_t->forward(h_t.view({2, 4}));
            torch::Tensor src_embedding = h2[0];
            torch::Tensor dst_embedding = h2[1];
            torch::Tensor src_time = h3[0];
            torch::Tensor dst_time = h3[1];
            torch::Tensor h_src_struct = gather(dysubg, evt.src_id);
            torch::Tensor h_dst_struct = gather(dysubg, evt.dst_id);
            torch::Tensor h_uv = this->W_struct->forward(torch::stack({h_src_struct, h_dst_struct}, 0));
            torch::Tensor src_new_embedding = torch::sigmoid(h_uv[1] + src_embedding + src_time);
            torch::Tensor dst_new_embedding = torch::sigmoid(h_uv[0] + dst_embedding + dst_time);
            update(evt.src_id, src_new_embedding);
            update(evt.dst_id, dst_new_embedding);
        };

        sampler->clear();
        double t_run=cpuSecond();
        run<LDGSubg>(events, update_graph, update_fun, sampler);
        Session::GetInstance()->addTimer("run",cpuSecond()-t_run);
        std::vector<int> non_types_0(events.size() * this->num_surv_samples, 0);
        std::vector<int> non_types_1(events.size() * this->num_surv_samples, 1);
        // std::vector<int> non_types_0(this->num_surv_samples,0);
        // std::vector<int> non_types_1(this->num_surv_samples,1);
        non_types_0.insert(non_types_0.end(), non_types_1.begin(), non_types_1.end());

        torch::Tensor embedding_1, embedding_2, non_embedding_1, non_embedding_2;
        std::tie(embedding_1, embedding_2, non_embedding_1, non_embedding_2) = sampler->getLabels();

        torch::Tensor lambda_uv_t = intensity_rate_lambda_batch(
            embedding_1, embedding_2, event_type);
        for (size_t i = 0; i < events.size(); i++)
        {
            auto e = events[i];
            int k = 0;
            if (e.eventType != Communication)
                k = 1;
            update_S(e.src_id, e.dst_id, vertex_degree[i][0], lambda_uv_t[i], k);
            update_S(e.dst_id, e.src_id, vertex_degree[i][1], lambda_uv_t[i], k);
        }
        torch::Tensor no_lambda_uv_t = intensity_rate_lambda_batch(
            non_embedding_1, non_embedding_2, non_types_0);

        torch::Tensor loss1 = -torch::sum(torch::log(lambda_uv_t) + 1e-10);
        torch::Tensor loss2 = torch::sum(no_lambda_uv_t / this->num_surv_samples);
        return std::make_tuple(loss1, loss2);
    }

    torch::Tensor LDG::intensity_rate_lambda_batch(torch::Tensor &u_embedding, torch::Tensor &v_embedding, const std::vector<int> &k)
    {
        torch::Tensor z_u = u_embedding.view({-1, this->n_hidden}).contiguous();
        torch::Tensor z_v = v_embedding.view({-1, this->n_hidden}).contiguous();

        torch::Tensor k_Tensor = torch::tensor(k).to(torch::kBool);
        int sum_1 = torch::sum(torch::tensor(k)).item().toInt();
        int sum_0 = k.size() - sum_1;
        std::vector<int> index_0_vec;
        std::vector<int> index_1_vec;
        for (size_t i = 0; i < k.size(); i++)
        {
            if (k[i] == 0)
                index_0_vec.push_back(i);
            else
                index_1_vec.push_back(i);
        }
        torch::Tensor index0 = torch::tensor(index_0_vec).to(device);
        torch::Tensor index1 = torch::tensor(index_1_vec).to(device);
        int64_t B = u_embedding.size(0);
        torch::Tensor cat_embedding1 = z_u.view({B, this->n_hidden});
        torch::Tensor cat_embedding2 = z_v.view({B, this->n_hidden});
        torch::Tensor result = torch::zeros({cat_embedding1.size(0), 1}).to(device);

        if (sum_1 > 0)
        {
            torch::Tensor cat1 = cat_embedding1.index_select(0, index1);
            torch::Tensor cat2 = cat_embedding2.index_select(0, index1);
            torch::Tensor gn = Omega1->forward(cat1, cat2);
            torch::Tensor gn_psi = torch::clamp(gn / (psi[1] + 1e-7), -75, 75);
            torch::Tensor gn_psi_exp = torch::log(1 + torch::exp(-gn_psi)) + gn_psi;
            torch::Tensor lambda1 = psi[1] * gn_psi_exp;
            for (size_t i = 0; i < index_1_vec.size(); i++)
            {
                result[index_1_vec[i]] = lambda1[i];
            }
        }
        if (sum_0 > 0)
        {

            torch::Tensor cat1 = cat_embedding1.index_select(0, index0);
            torch::Tensor cat2 = cat_embedding2.index_select(0, index0);
            torch::Tensor gn = Omega0->forward(cat1, cat2);
            torch::Tensor gn_psi = torch::clamp(gn / (psi[0] + 1e-7), -75, 75);
            torch::Tensor gn_psi_exp = torch::log(1 + torch::exp(-gn_psi)) + gn_psi;
            torch::Tensor lambda0 = psi[0] * gn_psi_exp;
            for (size_t i = 0; i < index_0_vec.size(); i++)
            {
                result[index_0_vec[i]] = lambda0[i];
            }
        }
        return result.flatten();
    }

} // namespace neutron