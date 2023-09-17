#include <torch/torch.h>
#include <utils/acc_utils.h>
#include <utils/util.h>
#include <dygstore/interface.hpp>
void test_grad(){
  auto x = torch::tensor({1.}, torch::requires_grad()); 
    std::cout<<x.requires_grad()<<std::endl;
    {
    torch::AutoGradMode enable_grad(true);
    auto y = x * 2;

    std::cout << y.requires_grad() << std::endl; // prints `true`
    }

    {
      torch::AutoGradMode enable_grad(false);
       // prints `false`
      auto y = x * 2;
      std::cout << y.requires_grad() << std::endl; // prints `false`
    }
    auto mm=torch::rand({2,3});
    std::cout<<mm.sizes()<<std::endl;
    std::cout<<mm.requires_grad()<<std::endl;
    mm.requires_grad();
}
void test_acc_compute(){
  torch::Tensor true_lables=torch::tensor({1,0,0,0,1,0,1,0});
  torch::Tensor pred_labels=torch::tensor({0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7});
  std::cout<<"auc:"<<neutron::acc::roc_auc_scores(true_lables,pred_labels)<<std::endl;
}
void test_emb_all(){
  const int DIM=4;
  std::vector<std::vector<int>> init_vec(10);
  for(size_t i=0;i<init_vec.size();i++){
    init_vec[i].resize(DIM,i);
  }
  torch::Tensor t=neutron::vector2tensor<int>(init_vec);
  std::cout<<t<<std::endl;
  neutron::InitEMB(t,DIM);
  std::cout<<neutron::index_all()<<std::endl;
  neutron::update(1,torch::tensor(std::vector<int>({11,11,11,11})));
  std::cout<<neutron::index_all()<<std::endl;
}

void test_average_precision(const std::string &input,const std::string &output){
  std::freopen(input.c_str(), "r", stdin);
  std::freopen(output.c_str(), "w", stdout);

  size_t classes_number;
  std::cin >> classes_number;
  std::vector<std::vector<size_t>> confusion_matrix(classes_number);
  std::vector<size_t> rows_sum(classes_number);
  std::vector<size_t> columns_sum(classes_number);
  size_t all_sum = 0;
  for (size_t i = 0; i < classes_number; ++i) {
    confusion_matrix.reserve(classes_number);
    for (size_t j = 0; j < classes_number; ++j) {
      size_t value;
      std::cin >> value;
      confusion_matrix[i].push_back(value);
      rows_sum[i] += value;
      columns_sum[j] += value;
      all_sum += value;
    }
  }

  double micro_f = 0.0;
  double average_precision = 0.0;
  double average_recall = 0.0;
  for (size_t class_index = 0; class_index < classes_number; ++class_index) {
    size_t true_positive = confusion_matrix[class_index][class_index];
    size_t false_positive = rows_sum[class_index] - confusion_matrix[class_index][class_index];
    size_t false_negative = columns_sum[class_index] - confusion_matrix[class_index][class_index];

    double precision =
        true_positive + false_positive == 0 ? 0 : (double) true_positive / (double) (true_positive + false_positive);
    double recall =
        true_positive + false_negative == 0 ? 0 : (double) true_positive / (double) (true_positive + false_negative);

    micro_f += precision + recall == 0 ? 0 : rows_sum[class_index] * 2.0 * precision * recall / (precision + recall);
    average_precision += rows_sum[class_index] * precision;
    average_recall += rows_sum[class_index] * recall;
  }

  double macro_f = 2.0 * average_precision * average_recall / (average_precision + average_recall) / all_sum;
  micro_f /= all_sum;

  std::cout << macro_f;

  std::fclose(stdin);
  std::fclose(stdout);
}
int main(int arg, char *argv[]){    
   //test_grad();
   //
  //  test_acc_compute();
  // test_emb_all();
  test_average_precision("input.txt","output.txt");
  return 0;
}