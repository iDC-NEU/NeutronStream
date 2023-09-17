#ifndef NEUTRONSTREAM_UTIL_H
#define NEUTRONSTREAM_UTIL_H
#include <vector>
#include <omp.h>
#include <torch/torch.h>
#include <utils/mytime.h>
namespace neutron
{
    //工具类
  /**
   * @brief 继承非复制
   * 
   */
  class Noncopyable {
  public:
      /**
       * @brief 默认构造函数
       */
      Noncopyable() = default;

      /**
       * @brief 默认析构函数
       */
      ~Noncopyable() = default;

      /**
       * @brief 拷贝构造函数(禁用)
       */
      Noncopyable(const Noncopyable&) = delete;
      Noncopyable(Noncopyable&) = delete;

      /**
       * @brief 赋值函数(禁用)
       */
      Noncopyable& operator=(const Noncopyable&) = delete;
  }; // class Noncopyable


  template <class T>
  class Singleton{
    public:
      static T* GetInstance(){
        static T v;
        return &v;
      }
  };

  template <class Type>
  torch::Tensor vector2tensor(std::vector<std::vector<Type>> &vec){
    if(vec.size() == 0) return torch::tensor({1});
    int cols=vec[0].size();
    int n=vec.size();
    torch::TensorOptions options;
    if(std::is_same<Type,int>::value){
      options=torch::TensorOptions().dtype(torch::kInt32);
    }
    else if(std::is_same<Type,float>::value || std:: is_same<Type,double>::value){
      options=torch::TensorOptions().dtype(torch::kFloat32);
    }
    else{
      CHECK(false)<<"Type is not supported\n";
    }

    torch::Tensor input_tensor = torch::zeros({ n, cols }, options);
    for (int i = 0; i < n; i++) {
      input_tensor.slice(0, i, i + 1) = torch::from_blob(vec[i].data(), { cols }, options).clone();
    }
    return input_tensor;
  }

  std::vector<torch::Tensor> tensor2vec(torch::Tensor &emb);
  torch::Tensor vec2tensor(const std::vector<torch::Tensor> &vec);

  torch::Tensor times2tensor(std::vector<TimePoint> &times);

  typedef std::tuple<size_t,size_t,std::string,std::string> Evt;
  std::vector<std::vector<std::string>> readCSV(const std::string& filepath);
  std::vector<std::vector<std::string>> readTxt(const std::string &filepath,char delm=',');
  void writeTxt(const std::vector<std::vector<size_t>> &edges,const std::string &path,std::string head="");
  void writeCSV(const std::vector<Evt> &events,
                const std::string &path,std::string head="");

  void writeFeatures(const std::vector<std::vector<int>> &features,const std::string &path);
  template<typename T>
  std::vector<T> &set_diff(std::vector<T> &op1,const std::vector<T> &op2,std::vector<T>&res){

      std::set_difference(op1.begin(),op1.end(),op2.begin(),op2.end(),
                          std::inserter(res,res.begin()));
      return res;
  }
  template<typename T>
  std::vector<T> &set_diff(std::set<T> &op1,const std::set<T> &op2,std::vector<T>&res){

      std::set_difference(op1.begin(),op1.end(),op2.begin(),op2.end(),
                          std::inserter(res,res.begin()));
      return res;
  }
  template<typename T>
  std::ostream& operator<< (std::ostream& out, std::vector<T>& _vec){
    typename std::vector<T>::const_iterator it = _vec.begin();
    out <<"[ ";
    for(; it != _vec.end(); ++it)
            out << *it<<" ";
    out<<"]";
    return out;
  }
  template<typename T>
  torch::Tensor generated_mask(int64_t arrange,std::vector<T> &idx_true){
    torch::Tensor mask_=torch::zeros(arrange).to(torch::kBool);
    #pragma omp parallel for
    for(size_t i=0;i<idx_true.size();i++){
      mask_[idx_true[i]]=true;
    }
    return mask_;
  }

} // namespace neutron

#endif