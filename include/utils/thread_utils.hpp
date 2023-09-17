#ifndef NEUTRONSTREAM_THREAD_UTILS_H
#define NEUTRONSTREAM_THREAD_UTILS_H
#include <thread>
#include <mutex>
#include <future>
#include <utils/thread_safe/thread_safe_queue.hpp>
#include <utils/util.h>
#include <iostream>
namespace neutron
{

  //RAII
  class thread_guard{
    std::thread &t;
    public:
      explicit thread_guard(std::thread& t_): t(t_) {}
      ~thread_guard() { 
        if(t.joinable()){ 
          t.join(); 
        } 
      }
      thread_guard(thread_guard const &)=delete;
      thread_guard &operator=(thread_guard const &) =delete;
  };
  // class join_threads{
  //   std::vector<std::thread> &threads;
  //   public:
  //     explicit join_threads(std::vector<std::thread> &thread_):
  //     threads(thread_){}
  //     ~join_threads(){
  //       for(auto &t:threads){
  //         if(t.joinable()){
  //           t.join();
  //         }
  //       }
  //     }
  // };

  class function_wrapper:Noncopyable{
    struct impl_base { 
      virtual void call()=0; 
      virtual ~impl_base() {} 
    };
    std::unique_ptr<impl_base> impl; 
    template<typename F> 
    struct impl_type: impl_base{
      F f; 
      impl_type(F&& f_): f(std::move(f_)) {} 
      void call() { f(); }
    };
    public:
      function_wrapper() = default;
      template<typename F>
      function_wrapper(F&& f):
      impl(new impl_type<F>(std::move(f))) {}

      void operator()() { 
        if(impl) 
          impl->call(); 
      }

      function_wrapper(function_wrapper&& other): 
      impl(std::move(other.impl)) {}

      function_wrapper& operator=(function_wrapper&& other) {
        
        impl=std::move(other.impl); 
        return *this; 
      }
      bool isNullTask() const {
        return impl ? false : true;
    }
  };

  class work_stealing_queue{
    private:
      typedef function_wrapper data_type;
      std::deque<data_type> m_queue;
      mutable std::mutex m_mutex;
    public:
      work_stealing_queue(){};
      work_stealing_queue(const work_stealing_queue& other)=delete;
      work_stealing_queue& operator=( const work_stealing_queue& other)=delete;

    /** 将数据插入队列头部 */
    void push_front(data_type data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push_front(std::move(data));
    }

    /** 将数据插入队列尾部 */
    void push_back(data_type data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push_back(std::move(data));
    }
    bool empty() const {
      std::lock_guard<std::mutex> lock(m_mutex); 
      return m_queue.empty();
    }

      /** 队列大小，未加锁 */
    size_t size() const {
        return m_queue.size();
    }
      bool try_pop(data_type &res){
        std::lock_guard<std::mutex> lock(m_mutex); 
        if(m_queue.empty()){
          return false;
        }
        res=std::move(m_queue.front()); 
        m_queue.pop_front(); 
        return true;
      }
      bool try_steal(data_type &res){
        std::lock_guard<std::mutex> lock(m_mutex); 
        if(m_queue.empty()){
          return false;
        }
        res=std::move(m_queue.back()); 
        m_queue.pop_back(); 
        return true;
      }
    
  };
  
  class thread_pool_with_steal{
  public:
    typedef std::shared_ptr<thread_pool_with_steal> ptr;
    thread_pool_with_steal():thread_pool_with_steal(std::thread::hardware_concurrency()){}
    explicit thread_pool_with_steal(size_t thread_count,bool util_empty=true):
    m_thread_count(thread_count),m_done(false),m_runnging_util_empty(util_empty){
      try{
        // 先初始化相关资源，再启动线程
        for(size_t i=0;i<thread_count;i++){
          m_threads_status.push_back(nullptr);
          m_queues.push_back(std::unique_ptr<work_stealing_queue>(new work_stealing_queue())); 
        }
        for(size_t i=0;i<thread_count;i++){
          m_threads.push_back(std::thread(&thread_pool_with_steal::worker_thread,this,i));
        }
      }catch(...){
        m_done=true;
        throw;
      }
    }
    ~thread_pool_with_steal(){
      if(!m_done){
        join();
      }
    }
      /** 获取工作线程数 */
    size_t worker_num() const {
        return m_thread_count;
    }

    /** 先线程池提交任务后返回的对应 future 的类型 */

    template <typename FunctionType>
    std::future<typename std::result_of<FunctionType()>::type> submit(FunctionType f) {
        if (m_thread_need_stop || m_done) {
            throw std::logic_error("Can't submit a task to the stopped StealThreadPool!");
        }
        typedef typename std::result_of<FunctionType()>::type result_type;
        std::packaged_task<result_type()> task(f);
        std::future<result_type> res(task.get_future());
        if (m_local_work_queue) {
            // 本地线程任务从前部入队列（递归成栈）
            m_local_work_queue->push_front(std::move(task));
        } else {
            m_master_work_queue.push(std::move(task));
        }
        m_cv.notify_one();
        return res;
    }
    /** 返回线程池结束状态 */
    bool done() const {
        return m_done;
    }
        /**
     * 等待各线程完成当前执行的任务后立即结束退出
     */
    void stop() {
        if (m_done) {
            return;
        }

        m_done = true;

        // 同时加入结束任务指示，以便在dll退出时也能够终止
        for (size_t i = 0; i < m_thread_count; i++) {
            if (m_threads_status[i]) {
                m_threads_status[i]->store(true);
            }
            m_queues[i]->push_front(function_wrapper());
        }

        m_cv.notify_all();  // 唤醒所有工作线程

        for (size_t i = 0; i < m_thread_count; i++) {
            if (m_threads[i].joinable()) {
                m_threads[i].join();
            }
        }

        printf("Quit StealThreadPool\n");
    }

    /**
     * 等待并阻塞至线程池内所有任务完成
     * @note 至此线程池能工作线程结束不可再使用
     */
    void join() {
        if (m_done) {
            return;
        }

        // 指示各工作线程在未获取到工作任务时，停止运行
        if (m_runnging_util_empty) {
            while (m_master_work_queue.size() != 0) {
                std::this_thread::yield();
            }
            for (size_t i = 0; i < m_thread_count; i++) {
                while (m_queues[i]->size() != 0) {
                    std::this_thread::yield();
                }
                if (m_threads_status[i]) {
                    m_threads_status[i]->store(true);
                }
            }
            m_done = true;
        }

        for (size_t i = 0; i < m_thread_count; i++) {
            m_master_work_queue.push(std::move(function_wrapper()));
        }

        // 唤醒所有工作线程
        m_cv.notify_all();

        // 等待线程结束
        for (size_t i = 0; i < m_thread_count; i++) {
            if (m_threads[i].joinable()) {
                m_threads[i].join();
            }
        }

        m_done = true;
    }
    private:
      typedef function_wrapper task_type;
      size_t m_thread_count;
      std::atomic_bool m_done; // 线程池全局需终止指示
      bool m_runnging_util_empty;    // 运行直到队列空时停止
      std::condition_variable m_cv;  // 信号量，无任务时阻塞线程并等待
      std::mutex m_cv_mutex;         // 配合信号量的互斥量
      std::vector<std::atomic_bool*> m_threads_status;         // 工作线程状态
      threadsafe::threadsafe_queue<task_type> m_master_work_queue;          // 主线程任务队列
      std::vector<std::unique_ptr<work_stealing_queue> > m_queues;  // 任务队列（每个工作线程一个）
      std::vector<std::thread> m_threads;                      // 工作线程

      // join_threads m_joiner;

          // 线程本地变量
      inline static thread_local work_stealing_queue* m_local_work_queue = nullptr;  // 本地任务队列
      inline static thread_local size_t m_index = -1;  //在线程池中的序号
      inline static thread_local std::atomic_bool m_thread_need_stop {false};  // 线程停止运行指示

      void worker_thread(int index) {
        m_index = index;
        m_thread_need_stop = false;
        m_threads_status[index] = &m_thread_need_stop;
        m_local_work_queue = m_queues[m_index].get();
        while (!m_thread_need_stop && !m_done) {
            run_pending_task();
        }
        m_threads_status[m_index] = nullptr;
      }
      void run_pending_task() {
        // 从本地队列提前工作任务，如本地无任务则从主队列中提取任务
        // 如果主队列中提取的任务是空任务，则认为需结束本线程，否则从其他工作队列中偷取任务
        task_type task;
        if (pop_task_from_local_queue(task)) {
            task();
            // std::this_thread::yield();
        } else if (pop_task_from_master_queue(task)) {
            if (!task.isNullTask()) {
                task();
                // std::this_thread::yield();
            } else {
                m_thread_need_stop = true;
            }
        } else if (pop_task_from_other_thread_queue(task)) {
            task();
            // std::this_thread::yield();
        } else {
            // std::this_thread::yield();
            std::unique_lock<std::mutex> lk(m_cv_mutex);
            m_cv.wait(lk, [=] { return this->m_done || !this->m_master_work_queue.empty(); });
        }

    }
    bool pop_task_from_master_queue(task_type& task) {
        return m_master_work_queue.try_pop(task);
    }
    bool pop_task_from_local_queue(task_type& task) {
        return m_local_work_queue && m_local_work_queue->try_pop(task);
    }
    bool pop_task_from_other_thread_queue(task_type& task) {
        for (size_t i = 0; i < m_thread_count; ++i) {
            size_t index = (m_index + i + 1) % m_thread_count;
            if (index != m_index && m_queues[index]->try_steal(task)) {
                return true;
            }
        }
        return false;
    }
  };

  using ThreadTask = std::function<void()>;

  template <typename T, typename... Args>
  static inline std::shared_ptr<T> MakeShared(Args &&... args) {
    typedef typename std::remove_const<T>::type T_nc;
    std::shared_ptr<T> ret(new (std::nothrow) T_nc(std::forward<Args>(args)...));
    return ret;
  }

  class ThreadPool {
  public:
    typedef std::shared_ptr<ThreadPool> ptr;
      explicit ThreadPool(uint32_t size = INT_MAX){
        idle_thrd_num_ = size < 1 ? 1 : size;
        if(idle_thrd_num_>std::thread::hardware_concurrency()){
            idle_thrd_num_=std::thread::hardware_concurrency();
        }
        for (uint32_t i = 0; i < idle_thrd_num_; ++i) {
            pool_.emplace_back(ThreadFunc, this);
        }
        is_stoped_.store(false);
      }

      ~ThreadPool(){
        is_stoped_.store(true);
          {
            std::unique_lock<std::mutex> lock{m_lock_};
            cond_var_.notify_all();
          }
        for (std::thread &thd : pool_) {
            if (thd.joinable()) {
                try {
                    thd.join();
                } catch (const std::system_error &) {
                    std::cout<<"system_error"<<std::endl;
                } catch (...) {
                    std::cout<<"exception"<<std::endl;
                }
            }
        }
      }
      template <class Func, class... Args>
      auto submit(Func &&func, Args &&... args) -> std::future<decltype(func(args...))> {
//            std::cout<<"commit run task enter."<<std::endl;
          using retType = decltype(func(args...));
          std::future<retType> fail_future;
          if (is_stoped_.load()) {
              std::cout<<"FAILED "<<"thread pool has been stopped."<<std::endl;
              return fail_future;
          }
          auto bindFunc = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
          auto task = MakeShared<std::packaged_task<retType()>>(bindFunc);
          if (task == nullptr) {
              std::cout<<"FAILED"<<"Make shared failed."<<std::endl;
              return fail_future;
          }
          std::future<retType> future = task->get_future();
          {
              std::lock_guard<std::mutex> lock{m_lock_};
              tasks_.emplace([task]() { (*task)(); });
          }
          cond_var_.notify_one();
//            std::cout<<"commit run task end"<<std::endl;
          return future;
      }

      uint32_t GetRunThreadNum(){return idle_thrd_num_.load();}

      static void ThreadFunc(ThreadPool *thread_pool){
        if (thread_pool == nullptr) {
            return;
        }
        while (!thread_pool->is_stoped_) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock{thread_pool->m_lock_};
                thread_pool->cond_var_.wait(
                        lock, [thread_pool] { return thread_pool->is_stoped_.load() || !thread_pool->tasks_.empty(); });
                if (thread_pool->is_stoped_ && thread_pool->tasks_.empty()) {
                    return;
                }
                task = std::move(thread_pool->tasks_.front());
                thread_pool->tasks_.pop();
            }
            --thread_pool->idle_thrd_num_;
            task();
            ++thread_pool->idle_thrd_num_;
        }
      }

      std::vector<std::thread::id> getThreadIDs(){
          std::vector<std::thread::id> rst;
          for(auto &t:pool_){
              rst.emplace_back(t.get_id());
          }
          return rst;
      };

  private:

      std::vector<std::thread> pool_;
      std::queue<ThreadTask> tasks_;
      std::mutex m_lock_;
      std::condition_variable cond_var_;
      std::atomic<bool> is_stoped_;
      std::atomic<uint32_t> idle_thrd_num_{};
  };
  




} // namespace neutron

#endif