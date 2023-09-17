#ifndef NEUTRONSTREAM_LOCKER_H
#define NEUTRONSTREAM_LOCKER_H
#include <thread>
#include <mutex>
#include "util.h"
namespace neutron
{
  
   template<class T>
    class ScopedLockImpl{
    public:
        ScopedLockImpl(T &mutex):m_mutex(mutex){
            m_mutex.lock();
            m_locked=true;
        }
        ~ScopedLockImpl(){
            unlock();
        }
        void lock() {
            if(!m_locked) {
                m_mutex.lock();
                m_locked = true;
            }
        }
        void unlock() {
            if(m_locked) {
                m_mutex.unlock();
                m_locked = false;
            }
        }
    private:
        T &m_mutex;
        bool m_locked;
    };
  /**
   * @brief 局部读模板的实现
   * 
   * @tparam T 
   */
  template<class T>
  struct ReadScopedLockImpl{
        ReadScopedLockImpl(T &mutex): 
        m_mutex(mutex) {
            m_mutex.rdlock();
            m_locked = true;
        }
        ~ReadScopedLockImpl() { 
            unlock(); 
        }
        void lock() {
            if(!m_locked){
                m_mutex.rdlock();
                m_locked = true;
            }
        }
        
        void unlock() {
            if(m_locked){
                m_mutex.unlock();
                m_locked = false;
            }
        }
        
    private:
        T& m_mutex;
        bool m_locked;
    };
    
  /**
   * @brief 局部写锁模板的实现
   * 
   * @tparam T 
   */
  template<class T>
  class WriteScopedLockImpl{
    public:
        WriteScopedLockImpl(T & mutex):m_mutex(mutex){
            m_mutex.wrlock();
            m_locked = true;
        }
        ~WriteScopedLockImpl() {
            unlock();
        }
        void lock(){
            if(!m_locked){
                m_mutex.wrlock();
                m_locked= true;
            }
        }
        void unlock(){  
            if(m_locked){
                m_mutex.unlock();
                m_locked =false;
            }
        }
        
    private:
        T& m_mutex;
        bool m_locked;
  };
  
  class Spinlock{
    public:
        /// 局部锁
        typedef ScopedLockImpl<Spinlock> Lock; 
        
        Spinlock(){
            pthread_spin_init(&m_mutex, 0);
        }
        ~Spinlock(){
            pthread_spin_destroy(&m_mutex);
        }
        void lock() {
            pthread_spin_lock(&m_mutex);
        }
        void unlock() {
            pthread_spin_unlock(&m_mutex);
        }
        
    private:
        pthread_spinlock_t m_mutex;
  };
  /**
   * @brief 读写互斥量
   */
  class RWMutex : public Noncopyable{
  public:

    /// 局部读锁
    typedef ReadScopedLockImpl<RWMutex> ReadLock;

    /// 局部写锁
    typedef WriteScopedLockImpl<RWMutex> WriteLock;

    /**
     * @brief 构造函数
     */
    RWMutex() {
        pthread_rwlock_init(&m_lock, nullptr);
    }
    
    /**
     * @brief 析构函数
     */
    ~RWMutex() {
        pthread_rwlock_destroy(&m_lock);
    }

    /**
     * @brief 上读锁
     */
    void rdlock() {
        pthread_rwlock_rdlock(&m_lock);
    }

    /**
     * @brief 上写锁
     */
    void wrlock() {
        pthread_rwlock_wrlock(&m_lock);
    }

    /**
     * @brief 解锁
     */
    void unlock() {
        pthread_rwlock_unlock(&m_lock);
    }
private:
    /// 读写锁
    pthread_rwlock_t m_lock;
};
} //namespace neutron
#endif