/*
Thread Safe Version STL in C++11
Copyright(c) 2021
Author: tashaxing
*/
#ifndef THREAD_SAFE_SET_H_INCLUDED
#define THREAD_SAFE_SET_H_INCLUDED

#include <set>
#include <utils/locker.h>

namespace thread_safe {

template < class Key, class Compare = std::less<Key>, class Allocator = std::allocator<Key> >
class set {
public:
    typedef typename std::set<Key, Compare, Allocator>::iterator iterator;
    typedef typename std::set<Key, Compare, Allocator>::const_iterator const_iterator;
    typedef typename std::set<Key, Compare, Allocator>::reverse_iterator reverse_iterator;
    typedef typename std::set<Key, Compare, Allocator>::const_reverse_iterator const_reverse_iterator;
    typedef typename std::set<Key, Compare, Allocator>::allocator_type allocator_type;
    typedef typename std::set<Key, Compare, Allocator>::size_type size_type;
    typedef typename std::set<Key, Compare, Allocator>::key_compare key_compare;
    typedef typename std::set<Key, Compare, Allocator>::value_compare value_compare;
    typedef neutron::Spinlock MutexType;
    // Constructors
    explicit set ( const Compare& comp = Compare(), const Allocator & alloc = Allocator() ) : storage( comp, alloc ) { }
    template <class InputIterator> set ( InputIterator first, InputIterator last, const Compare& comp = Compare(), const Allocator & alloc = Allocator() ) : storage( first, last, comp, alloc ) { }
    set( const thread_safe::set<Key, Compare, Allocator> & x ) : storage( x.storage ) { }

    // Copy
    thread_safe::set<Key, Compare, Allocator> & operator=( const thread_safe::set<Key,Compare,Allocator> & x ) { MutexType::Lock lock(m_mutex); std::lock_guard<std::mutex> lock2( x.mutex ); storage = x.storage; return *this; }

    // Destructor
    ~set( void ) { }

    // Iterators
    iterator begin( void ) { MutexType::Lock lock(m_mutex); return storage.begin(); }
    const_iterator begin( void ) const { MutexType::Lock lock(m_mutex); return storage.begin(); }

    iterator end( void ) { MutexType::Lock lock(m_mutex); return storage.end(); }
    const_iterator end( void ) const { MutexType::Lock lock(m_mutex); return storage.end(); }

    reverse_iterator rbegin( void ) { MutexType::Lock lock(m_mutex); return storage.rbegin(); }
    const_reverse_iterator rbegin( void ) const { MutexType::Lock lock(m_mutex); return storage.rbegin(); }

    reverse_iterator rend( void ) { MutexType::Lock lock(m_mutex); return storage.rend(); }
    const_reverse_iterator rend( void ) const { MutexType::Lock lock(m_mutex); return storage.rend(); }

    // Capacity
    size_type size( void ) const { MutexType::Lock lock(m_mutex); return storage.size(); }

    size_type max_size( void ) const { MutexType::Lock lock(m_mutex); return storage.max_size(); }

    bool empty( void ) const { MutexType::Lock lock(m_mutex); return storage.empty(); }

    // Modifiers
    std::pair<iterator, bool> insert( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.insert( x ); }
    iterator insert( iterator position, const Key & x ) { MutexType::Lock lock(m_mutex); return storage.insert( position, x ); }
    template <class InputIterator> void insert( InputIterator first, InputIterator last ) { MutexType::Lock lock(m_mutex); storage.insert( first, last ); }

    void erase( iterator pos ) { MutexType::Lock lock(m_mutex); storage.erase( pos ); }
    size_type erase( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.erase( x ); }
    void erase( iterator begin, iterator end ) { MutexType::Lock lock(m_mutex); storage.erase( begin, end ); }

    void swap( thread_safe::set<Key, Compare, Allocator> & x ) { MutexType::Lock lock(m_mutex); std::lock_guard<std::mutex> lock2( x.mutex ); storage.swap( x.storage ); }

    void clear( void ) { MutexType::Lock lock(m_mutex); storage.clear(); }

    // Observers
    key_compare key_comp( void ) const { MutexType::Lock lock(m_mutex); return storage.key_comp(); }
    value_compare value_comp( void ) const { MutexType::Lock lock(m_mutex); return storage.value_comp(); }

    // Operations
    const_iterator find( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.find( x ); }
    iterator find( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.find( x ); }

    size_type count( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.count( x ); }

    const_iterator lower_bound( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.lower_bound( x ); }
    iterator lower_bound( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.lower_bound( x ); }

    const_iterator upper_bound( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.upper_bound( x ); }
    iterator upper_bound( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.upper_bound( x ); }

    std::pair<const_iterator,const_iterator> equal_range( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.equal_range( x ); }
    std::pair<iterator,iterator> equal_range( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.equal_range( x ); }

    // Allocator
    allocator_type get_allocator( void ) const { MutexType::Lock lock(m_mutex); return storage.get_allocator(); }

private:
    std::set< Key, Compare, Allocator > storage;
    mutable MutexType m_mutex;
};

template < class Key, class Compare = std::less<Key>, class Allocator = std::allocator<Key> >
class multiset {
public:
    typedef typename std::multiset<Key, Compare, Allocator>::iterator iterator;
    typedef typename std::multiset<Key, Compare, Allocator>::const_iterator const_iterator;
    typedef typename std::multiset<Key, Compare, Allocator>::reverse_iterator reverse_iterator;
    typedef typename std::multiset<Key, Compare, Allocator>::const_reverse_iterator const_reverse_iterator;
    typedef typename std::multiset<Key, Compare, Allocator>::allocator_type allocator_type;
    typedef typename std::multiset<Key, Compare, Allocator>::size_type size_type;
    typedef typename std::multiset<Key, Compare, Allocator>::key_compare key_compare;
    typedef typename std::multiset<Key, Compare, Allocator>::value_compare value_compare;
    typedef neutron::Spinlock MutexType;
    // Constructors
    explicit multiset ( const Compare& comp = Compare(), const Allocator & alloc = Allocator() ) : storage( comp, alloc ) { }
    template <class InputIterator>multiset ( InputIterator first, InputIterator last, const Compare& comp = Compare(), const Allocator & alloc = Allocator() ) : storage( first, last, comp, alloc ) { }
    multiset( const thread_safe::multiset<Key, Compare, Allocator> & x ) : storage( x.storage ) { }

    // Copy
    thread_safe::multiset<Key, Compare, Allocator> & operator=( const thread_safe::multiset<Key,Compare,Allocator> & x ) { MutexType::Lock lock(m_mutex); std::lock_guard<std::mutex> lock2( x.mutex ); storage = x.storage; return *this; }

    // Destructor
    ~multiset( void ) { }

    // Iterators
    iterator begin( void ) { MutexType::Lock lock(m_mutex); return storage.begin(); }
    const_iterator begin( void ) const { MutexType::Lock lock(m_mutex); return storage.begin(); }

    iterator end( void ) { MutexType::Lock lock(m_mutex); return storage.end(); }
    const_iterator end( void ) const { MutexType::Lock lock(m_mutex); return storage.end(); }

    reverse_iterator rbegin( void ) { MutexType::Lock lock(m_mutex); return storage.rbegin(); }
    const_reverse_iterator rbegin( void ) const { MutexType::Lock lock(m_mutex); return storage.rbegin(); }

    reverse_iterator rend( void ) { MutexType::Lock lock(m_mutex); return storage.rend(); }
    const_reverse_iterator rend( void ) const { MutexType::Lock lock(m_mutex); return storage.rend(); }

    // Capacity
    size_type size( void ) const { MutexType::Lock lock(m_mutex); return storage.size(); }

    size_type max_size( void ) const { MutexType::Lock lock(m_mutex); return storage.max_size(); }

    bool empty( void ) const { MutexType::Lock lock(m_mutex); return storage.empty(); }

    void reserve(size_type n) { MutexType::Lock lock(m_mutex); storage.reserve(n); }

    // Modifiers
    std::pair<iterator, bool> insert( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.insert( x ); }
    iterator insert( iterator position, const Key & x ) { MutexType::Lock lock(m_mutex); return storage.insert( position, x ); }
    template <class InputIterator> void insert( InputIterator first, InputIterator last ) { MutexType::Lock lock(m_mutex); storage.insert( first, last ); }

    void erase( iterator pos ) { MutexType::Lock lock(m_mutex); storage.erase( pos ); }
    size_type erase( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.erase( x ); }
    void erase( iterator begin, iterator end ) { MutexType::Lock lock(m_mutex); storage.erase( begin, end ); }

    void swap( thread_safe::multiset<Key, Compare, Allocator> & x ) { MutexType::Lock lock(m_mutex); std::lock_guard<std::mutex> lock2( x.mutex ); storage.swap( x.storage ); }

    void clear( void ) { MutexType::Lock lock(m_mutex); storage.clear(); }

    // Observers
    key_compare key_comp( void ) const { MutexType::Lock lock(m_mutex); return storage.key_comp(); }
    value_compare value_comp( void ) const { MutexType::Lock lock(m_mutex); return storage.value_comp(); }

    // Operations
    const_iterator find( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.find( x ); }
    iterator find( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.find( x ); }

    size_type count( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.count( x ); }

    const_iterator lower_bound( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.lower_bound( x ); }
    iterator lower_bound( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.lower_bound( x ); }

    const_iterator upper_bound( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.upper_bound( x ); }
    iterator upper_bound( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.upper_bound( x ); }

    std::pair<const_iterator,const_iterator> equal_range( const Key & x ) const { MutexType::Lock lock(m_mutex); return storage.equal_range( x ); }
    std::pair<iterator,iterator> equal_range( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.equal_range( x ); }

    // Allocator
    allocator_type get_allocator( void ) const { MutexType::Lock lock(m_mutex); return storage.get_allocator(); }

private:
    std::multiset< Key, Compare, Allocator > storage;
    mutable MutexType m_mutex;
};

}

#endif // THREAD_SAFE_SET_H_INCLUDED
