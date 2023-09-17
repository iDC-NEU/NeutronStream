/*
Thread Safe Version STL in C++11
Copyright(c) 2021
Author: tashaxing
*/
#ifndef THREAD_SAFE_MAP_H_INCLUDED
#define THREAD_SAFE_MAP_H_INCLUDED

#include <map>
#include <utils/locker.h>

namespace thread_safe {

template < class Key, class T, class Compare = std::less<Key>, class Allocator = std::allocator<std::pair<const Key,T> > >
class map {
public:
    typedef typename std::map<Key, T, Compare, Allocator>::iterator iterator;
    typedef typename std::map<Key, T, Compare, Allocator>::const_iterator const_iterator;
    typedef typename std::map<Key, T, Compare, Allocator>::reverse_iterator reverse_iterator;
    typedef typename std::map<Key, T, Compare, Allocator>::const_reverse_iterator const_reverse_iterator;
    typedef typename std::map<Key, T, Compare, Allocator>::allocator_type allocator_type;
    typedef typename std::map<Key, T, Compare, Allocator>::size_type size_type;
    typedef typename std::map<Key, T, Compare, Allocator>::key_compare key_compare;
    typedef typename std::map<Key, T, Compare, Allocator>::value_compare value_compare;
    typedef typename std::map<Key, T, Compare, Allocator>::value_type value_type;
    typedef neutron::Spinlock MutexType;
    // Constructors
    explicit map ( const Compare& comp = Compare(), const Allocator & alloc = Allocator() ) : storage( comp, alloc ) { }
    template <class InputIterator> map ( InputIterator first, InputIterator last, const Compare& comp = Compare(), const Allocator & alloc = Allocator() ) : storage( first, last, comp, alloc ) { }
    map( const thread_safe::map<Key, T, Compare, Allocator> & x ) : storage( x.storage ) { }

    // Copy
    thread_safe::map<Key, Compare, Allocator> & operator=( const thread_safe::map<Key,Compare,Allocator> & x ) { MutexType::Lock lock(m_mutex); std::lock_guard<std::mutex> lock2( x.mutex ); storage = x.storage; return *this; }

    // Destructor
    ~map( void ) { }

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

    // Element Access
    T & operator[]( const Key & x ) { MutexType::Lock lock(m_mutex); return storage[x]; }

    // Modifiers
    std::pair<iterator, bool> insert( const value_type & x ) { MutexType::Lock lock(m_mutex); return storage.insert( x ); }
    iterator insert( iterator position, const value_type & x ) { MutexType::Lock lock(m_mutex); return storage.insert( position, x ); }
    template <class InputIterator> void insert( InputIterator first, InputIterator last ) { MutexType::Lock lock(m_mutex); storage.insert( first, last ); }

    void erase( iterator pos ) { MutexType::Lock lock(m_mutex); storage.erase( pos ); }
    size_type erase( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.erase( x ); }
    void erase( iterator begin, iterator end ) { MutexType::Lock lock(m_mutex); storage.erase( begin, end ); }

    void swap( thread_safe::map<Key, T, Compare, Allocator> & x ) { MutexType::Lock lock(m_mutex); std::lock_guard<std::mutex> lock2( x.mutex ); storage.swap( x.storage ); }

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
    std::map<Key, T, Compare, Allocator> storage;
    mutable MutexType m_mutex;
};

template < class Key, class T, class Compare = std::less<Key>, class Allocator = std::allocator<std::pair<const Key,T> > >
class multimap {
public:
    typedef typename std::multimap<Key, T, Compare, Allocator>::iterator iterator;
    typedef typename std::multimap<Key, T, Compare, Allocator>::const_iterator const_iterator;
    typedef typename std::multimap<Key, T, Compare, Allocator>::reverse_iterator reverse_iterator;
    typedef typename std::multimap<Key, T, Compare, Allocator>::const_reverse_iterator const_reverse_iterator;
    typedef typename std::multimap<Key, T, Compare, Allocator>::allocator_type allocator_type;
    typedef typename std::multimap<Key, T, Compare, Allocator>::size_type size_type;
    typedef typename std::multimap<Key, T, Compare, Allocator>::key_compare key_compare;
    typedef typename std::multimap<Key, T, Compare, Allocator>::value_compare value_compare;
    typedef typename std::multimap<Key, T, Compare, Allocator>::value_type value_type;
    typedef neutron::Spinlock MutexType;
    // Constructors
    explicit multimap ( const Compare& comp = Compare(), const Allocator & alloc = Allocator() ) : storage( comp, alloc ) { }
    template <class InputIterator> multimap ( InputIterator first, InputIterator last, const Compare& comp = Compare(), const Allocator & alloc = Allocator() ) : storage( first, last, comp, alloc ) { }
    multimap( const thread_safe::multimap<Key, T, Compare, Allocator> & x ) : storage( x.storage ) { }

    // Copy
    thread_safe::multimap<Key, Compare, Allocator> & operator=( const thread_safe::multimap<Key,Compare,Allocator> & x ) { MutexType::Lock lock(m_mutex); std::lock_guard<std::mutex> lock2( x.mutex ); storage = x.storage; return *this; }

    // Destructor
    ~multimap( void ) { }

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
    std::pair<iterator, bool> insert( const value_type & x ) { MutexType::Lock lock(m_mutex); return storage.insert( x ); }
    iterator insert( iterator position, const value_type & x ) { MutexType::Lock lock(m_mutex); return storage.insert( position, x ); }
    template <class InputIterator> void insert( InputIterator first, InputIterator last ) { MutexType::Lock lock(m_mutex); storage.insert( first, last ); }

    void erase( iterator pos ) { MutexType::Lock lock(m_mutex); storage.erase( pos ); }
    size_type erase( const Key & x ) { MutexType::Lock lock(m_mutex); return storage.erase( x ); }
    void erase( iterator begin, iterator end ) { MutexType::Lock lock(m_mutex); storage.erase( begin, end ); }

    void swap( thread_safe::multimap<Key, T, Compare, Allocator> & x ) { MutexType::Lock lock(m_mutex); std::lock_guard<std::mutex> lock2( x.mutex ); storage.swap( x.storage ); }

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
    std::multimap<Key, T, Compare, Allocator> storage;
    mutable MutexType m_mutex;
};


}

#endif // THREAD_SAFE_MAP_H_INCLUDED
