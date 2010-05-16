/************************************************************************
     Main File:
 
     File:        SJCQueue.h
 
     Author:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
   
     Comment:     This class implement the priority queue
                  1. SJCQueueKey: implement the key and comparison which should
                     be implemented with template
                  2. SJCQueueEntry: implement the entries with the void* as the
                     data
                  3. SJCQueue: the priority queue structure
                     => Array structure to move up and down, parent is 
                        (i + 1) / 2 - 1

  *   Code derived from:
      Jeffrey H. Kingston,
      "Algorithms and Data Structures: Design, Correctness, Analysis",
       Addison Wesley, 1990            
     Contructor:

     Function:
          
                 >>: input operator
                 <<: output operator
     Compiler:    g++
 
     Platform:    Linux
*************************************************************************/

#ifndef _SJCQUEUE_H_
#define _SJCQUEUE_H_

#include <Globals.h>

#include <vector>

using namespace std;

class SJCQueueKey {
 private:
  double  	  p;
  unsigned char   s;
  
 public:
  SJCQueueKey(const double k, const unsigned char sk) {
    p = k; s = sk;
  }
  SJCQueueKey(const SJCQueueKey &k) {
    p = k.p; s = k.s;
  }
  
  double  	  primary(void) const { return p; }
  unsigned char   secondary(void) const { return s; }
  
  bool operator<(const SJCQueueKey &b) const {
    return ( p < b.p ) || ( ( p == b.p ) && ( s < b.s ) );
  }
  bool operator<=(const SJCQueueKey &b) const {
    return ( p < b.p ) || ( ( p == b.p ) && ( s <= b.s ) );
  }
  bool operator>(const SJCQueueKey &b) const {
    return ( p > b.p ) || ( ( p == b.p ) && ( s > b.s ) );
  }
  bool operator>=(const SJCQueueKey &b) const {
    return ( p > b.p ) || ( ( p == b.p ) && ( s >= b.s ) );
  }
  bool operator==(const SJCQueueKey &b) const {
    return ( p == b.p ) && ( s == b.s );
  }
};


class SJCQueueEntry {
  friend class SJCQueue;

 private:
  SJCQueueKey	k;       // key for the entry
  void    	*v;      // The loading data 
  int	    	back;    // pointer to the parent 
  
  SJCQueueEntry(SJCQueueKey key, void *d) : k(key) { v = d; }
  
 public:
  SJCQueueKey	key(void) const { return k; }
  void*	        value(void) const { return v; }
  
};

//***************************************************************************
// Only clear the entry
// User must clear the entry's data
//***************************************************************************

class SJCQueue {
 private:
  vector<SJCQueueEntry*>  entries;
  
  void    addLeaf(int i);
  void    addRoot(int i);
  
 public:
  // Constructor
  SJCQueue(void) : entries() { entries.reserve(128); }
  SJCQueue(const int max_size) : entries() { entries.reserve(max_size); }

  void Clear(void);
  
  // Check whether the entries is empty
  bool    	  empty(void) const { return ! entries.size(); }
  // Return the entry with minimum value
  SJCQueueEntry*  min(void) { return empty() ? 0 : entries[0]; }
  // Insert an entry
  SJCQueueEntry*  insert(SJCQueueKey key, void *value);
  // Remove the minimum from the queue
  SJCQueueEntry*  removeMin(void);
  // Remove the entry from the queue
  SJCQueueEntry*  remove(SJCQueueEntry *e);
  // Check the entry's key with k
  void change(SJCQueueEntry *e, SJCQueueKey k);
};


#endif
