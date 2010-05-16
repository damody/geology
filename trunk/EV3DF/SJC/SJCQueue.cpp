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
              
     Contructor:

     Function:
                 
                 >>: input operator
                 <<: output operator
     Compiler:    g++
 
     Platform:    Linux
*************************************************************************/
#include "SJCQueue.h"

#define Swap(a, b, c) ( (c) = (a), (a) = (b), (b) = (c) )

//****************************************************************************
//
// * Clear the data
//=============================================================================
void SJCQueue::
Clear(void)
//=============================================================================
{
  for(uint i = 0; i < entries.size(); i++){
    delete entries[i];
  }
  entries.clear();
  
  
}

//****************************************************************************
//
// * Indicate that we add a leaf to this tree and we need to process it to the
//   top to maintain the priority queue
//=============================================================================
void SJCQueue::
addLeaf(int i)
//=============================================================================
{
  SJCQueueEntry   *e;
  int	    	   j;
  
  e = entries[i];
  j = ( i + 1 ) / 2 - 1;
  while ( ( j >= 0 ) && ( entries[j]->k > e->k ) )   {
    // do the swapping
    entries[i] = entries[j];
    entries[i]->back = i;
    i = j;
    j = ( i + 1 ) / 2 - 1;
  }
  entries[i] = e;
  entries[i]->back = i;
}

//****************************************************************************
//
// * add a node to this subtree root and we need to process it to proper
//   Position 
//=============================================================================
void SJCQueue::
addRoot(int i)
//=============================================================================
{
  int   j;
  
  j = 2 * ( i + 1 ) - 1;
  
  if ( j < (int)entries.size() )   {
    if ( ( j < ((int)entries.size()) - 1 )
	 && ( entries[j]->k > entries[j+1]->k ) )
      j++;
    if ( entries[i]->k > entries[j]->k ){
      SJCQueueEntry   *temp;
      Swap(entries[i], entries[j], temp);
      entries[i]->back = i;
      entries[j]->back = j;
      // Recursively process down
      addRoot(j);
    }// end of if
  }// end of if
}

//****************************************************************************
//
// Insert an entry
//=============================================================================
SJCQueueEntry* SJCQueue::
insert(SJCQueueKey key, void *value)
//=============================================================================
{
  SJCQueueEntry   *res = new SJCQueueEntry(key, value);
  
  res->back = entries.size();
  
  entries.push_back(res);
  addLeaf(entries.size() - 1);
  
  return res;
}

//****************************************************************************
//
// Remove the minimum entry
//=============================================================================
SJCQueueEntry* SJCQueue::
removeMin(void)
//=============================================================================
{
  SJCQueueEntry   *e;

  
  if ( empty() ) 
    return 0;
  
  e = entries[0];
  
  if ( entries.size() > 1 )    {
    entries[0] = entries[entries.size() - 1];
    entries[0]->back = 0;
    entries.pop_back();
    addRoot(0);
  }
  else
    entries.pop_back();
  
  e->back = -1;
  
  return e;
}

//****************************************************************************
//
// Remove the entry from the queue
//=============================================================================
SJCQueueEntry* SJCQueue::
remove(SJCQueueEntry *e)
//=============================================================================
{
  int	i;
  
  if ( e->back < (int)entries.size() - 1 )   {
    i = e->back;
    entries[i] = entries[entries.size() - 1];
    entries[i]->back = i;
    entries.pop_back();
    addRoot(i);
    addLeaf(i);
  }
  else
    entries.pop_back();
  
  e->back = -1;
  
  return e;
}

//****************************************************************************
//
// Change the key value
//=============================================================================
void SJCQueue::
change(SJCQueueEntry *e, SJCQueueKey k)
//=============================================================================
{
  int	i = e->back;

  e->k = k;
  addRoot(i);
  addLeaf(i);
}


