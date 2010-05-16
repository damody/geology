/************************************************************************
     Main File:

     File:        SJCScalarField2.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
    Comment:     Class to handle the scalar field in 2D

   Constructors:
                  1. 0 : the default contructor
                  2. 6 : constructor to set up all value by input parameters
                  3. 1 : set up the class by using the scalar field
                  4. 1 : copy contructor
                   
     Functions:  what r for?
                 1. = : Assign operator which copy the parameter of random
                 2. (): Get the value of the scalar field
                 3. value: get the value of the scalar field
                 4. grad: get the gradient of the scalar field
                 5. curl: get the curl of the scalar field
                 6. MinX, MinY, MaxX, MaxY: get the maximum and minimum value 
                    of X, y
************************************************************************/
     
#ifndef SJC_MEMORY_H
#define SJC_MEMORY_H

void *AllocAligned(size_t size);
void FreeAligned(void *);
COREDLL void FreeAligned(void *ptr) {
#ifdef WIN32 // NOBOOK
	_aligned_free(ptr);
#else // NOBOOK
	free(ptr);
#endif // NOBOOK
}

//*****************************************************************************
//
// *
//
//*****************************************************************************
template <class T> class ObjectArena {
 public:
  // Contructor
  ObjectArena(void) { m_uNumAbailable = 0; }
  // Destructor
  ~ObjectArena(void) { FreeAll(); }
  // Allocate the memory
  T *Alloc(void) {
    if (m_uNumAbailable == 0) {
      int nAlloc = SJCMax((unsigned long)16,
			  (unsigned long)(65536 / sizeof(T)));
      m_Mems = (T *)AllocAligned(nAlloc * sizeof(T));
      m_uNumAbailable = nAlloc;
      m_VToDelete.push_back(m_Mems);
    }
    --m_uNumAbailable;
    return m_Mems++;
  }
  // Get a new memory
  operator T *() {
    return Alloc();
  }

  // Free all the m_Memsory
  void FreeAll(void) {
    for (u_int i = 0; i < m_VToDelete.size(); ++i)
      FreeAligned(m_VToDelete[i]);
    m_VToDelete.erase(m_VToDelete.begin(), m_VToDelete.end());
    m_uNumAbailable = 0;
  }
 private:
  // ObjectArena Private Data
  T           *m_Mems;                // The memory allocated
  uint         m_uNumAbailable;
  vector<T *>  m_VToDelete;
};

//*****************************************************************************
//
// *
//
//*****************************************************************************
class MemoryArena {
 public:
  // Constructor
  MemoryArena(u_int bs = 32768) {
    m_uBlockSize    = bs;
    m_uCurBlockPos  = 0;
    m_CurrentBlock = (char *)AllocAligned(m_uBlockSize);
  }

  // Destructor
  ~MemoryArena(void) {
    FreeAligned(m_CurrentBlock);
    for (u_int i = 0; i < m_VUsedBlocks.size(); ++i)
      FreeAligned(m_VUsedBlocks[i]);
    for (u_int i = 0; i < m_VAvailableBlocks.size(); ++i)
      FreeAligned(m_VAvailableBlocks[i]);
  }

  // Allocate the block
  void *Alloc(u_int sz) {
    // Round up _sz_ to minimum machine alignment
    sz = ((sz + 7) & (~7));
    if (m_uCurBlockPos + sz > m_uBlockSize) {
      // Get new block of memory for _MemoryArena_
      m_VUsedBlocks.push_back(m_CurrentBlock);
      if (m_VAvailableBlocks.size() && sz <= m_uBlockSize) {
	m_CurrentBlock = m_VAvailableBlocks.back();
	m_VAvailableBlocks.pop_back();
      }
      else
	m_CurrentBlock = (char *)AllocAligned(max(sz, m_uBlockSize));
      m_uCurBlockPos = 0;
    }

    void *ret = m_CurrentBlock + m_uCurBlockPos;
    m_uCurBlockPos += sz;
    return ret;
  }
  // Free all the memory allocated
  void FreeAll(void) {
    m_uCurBlockPos = 0;
    while (m_VUsedBlocks.size()) {
      m_VAvailableBlocks.push_back(m_VUsedBlocks.back());
      m_VUsedBlocks.pop_back();
    }
  }
 private:
  uint            m_uCurBlockPos;        // Indicate the current position
  uint            m_uBlockSize;
  char           *m_CurrentBlock;
  vector<char *>  m_VUsedBlocks;
  vector<char *>  m_VAvailableBlocks;
};

//*****************************************************************************
//
// * Set up a block array
//
//*****************************************************************************
template<class T, int logBlockSize> class BlockedArray 
{
 public:
  // Constructor
  BlockedArray(int nu, int nv, const T *d = NULL) {
    m_uURes    = nu;
    m_uVRes    = nv;
    m_uBlocks  = RoundUp(m_uURes) >> logBlockSize;
    int nAlloc = RoundUp(m_uURes) * RoundUp(m_uVRes);
    m_Data     = (T *)AllocAligned(nAlloc * sizeof(T));
    for (int i = 0; i < nAlloc; ++i)
      new (&m_Data[i]) T(); //???
    if (d)
      for (int v = 0; v < nv; ++v)
	for (int u = 0; u < nu; ++u)
	  (*this)(u, v) = d[v * m_uURes + u];
  }

  // Destructor
  ~BlockedArray(void) {
    for (int i = 0; i < m_uURes * m_uVRes; ++i)
      m_Data[i].~T();
    FreeAligned(m_Data);
  }

  // Get the size of the block
  uint BlockSize(void) const { return 1 << logBlockSize; }
  // Round up the size
  uint RoundUp(int x) const {
    return (x + BlockSize() - 1) & ~(BlockSize() - 1);
  }
  // Get the u size
  uint USize() const { return m_uURes; }
  // Get the v size
  uint VSize() const { return m_uVRes; }

  // 
  uint Block(int a) const { return a >> logBlockSize; }
  // 
  uint Offset(int a) const { return (a & (BlockSize() - 1)); }

  T &operator()(int u, int v) {
    uint bu     = Block(u),  bv = Block(v);
    uint ou     = Offset(u), ov = Offset(v);
    uint offset = BlockSize() * BlockSize() *
      (m_uBlocks * bv + bu);
    offset += BlockSize() * ov + ou;
    return m_Data[offset];
  }

  const T &operator()(uint u, uint v) const {
    uint bu     = Block(u),  bv = Block(v);
    uint ou     = Offset(u), ov = Offset(v);
    uint offset = BlockSize() * BlockSize() * (m_uBlocks * bv + bu);
    offset     += BlockSize() * ov + ou;
    return m_Data[offset];
  }

  void GetLinearArray(T *a) const {
    for (int v = 0; v < m_uVRes; ++v)
      for (int u = 0; u < m_uURes; ++u)
	*a++ = (*this)(u, v);
  }
 private:
  // BlockedArray Private M_Data 
  T    *m_Data;          // All data
  uint m_uURes;          // The block in U direction 
  uint m_uVRes;          // The block in V direction
  uint m_uBlocks;
};


#endif
