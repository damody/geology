/************************************************************************
     Main File:

     File:        SJCTimer.h

     Author:      
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
 
     Comment:     Record the timing information
	
     Contructor:
		  0 paras: identity 
     Function:
                 1. Start: set up the start information
                 2. Clear: clear all the data
                 3. Mark: mark the time
                 4. Restart: restart the data
                 5. TimeString: get the time string
                 6. << output operator

 
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef SJCTIMER_H_
#define SJCTIMER_H_

#include <SJC/SJC.h>

#include <vector>
#include <string>

#ifndef WIN32
#	include <sys/time.h>
#else
#	include <time.h>
#endif

class SJCTimer {

 private:
  timeval          m_StartTime;
  struct timezone  m_DumpZone;
  vector<string>   m_sVMarkMessages;   // The mark time message
  vector<timeval>  m_VMarkTimes;
    
 public:
  // Constructor
  SJCTimer(void){}
  ~SJCTimer(void){}
  
  // Set up the start
  void    Start(void) { gettimeofday(&m_StartTime, &m_DumpZone); }

  // Set up the clear
  void    Clear(void) {
    m_sVMarkMessages.clear();
    m_VMarkTimes.clear();
  }

  // Mark the time
  void    Mark(string& message){
    m_sVMarkMessages.push_back(message);
    timeval t;
    gettimeofday(&t, &m_DumpZone);
    m_VMarkTimes.push_back(t);
  }

  void    Mark(const char* cmessage){
    string message = cmessage;
    m_sVMarkMessages.push_back(message);
    timeval t;
    gettimeofday(&t, &m_DumpZone);
    m_VMarkTimes.push_back(t);
  }


  // Restart 
  void Restart(void){
    Clear();
    gettimeofday(&m_StartTime, &m_DumpZone);
  }

  // Get the time string
  string TimeString(timeval& start, timeval& end) {
    float duration = (float)(end.tv_sec - start.tv_sec) + 
                     (float)(end.tv_usec - start.tv_usec) * 0.000001f;
    char   buffer[200];
    sprintf(buffer, " Taking ,%8.2f s", duration);
    return string(buffer);
  }
   
  friend ostream& operator<<(ostream& o, SJCTimer& m){
    for(uint i = 0; i < m.m_sVMarkMessages.size(); i++){
      o << m.m_sVMarkMessages[i] 
	<< m.TimeString(m.m_StartTime, m.m_VMarkTimes[i]);
      if (i > 0)
	o << m.TimeString(m.m_VMarkTimes[i - 1], m.m_VMarkTimes[i]);
      o << "\n";
    }// end of for i
    return o;
  }

};


#endif

