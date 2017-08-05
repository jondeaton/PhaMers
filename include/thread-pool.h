/**
 * File: thread-pool.h
 * -------------------
 * This class defines the ThreadPool class, which accepts a collection
 * of thunks (which are zero-argument functions that don't return a value)
 * and schedules them in a FIFO manner to be executed by a constant number
 * of child threads that exist solely to invoke previously scheduled thunks.
 */

#ifndef _thread_pool_
#define _thread_pool_

#include <cstddef>     // for size_t
#include <functional>  // for the function template used in the schedule signature
#include <thread>      // for thread
#include <vector>      // for vector
#include <map>         // for map
#include <queue>       // for queue
#include "semaphore.h" // for semaphore
#include <mutex>
#include "condition_variable"

class ThreadPool {
public:

  /**
   * Constructor
   * -----------
   * Constructs a ThreadPool object configured to spawn up to the specified
   * number of threads.
   * @param numThreads : The number of threads in this thread pool
   */
  ThreadPool(size_t numThreads);

  /**
   * Public Method: schedule
   * -----------------------
   * Schedules the provided thunk (which is something that can
   * be invoked as a zero-argument function without a return value)
   * to be executed by one of the ThreadPool's threads as soon as
   * all previously scheduled thunks have been handled.
   * @param thunk : A function with no parameters and no return value
   * that will be scheduled for execution by a worker thread
   */
  void schedule(const std::function<void(void)>& thunk);

  /**
   * Public Method: wait
   * -------------------
   * This function blocks until all previously scheduled thunks have
   * been executed. This is accomplished by setting up a local
   * semaphore and then scheduling a thunk to signal that semaphore.
   * After being signaled, wait() knows that all thunks in the queue have been
   * dispatched, however some previously scheduled thunks may need to finish.
   * At this point, wait will halt dispatching and wait until all workers
   * become available.
   */
  void wait();

  /**
   * Deconstructor
   * -------------
   * Waits for all previously scheduled thunks to execute, and then
   * properly brings down the ThreadPool and any resources tapped
   * over the course of its lifetime.
   */
  ~ThreadPool();

private:
  std::thread dt;                // dispatcher thread handle
  std::vector<std::thread> wts;  // worker thread handles
  size_t numThreads;             // Number of worker threads

  typedef size_t id_t; // Type for storing Worker IDs

  // For keeping track of which workers are available
  std::map<id_t, bool> freeWorkerMap;
  std::mutex freeWorkerLock;

  // For indicating when all threads should exit
  bool threadsShouldExit;

  // Map of functions assigned to each worker ID
  std::map<id_t, std::function<void(void)>> workerFunctions;
  std::mutex funcitonMapLock;

  // Queue of scheduled functions
  std::queue<std::function<void(void)>> functionQueue;
  std::mutex queueLock;

  // For wait to know when tasks are done
  std::mutex workerFinished;
  std::condition_variable_any cv;

  std::mutex dispatchingLock; // For halting dispatching during (part of) a wait
  semaphore dispatchSignal; // Indicates dispatcher when task is scheduled

  // For signaling a worker to execute task
  std::map<id_t, std::unique_ptr<semaphore>> workerSignalMap;
  std::mutex workerSignalLock;

  /**
   * Private Method: dispatcher
   * --------------------------
   * This method will continuously wait until there is something in the
   * thunk queue and then dispatch that thunk to be executed by a worker
   */
  void dispatcher();

  /**
   * Private Method: worker
   * ---------------------
   * This method first waits until signaled by the workerSignalLock associated
   * with the ID, then it will retrieve the function to execute from the queue
   * and execute the funciton. This method will also indicate that it is available
   * after it has completed it's assigned task and notify the condition_variable_any
   * cv that it has finished.
   * This method is meant to be executed as a thread routine.
   * @param workerID : The ID of the worker that is
   */
  void worker(const id_t workerID);

  /**
   * ThreadPools are the type of thing that shouldn't be cloneable, since it's
   * not clear what it means to clone a ThreadPool (should copies of all outstanding
   * functions to be executed be copied?).
   *
   * In order to prevent cloning, we remove the copy constructor and the
   * assignment operator.  By doing so, the compiler will ensure we never clone
   * a ThreadPool.
   */
  ThreadPool(const ThreadPool& original) = delete;
  ThreadPool& operator=(const ThreadPool& rhs) = delete;
};

#endif
