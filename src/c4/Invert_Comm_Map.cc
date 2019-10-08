//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   c4/Invert_Comm_Map.cc
 * \author Rob Lowrie
 * \date   Mon Nov 19 10:09:11 2007
 * \brief  Implementation of Invert_Comm_Map
 * \note   Copyright (C) 2016-2019 Triad National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include "Invert_Comm_Map.hh"
#include "MPI_Traits.hh"
#include "ds++/Assert.hh"
#include <vector>

namespace rtt_c4 {

//---------------------------------------------------------------------------//
// MPI version of get_num_recv()
#ifdef C4_MPI
int get_num_recv(Invert_Comm_Map_t::const_iterator first,
                 Invert_Comm_Map_t::const_iterator last) {
  const int my_proc = rtt_c4::node();
  Remember(const int num_procs = rtt_c4::nodes());
  const int one = 1;

  // pointer that will be allocated by MPI_Win_allocate
  int *num_recv_win;

  // Create the RMA memory windows for each value
  MPI_Win win;
  MPI_Info info;
  MPI_Info_create(&info);
  // let MPI know that the accumulate will be on a single value to enable
  // network atomics
  MPI_Info_set(info, "accumulate_ops", "same_op");
  MPI_Info_set(info, "same_size", "true");
  MPI_Info_set(info, "same_disp", "true");

  int mpi_int_size;
  MPI_Type_size(MPI_INT, &mpi_int_size);
  MPI_Win_allocate(mpi_int_size, mpi_int_size, info, MPI_COMM_WORLD, &num_recv_win, &win);

  *num_recv_win = 0;

  MPI_Barrier(MPI_COMM_WORLD);
  // use use MPI_MODE_NOCHECK because there will be no conflicting locks on this
  // window
  const int lock_assert=MPI_MODE_NOCHECK;

  MPI_Win_lock_all(lock_assert, win);

  // Accumulate the local and remote data values
  for (auto it = first; it != last; ++it) {
    Require(it->first >= 0);
    Require(it->first < num_procs);
    if (it->first != my_proc) { // treat only non-local sends
      // ...increment the remote number of receives
      MPI_Accumulate(&one, 1, MPI_Traits<int>::element_type(), it->first, 0, 1,
                     MPI_Traits<int>::element_type(), MPI_SUM, win);
    }
  }
  MPI_Win_unlock_all(win);
  MPI_Barrier(MPI_COMM_WORLD);

  // pull out value before deallocating
  int num_recv = *num_recv_win;
  MPI_Win_free(&win);
  MPI_Info_free(&info);
  Ensure(num_recv >= 0 && num_recv < num_procs);
  return num_recv;
}
//---------------------------------------------------------------------------//
// SCALAR version of get_num_recv
#elif defined(C4_SCALAR)
int get_num_recv(Invert_Comm_Map_t::const_iterator /*first*/,
                 Invert_Comm_Map_t::const_iterator /*last*/) {
  return 0;
}
#else
//---------------------------------------------------------------------------//
// Default version of get_num_recv, which throws an error.
int get_num_recv(Invert_Comm_Map_t::const_iterator /*first*/,
                 Invert_Comm_Map_t::const_iterator /*last*/) {
  Insist(0, "get_num_recv not implemented for this communication type!");
}
#endif // ifdef C4_MPI

//---------------------------------------------------------------------------//
void invert_comm_map(Invert_Comm_Map_t const &to_map,
                     Invert_Comm_Map_t &from_map) {
  const int my_proc = rtt_c4::node();
  Remember(const int num_procs = rtt_c4::nodes());

  // number of procs we will receive data from
  const int num_recv = get_num_recv(to_map.begin(), to_map.end());

  // request handle for the receives
  std::vector<C4_Req> recvs(num_recv);

  // the number of data elements to be received from each proc.  This data will
  // ultimately be loaded into from_map, once we know the sending proc ids.
  std::vector<size_t> sizes(num_recv);

  // communication tag for sends/recvs
  const int tag = 201;

  // Posts the receives for the data sizes.  We don't yet know the proc numbers
  // sending the data, so use any_source.
  for (int i = 0; i < num_recv; ++i) {
    receive_async(recvs[i], &sizes[i], 1, any_source, tag);
  }

  from_map.clear(); // empty whatever came in

  // Send the data sizes and take care of on-proc map.
  for (auto it = to_map.begin(); it != to_map.end(); ++it) {
    Require(it->first >= 0);
    Require(it->first < num_procs);
    Require(it->second > 0);
    if (it->first == my_proc) {
      Check(from_map.find(my_proc) == from_map.end());
      // on-proc map
      from_map[my_proc] = it->second;
    } else {
      // we can ignore the request returned, because our send buffers are
      // not shared, and we'll wait on the receives below.
      send_async(&(it->second), 1, it->first, tag);
    }
  }

  // Wait on the receives and populate the map
  C4_Status status;
  for (int i = 0; i < num_recv; ++i) {
    recvs[i].wait(&status);
    Check(status.get_message_size() == sizeof(size_t));
    const int proc = status.get_source();
    Check(proc >= 0);
    Check(proc < num_procs);

    // proc should not yet exist in map
    Check(from_map.find(proc) == from_map.end());

    from_map[proc] = sizes[i];
  }

  return;
}

} // end namespace rtt_c4

//---------------------------------------------------------------------------//
// end of Invert_Comm_Map.cc
//---------------------------------------------------------------------------//
